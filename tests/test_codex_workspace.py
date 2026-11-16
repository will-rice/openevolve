"""The codex_app_server backend's sandbox, launcher and per-call workspace.

Every call here starts a real process: `codex_command` runs
tests/fake_codex_app_server.py, which logs what each turn saw.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from openevolve import OpenEvolve
from openevolve.config import Config, LLMModelConfig
from openevolve.controller import EvolutionFailedError
from openevolve.llm.codex_app_server import CodexAppServerLLM, CodexReplyFileError

FAKE = Path(__file__).parent / "fake_codex_app_server.py"


def model_config(log: Path, sleep: float = 0, **changes) -> LLMModelConfig:
    """A codex_app_server model whose app-server is the fake logging to `log`."""
    values = {
        "name": "gpt-test",
        "provider": "codex_app_server",
        "timeout": 30,
        "retries": 0,
        "retry_delay": 0,
        "codex_command": [sys.executable, str(FAKE), str(log), str(sleep), "no change"],
    }
    return LLMModelConfig(**{**values, **changes})


def turns(log: Path) -> list:
    """What each turn the fake answered saw, in order."""
    return [json.loads(line) for line in log.read_text().splitlines()]


class TestSandboxAndWorkspace(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        self.log = self.tmp / "turns.jsonl"
        self.root = self.tmp / "workspaces"
        self.root.mkdir()

    def test_read_only_runs_in_the_callers_cwd_without_files(self):
        llm = CodexAppServerLLM(model_config(self.log))
        reply = asyncio.run(llm.generate("p", workspace_files={"program.py": "x = 1\n"}))

        [turn] = turns(self.log)
        self.assertEqual(reply, "no change")
        self.assertEqual(turn["sandbox"], "read-only")
        self.assertIsNone(turn["thread_cwd"])
        self.assertEqual(turn["process_cwd"], os.getcwd())
        self.assertEqual(turn["files"], {})

    def test_each_call_gets_a_fresh_workspace_holding_its_files(self):
        """Concurrent calls run in distinct directories, each removed afterwards."""
        cfg = model_config(
            self.log,
            sleep=1,
            sandbox_mode="workspace-write",
            workspace_root=str(self.root),
        )
        llm = CodexAppServerLLM(cfg)

        async def both():
            return await asyncio.gather(
                llm.generate("a", workspace_files={"program.py": "a = 1\r\n"}),
                llm.generate("b", workspace_files={"program.py": "b = 2\n"}),
            )

        asyncio.run(both())

        seen = turns(self.log)
        self.assertEqual(len(seen), 2)
        self.assertEqual({t["sandbox"] for t in seen}, {"workspace-write"})
        cwds = {t["process_cwd"] for t in seen}
        self.assertEqual(len(cwds), 2)
        for turn in seen:
            self.assertEqual(turn["thread_cwd"], turn["process_cwd"])
            self.assertEqual(Path(turn["process_cwd"]).parent, self.root)
        self.assertEqual(sorted(t["files"]["program.py"] for t in seen), ["a = 1\r\n", "b = 2\n"])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_a_failed_call_removes_its_workspace(self):
        cfg = model_config(
            self.log,
            sandbox_mode="workspace-write",
            workspace_root=str(self.root),
            codex_command=[sys.executable, "-c", "raise SystemExit(1)"],
        )
        llm = CodexAppServerLLM(cfg)

        with self.assertRaises(RuntimeError):
            asyncio.run(llm.generate("p", workspace_files={"program.py": "x"}))

        self.assertEqual(list(self.root.iterdir()), [])

    def test_a_writable_sandbox_needs_a_workspace_root(self):
        with self.assertRaisesRegex(ValueError, "workspace_root"):
            CodexAppServerLLM(model_config(self.log, sandbox_mode="workspace-write"))

    def test_an_unknown_sandbox_mode_is_refused(self):
        with self.assertRaisesRegex(ValueError, "sandbox_mode"):
            CodexAppServerLLM(model_config(self.log, sandbox_mode="read_only"))


class TestReplyFile(unittest.TestCase):
    """With reply_file, the completed turn's file is the reply, not the message."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        self.log = self.tmp / "turns.jsonl"
        self.root = self.tmp / "workspaces"
        self.root.mkdir()

    def editing(self, program: str, **changes) -> CodexAppServerLLM:
        """A writable provider whose fake writes `program` to program.py, replying "done"."""
        command = [sys.executable, str(FAKE), str(self.log), str(changes.pop("sleep", 0))]
        values = {
            "codex_command": [*command, "done", program],
            "sandbox_mode": "workspace-write",
            "workspace_root": str(self.root),
            "reply_file": "program.py",
        }
        return CodexAppServerLLM(model_config(self.log, **{**values, **changes}))

    def test_the_edited_file_is_the_reply_byte_for_byte(self):
        llm = self.editing("x = 2\r\ny = 3\n")

        reply = asyncio.run(llm.generate("p", workspace_files={"program.py": "x = 1\n"}))

        [turn] = turns(self.log)
        self.assertEqual(turn["files"]["program.py"], "x = 1\n")
        self.assertEqual(reply, "x = 2\r\ny = 3\n")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_a_file_the_turn_removed_fails_without_a_retry(self):
        llm = self.editing("-", retries=1)

        with self.assertRaisesRegex(CodexReplyFileError, "left no program.py"):
            asyncio.run(llm.generate("p", workspace_files={"program.py": "x = 1\n"}))

        self.assertEqual(len(turns(self.log)), 1)

    def test_a_file_holding_a_code_fence_fails(self):
        """The full-rewrite parser would take the fence as the program's end."""
        llm = self.editing("x = 1\n# ```\ny = 2\n")

        with self.assertRaisesRegex(CodexReplyFileError, "code fence"):
            asyncio.run(llm.generate("p", workspace_files={"program.py": "x = 1\n"}))

    def test_a_turn_that_times_out_mid_edit_returns_no_file(self):
        """The fake has written program.py when the turn runs out of time."""
        llm = self.editing("half = ", sleep=5, timeout=1)

        with self.assertRaises(Exception) as raised:
            asyncio.run(llm.generate("p", workspace_files={"program.py": "x = 1\n"}))

        self.assertNotIsInstance(raised.exception, CodexReplyFileError)
        self.assertNotIn("half = ", str(raised.exception))

    def test_a_read_only_sandbox_cannot_reply_with_a_file(self):
        with self.assertRaisesRegex(ValueError, "writable"):
            CodexAppServerLLM(model_config(self.log, reply_file="program.py"))


# The program the run evolves, and the codex reply that edits it: a SEARCH/REPLACE
# diff, as OpenEvolve's default diff-based evolution asks for.
INITIAL_PROGRAM = "def f():\n    return 0\n"
DIFF_REPLY = "<<<<<<< SEARCH\n    return 0\n=======\n    return 1\n>>>>>>> REPLACE\n"
EVALUATOR = "def evaluate(program_path):\n    return {'combined_score': 0.5}\n"


class TestRunWorkspace(unittest.TestCase):
    """A real OpenEvolve run: its worker process gets the codex options and the parent."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)

    def test_the_parent_program_reaches_codex_in_a_worker(self):
        """The codex options survive the config's trip to a worker process (asdict,
        then LLMModelConfig(**m)), and the call's workspace holds the parent."""
        log, root = self.tmp / "turns.jsonl", self.tmp / "workspaces"
        root.mkdir()
        (self.tmp / "initial_program.py").write_text(INITIAL_PROGRAM)
        (self.tmp / "evaluator.py").write_text(EVALUATOR)
        config = Config()
        config.evaluator.parallel_evaluations = 1
        config.database.num_islands = 1
        config.database.in_memory = True
        command = [sys.executable, str(FAKE), str(log), "0", DIFF_REPLY]
        config.llm.models = [
            model_config(
                log,
                codex_command=command,
                sandbox_mode="workspace-write",
                workspace_root=str(root),
            )
        ]
        config.llm.evaluator_models = config.llm.models
        run = OpenEvolve(
            initial_program_path=str(self.tmp / "initial_program.py"),
            evaluation_file=str(self.tmp / "evaluator.py"),
            config=config,
            output_dir=str(self.tmp / "out"),
        )

        best = asyncio.run(run.run(iterations=1))

        [turn] = turns(log)
        [seed_id] = [p.id for p in run.database.programs.values() if p.parent_id is None]
        self.assertEqual(turn["sandbox"], "workspace-write")
        self.assertEqual(
            turn["files"],
            {"program.py": INITIAL_PROGRAM, f"{seed_id}.py": INITIAL_PROGRAM},
        )
        self.assertNotEqual(turn["process_cwd"], os.getcwd())
        self.assertEqual(list(root.iterdir()), [])
        self.assertIsNotNone(best)

    def test_a_template_names_the_workspace_file_of_each_program_it_shows(self):
        """A top_program template that shows `./{program_id}.py`, not the code, names
        a file the call's workspace holds, with that program's code."""
        log, root, templates = (self.tmp / n for n in ("turns.jsonl", "ws", "prompts"))
        root.mkdir()
        templates.mkdir()
        (templates / "top_program.txt").write_text(
            "### Program {program_number} (Score: {score}): ./{program_id}.py"
        )
        (self.tmp / "initial_program.py").write_text(INITIAL_PROGRAM)
        (self.tmp / "evaluator.py").write_text(EVALUATOR)
        config = Config()
        config.evaluator.parallel_evaluations = 1
        config.database.num_islands = 1
        config.database.in_memory = True
        config.prompt.template_dir = str(templates)
        command = [sys.executable, str(FAKE), str(log), "0", DIFF_REPLY]
        config.llm.models = [
            model_config(
                log,
                codex_command=command,
                sandbox_mode="workspace-write",
                workspace_root=str(root),
            )
        ]
        config.llm.evaluator_models = config.llm.models
        run = OpenEvolve(
            initial_program_path=str(self.tmp / "initial_program.py"),
            evaluation_file=str(self.tmp / "evaluator.py"),
            config=config,
            output_dir=str(self.tmp / "out"),
        )

        asyncio.run(run.run(iterations=1))

        [turn] = turns(log)
        [seed_id] = [p.id for p in run.database.programs.values() if p.parent_id is None]
        self.assertIn(f"### Program 1 (Score: 0.5000): ./{seed_id}.py", turn["prompt"])
        self.assertEqual(turn["files"][f"{seed_id}.py"], INITIAL_PROGRAM)


class TestRunEditsInPlace(unittest.TestCase):
    """A real OpenEvolve run with full rewrites whose reply is the edited program.py."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)

    def run_editing(self, program: str) -> OpenEvolve:
        """One iteration whose codex writes `program` to program.py.

        Raises:
            EvolutionFailedError: The iteration produced no child.
        """
        log, root = self.tmp / "turns.jsonl", self.tmp / "workspaces"
        root.mkdir()
        (self.tmp / "initial_program.py").write_text(INITIAL_PROGRAM)
        (self.tmp / "evaluator.py").write_text(EVALUATOR)
        config = Config()
        config.diff_based_evolution = False
        config.evaluator.parallel_evaluations = 1
        config.database.num_islands = 1
        config.database.in_memory = True
        command = [sys.executable, str(FAKE), str(log), "0", "done", program]
        config.llm.models = [
            model_config(
                log,
                codex_command=command,
                sandbox_mode="workspace-write",
                workspace_root=str(root),
                reply_file="program.py",
            )
        ]
        config.llm.evaluator_models = config.llm.models
        run = OpenEvolve(
            initial_program_path=str(self.tmp / "initial_program.py"),
            evaluation_file=str(self.tmp / "evaluator.py"),
            config=config,
            output_dir=str(self.tmp / "out"),
        )
        asyncio.run(run.run(iterations=1))
        return run

    def test_the_edited_file_is_the_child(self):
        edited = "def f():\n    return 1\n"

        run = self.run_editing(edited)

        [child] = [p for p in run.database.programs.values() if p.parent_id is not None]
        self.assertEqual(child.code, edited)
        self.assertEqual(child.metadata["evolution_mode"], "full_rewrite")

    def test_an_unchanged_file_fails_the_iteration(self):
        """No silent fallback: the parent is not submitted again as its own child."""
        with self.assertRaisesRegex(EvolutionFailedError, "identical to the parent"):
            self.run_editing(INITIAL_PROGRAM)


if __name__ == "__main__":
    unittest.main()
