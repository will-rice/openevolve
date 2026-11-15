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
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

from openevolve import process_parallel as process_parallel_module
from openevolve.config import Config, LLMModelConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.llm.codex_app_server import CodexAppServerLLM
from openevolve.process_parallel import ProcessParallelController

FAKE = Path(__file__).parent / "fake_codex_app_server.py"
# Worker globals _worker_init sets; restored after the test that sets them.
WORKER_GLOBALS = (
    "_worker_config",
    "_worker_evaluation_file",
    "_worker_evaluator",
    "_worker_llm_ensemble",
    "_worker_prompt_sampler",
)
MISSING = object()


def model_config(log: Path, sleep: float = 0, **changes) -> LLMModelConfig:
    """A codex_app_server model whose app-server is the fake logging to `log`."""
    values = {
        "name": "gpt-test",
        "provider": "codex_app_server",
        "timeout": 30,
        "retries": 0,
        "retry_delay": 0,
        "codex_command": [sys.executable, str(FAKE), str(log), str(sleep)],
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


class TestWorkerWorkspace(unittest.TestCase):
    """A worker process builds its model from the shipped config and passes the parent."""

    def setUp(self):
        saved = {name: getattr(process_parallel_module, name, MISSING) for name in WORKER_GLOBALS}

        def restore():
            for name, value in saved.items():
                if value is MISSING:
                    if hasattr(process_parallel_module, name):
                        delattr(process_parallel_module, name)
                else:
                    setattr(process_parallel_module, name, value)

        self.addCleanup(restore)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)

    def test_the_parent_program_reaches_codex_in_a_worker(self):
        """The codex options survive the worker's config round trip (asdict, then
        LLMModelConfig(**m)), and the call's workspace holds the parent program."""
        log, root = self.tmp / "turns.jsonl", self.tmp / "workspaces"
        root.mkdir()
        config = Config()
        model = model_config(log, sandbox_mode="workspace-write", workspace_root=str(root))
        config.llm.models = [model]
        config.llm.evaluator_models = [model]
        self.assertEqual(LLMModelConfig(**asdict(model)), model)
        controller = ProcessParallelController(
            config, "evaluator.py", ProgramDatabase(config.database)
        )
        config_dict = controller._serialize_config(config)
        parent = Program(id="parent", code="x = 1\n", metrics={}, metadata={"island": 0})
        snapshot = {
            "programs": {"parent": parent.to_dict()},
            "artifacts": {},
            "islands": [["parent"]],
            "current_island": 0,
            "sampling_island": 0,
            "feature_dimensions": [],
        }

        process_parallel_module._worker_init(config_dict, "evaluator.py", None)
        process_parallel_module._worker_prompt_sampler = MagicMock()
        process_parallel_module._worker_prompt_sampler.build_prompt.return_value = {
            "system": "sys",
            "user": "usr",
        }
        process_parallel_module._worker_evaluator = MagicMock()
        process_parallel_module._run_iteration_worker(1, snapshot, "parent", [])

        [turn] = turns(log)
        self.assertEqual(turn["sandbox"], "workspace-write")
        self.assertEqual(turn["files"], {"program.py": "x = 1\n"})
        self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
