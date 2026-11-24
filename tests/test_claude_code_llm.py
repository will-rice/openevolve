"""Tests for the Claude Code CLI LLM backend.

Every call starts a real process: `claude_command` runs tests/fake_claude_code.py, which
logs what it was given and prints a recorded run's stream-json:
claude_code_edit_turn.jsonl, Claude Code 2.1.321 on deepseek-v4.1-flash behind an
Anthropic-compatible endpoint editing program.py (2026-11-20), and
claude_code_rate_limited_turn.jsonl, a run there that ended on the endpoint's 429, and
claude_code_refused_turn.jsonl, Claude Code 2.1.321 in dontAsk mode with
permissions.blockReadsOutsideWorkingDirectories and an MCP server `qcc`, on a local
Anthropic Messages server answering a fixed script (2026-11-22): a 429 it retried, a
Bash call its permission layer refused, a Read outside its working directory, a Bash
`cat` of a missing file, an MCP call that returned an error, a Read of a missing file,
an Edit of program.py, an automatic compaction, then the final message.
"""

import asyncio
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

from openevolve.config import Config, LLMModelConfig
from openevolve.llm.claude_code import ClaudeCodeLLM, init_claude_code_client

FAKE = Path(__file__).parent / "fake_claude_code.py"
EDIT_TURN = Path(__file__).parent / "claude_code_edit_turn.jsonl"
RATE_LIMITED_TURN = Path(__file__).parent / "claude_code_rate_limited_turn.jsonl"
# A run there stopped by `--max-turns 2` after it edited program.py (2026-11-21).
MAX_TURNS_TURN = Path(__file__).parent / "claude_code_max_turns_turn.jsonl"
REFUSED_TURN = Path(__file__).parent / "claude_code_refused_turn.jsonl"
# The edit turn's final message.
REPLY = "`program.py` now returns `x+2`. Output of the command:\n\n```\n3\n```"


def model_config(log: Path, stream: Path, sleep: float = 0, *program: str, **changes):
    """A claude_code model whose `claude` is the fake logging to `log`."""
    values = {
        "name": "deepseek-v4.1-flash",
        "provider": "claude_code",
        "timeout": 30,
        "retries": 0,
        "retry_delay": 0,
        "claude_command": [sys.executable, str(FAKE), str(log), str(stream), str(sleep), *program],
    }
    return LLMModelConfig(**{**values, **changes})


def runs(log: Path) -> list:
    """What each run the fake answered saw, in order."""
    return [json.loads(line) for line in log.read_text().splitlines()]


class TestClaudeCodeLLM(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        self.log = self.tmp / "runs.jsonl"
        self.root = self.tmp / "workspaces"
        self.root.mkdir()

    def test_the_reply_is_the_runs_final_message(self):
        """The system message and prompt go on stdin; the reply is the result's text."""
        llm = ClaudeCodeLLM(model_config(self.log, EDIT_TURN, claude_args=["--tools", "Read"]))

        reply = asyncio.run(llm.generate("p", system_message="s"))

        [run] = runs(self.log)
        self.assertEqual(reply, REPLY)
        self.assertEqual(llm.last_message, REPLY)
        self.assertEqual(llm.last_backend, "claude-code/deepseek-v4.1-flash")
        self.assertEqual(run["prompt"], "s\n\np")
        self.assertEqual(
            run["args"],
            [
                *("-p", "--output-format", "stream-json", "--verbose"),
                *("--model", "deepseek-v4.1-flash", "--tools", "Read"),
            ],
        )

    def test_a_writable_runs_reply_is_the_file_it_left(self):
        """The run sees its workspace files; the reply is reply_file as it left it."""
        llm = ClaudeCodeLLM(
            model_config(
                self.log,
                EDIT_TURN,
                0,
                "def f(x):\n    return x+2\n",
                sandbox_mode="workspace-write",
                workspace_root=str(self.root),
                reply_file="program.py",
            )
        )

        reply = asyncio.run(
            llm.generate("p", workspace_files={"program.py": "def f(x):\n    return x+1\n"})
        )

        [run] = runs(self.log)
        self.assertEqual(reply, "def f(x):\n    return x+2\n")
        self.assertEqual(llm.last_message, REPLY)
        self.assertEqual(run["files"], {"program.py": "def f(x):\n    return x+1\n"})
        self.assertEqual(Path(run["cwd"]).parent, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_a_run_that_ends_on_an_error_fails_and_is_retried(self):
        """A result with is_error (subtype success here) fails with its text; a retry reruns."""
        llm = ClaudeCodeLLM(model_config(self.log, RATE_LIMITED_TURN, retries=1))

        with self.assertRaisesRegex(
            RuntimeError, r"failed: success: API Error: Request rejected \(429\)"
        ):
            asyncio.run(llm.generate("p"))

        self.assertEqual(len(runs(self.log)), 2)

    def test_a_writable_run_stopped_at_its_turn_limit_replies_with_its_file(self):
        """The file the stopped run left is the reply; without reply_file it fails."""
        program = "def f(x):\n    return x+2\n"
        writable = ClaudeCodeLLM(
            model_config(
                self.log,
                MAX_TURNS_TURN,
                0,
                program,
                sandbox_mode="workspace-write",
                workspace_root=str(self.root),
                reply_file="program.py",
            )
        )

        reply = asyncio.run(writable.generate("p", workspace_files={"program.py": "x\n"}))

        self.assertEqual(reply, program)
        self.assertIn("Reached maximum number of turns (2)", writable.last_message)
        text_only = ClaudeCodeLLM(model_config(self.log, MAX_TURNS_TURN))
        with self.assertRaisesRegex(RuntimeError, "failed: error_max_turns"):
            asyncio.run(text_only.generate("p"))

    def test_a_run_without_progress_times_out_at_the_idle_limit(self):
        """Events restart idle_timeout; a run silent for longer fails."""
        progressing = ClaudeCodeLLM(
            model_config(self.log, EDIT_TURN, 0.2, timeout=1, idle_timeout=1)
        )
        self.assertEqual(asyncio.run(progressing.generate("p")), REPLY)

        silent = ClaudeCodeLLM(model_config(self.log, EDIT_TURN, 2, timeout=1, idle_timeout=1))
        with self.assertRaisesRegex(asyncio.TimeoutError, "made no progress for 1s"):
            asyncio.run(silent.generate("p"))

    def test_a_turns_report_counts_its_requests_tools_refusals_and_retries(self):
        """The report comes from the stream; it is logged and appended to the file."""
        report_file = self.tmp / "turn_reports.jsonl"
        llm = ClaudeCodeLLM(
            model_config(
                self.log,
                REFUSED_TURN,
                0,
                "def f(x):\n    return x+2\n",
                sandbox_mode="workspace-write",
                workspace_root=str(self.root),
                reply_file="program.py",
                turn_report_file=str(report_file),
            )
        )

        with self.assertLogs("openevolve.llm.claude_code", logging.INFO) as logs:
            asyncio.run(llm.generate("p", workspace_files={"program.py": "x\n"}))

        report = llm.last_turn_report
        self.assertEqual(report.agent, "claude-code")
        self.assertEqual(report.backend, "claude-code/deepseek-v4.1-flash")
        self.assertEqual(report.requests, 7)
        self.assertEqual(report.compactions, 1)
        self.assertEqual(report.api_retries, {"429": 1})
        self.assertEqual(report.tool_calls, {"Bash": 2, "Read": 2, "mcp__qcc__check": 1, "Edit": 1})
        failures = report.tool_failures
        self.assertEqual(
            {reason: failures[reason].count for reason in failures},
            {"permission": 1, "outside_workdirs": 1, "missing_file": 1, "other": 2},
        )
        [refused] = failures["permission"].examples
        self.assertEqual(refused.tool, "Bash")
        self.assertIn('"command": "ls /"', refused.input)
        self.assertTrue(refused.error.startswith("Permission to use Bash has been denied"))
        self.assertEqual(len(refused.error), 160)
        self.assertEqual(failures["outside_workdirs"].examples[0].tool, "Read")
        self.assertIn(
            "missing_input.txt: No such file", failures["missing_file"].examples[0].error
        )
        self.assertEqual(
            [example.tool for example in failures["other"].examples], ["mcp__qcc__check", "Read"]
        )
        self.assertTrue(report.reply_changed)
        self.assertEqual(report.end, "completed")
        self.assertIsNone(report.error)
        self.assertIn(f"turn report: {report.to_json()}", logs.output[-1])
        self.assertEqual(report_file.read_text(), report.to_json() + "\n")

    def test_a_turn_that_left_its_reply_file_reports_it_unchanged(self):
        """The fake writes no program: the reply file is as the call wrote it."""
        llm = ClaudeCodeLLM(
            model_config(
                self.log,
                EDIT_TURN,
                sandbox_mode="workspace-write",
                workspace_root=str(self.root),
                reply_file="program.py",
            )
        )

        reply = asyncio.run(llm.generate("p", workspace_files={"program.py": "x\n"}))

        report = llm.last_turn_report
        self.assertEqual(reply, "x\n")
        self.assertFalse(report.reply_changed)
        self.assertEqual(report.requests, 4)
        self.assertEqual(report.tool_calls, {"Read": 1, "Edit": 1, "Bash": 1})
        self.assertEqual(report.tool_failures, {})
        self.assertEqual(report.end, "completed")

    def test_a_failed_turns_report_is_set_and_written_before_it_raises(self):
        """Each attempt, retried or not, writes its report; the last is on the LLM."""
        report_file = self.tmp / "turn_reports.jsonl"
        llm = ClaudeCodeLLM(
            model_config(self.log, RATE_LIMITED_TURN, retries=1, turn_report_file=str(report_file))
        )

        with self.assertRaises(RuntimeError):
            asyncio.run(llm.generate("p"))

        report = llm.last_turn_report
        self.assertEqual(report.end, "failed")
        self.assertTrue(report.error.startswith("success: API Error: Request rejected (429)"))
        self.assertEqual(report.api_retries, {"429": 1})
        # The run's own "<synthetic>" message for the failed request is not a request.
        self.assertEqual(report.requests, 1)
        self.assertIsNone(report.reply_changed)
        lines = report_file.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[-1], report.to_json())

    def test_a_turn_stopped_at_its_turn_limit_reports_max_turns(self):
        """With or without reply_file, the end is max_turns."""
        writable = ClaudeCodeLLM(
            model_config(
                self.log,
                MAX_TURNS_TURN,
                0,
                "def f(x):\n    return x+2\n",
                sandbox_mode="workspace-write",
                workspace_root=str(self.root),
                reply_file="program.py",
            )
        )
        asyncio.run(writable.generate("p", workspace_files={"program.py": "x\n"}))
        self.assertEqual(writable.last_turn_report.end, "max_turns")
        self.assertTrue(writable.last_turn_report.reply_changed)
        self.assertEqual(writable.last_turn_report.requests, 2)

        text_only = ClaudeCodeLLM(model_config(self.log, MAX_TURNS_TURN))
        with self.assertRaises(RuntimeError):
            asyncio.run(text_only.generate("p"))
        self.assertEqual(text_only.last_turn_report.end, "max_turns")
        self.assertIn("Reached maximum number of turns (2)", text_only.last_turn_report.error)

    def test_a_writable_run_past_its_turn_timeout_replies_with_its_file(self):
        """Past turn_timeout a progressing run is stopped; the file it left is the reply.

        Events every 0.4 s keep the idle timeout from firing; the 1 s turn timeout ends
        the run. Without reply_file the timeout fails the call.
        """
        program = "def f(x):\n    return x+3\n"
        writable = ClaudeCodeLLM(
            model_config(
                self.log, EDIT_TURN, 0.4, program,
                timeout=30, idle_timeout=30, turn_timeout=1,
                sandbox_mode="workspace-write", workspace_root=str(self.root),
                reply_file="program.py",
            )
        )  # fmt: skip

        reply = asyncio.run(writable.generate("p", workspace_files={"program.py": "x\n"}))

        self.assertEqual(reply, program)
        self.assertIn("turn timeout", writable.last_message)
        self.assertEqual(writable.last_turn_report.end, "timeout")
        text_only = ClaudeCodeLLM(
            model_config(self.log, EDIT_TURN, 0.4, timeout=30, idle_timeout=30, turn_timeout=1)
        )
        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(text_only.generate("p"))

    def test_a_turn_without_progress_reports_its_idle_timeout(self):
        silent = ClaudeCodeLLM(model_config(self.log, EDIT_TURN, 2, timeout=1, idle_timeout=1))
        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(silent.generate("p"))
        self.assertEqual(silent.last_turn_report.end, "idle_timeout")
        self.assertGreaterEqual(silent.last_turn_report.wall_seconds, 1)

    def test_factory_function(self):
        llm = init_claude_code_client(LLMModelConfig(name="sonnet"))
        self.assertIsInstance(llm, ClaudeCodeLLM)
        self.assertEqual(llm.model, "sonnet")


class TestMaxBudgetConfig(unittest.TestCase):
    def test_max_budget_usd_in_model_config(self):
        cfg = LLMModelConfig(max_budget_usd=2.5)
        self.assertEqual(cfg.max_budget_usd, 2.5)

    def test_max_budget_usd_default_none(self):
        cfg = LLMModelConfig()
        self.assertIsNone(cfg.max_budget_usd)

    def test_max_budget_usd_from_dict(self):
        config = Config.from_dict(
            {
                "llm": {
                    "provider": "claude_code",
                    "models": [{"name": "sonnet", "max_budget_usd": 3.0, "weight": 1.0}],
                }
            }
        )
        self.assertEqual(config.llm.models[0].max_budget_usd, 3.0)

    def test_max_budget_usd_reaches_the_command(self):
        llm = ClaudeCodeLLM(LLMModelConfig(name="sonnet", max_budget_usd=2.5))
        command = llm._build_command()
        self.assertEqual(command[command.index("--max-budget-usd") + 1], "2.5")


class TestProviderRegistry(unittest.TestCase):
    def test_claude_code_in_registry(self):
        from openevolve.llm.ensemble import _PROVIDER_REGISTRY

        self.assertIn("claude_code", _PROVIDER_REGISTRY)

    def test_ensemble_creates_claude_code(self):
        from openevolve.llm.ensemble import _create_model

        model = _create_model(LLMModelConfig(name="sonnet", provider="claude_code"))
        self.assertIsInstance(model, ClaudeCodeLLM)


if __name__ == "__main__":
    unittest.main()
