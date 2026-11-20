"""Tests for the Claude Code CLI LLM backend.

Every call starts a real process: `claude_command` runs tests/fake_claude_code.py, which
logs what it was given and prints a recorded run's stream-json:
claude_code_edit_turn.jsonl, Claude Code 2.1.321 on deepseek-v4.1-flash behind an
Anthropic-compatible endpoint editing program.py (2026-11-20), and
claude_code_rate_limited_turn.jsonl, a run there that ended on the endpoint's 429.
"""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from openevolve.config import Config, LLMModelConfig
from openevolve.llm.claude_code import ClaudeCodeLLM, init_claude_code_client

FAKE = Path(__file__).parent / "fake_claude_code.py"
EDIT_TURN = Path(__file__).parent / "claude_code_edit_turn.jsonl"
RATE_LIMITED_TURN = Path(__file__).parent / "claude_code_rate_limited_turn.jsonl"
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

    def test_a_run_without_progress_times_out_at_the_idle_limit(self):
        """Events restart idle_timeout; a run silent for longer fails."""
        progressing = ClaudeCodeLLM(
            model_config(self.log, EDIT_TURN, 0.2, timeout=1, idle_timeout=1)
        )
        self.assertEqual(asyncio.run(progressing.generate("p")), REPLY)

        silent = ClaudeCodeLLM(model_config(self.log, EDIT_TURN, 2, timeout=1, idle_timeout=1))
        with self.assertRaisesRegex(asyncio.TimeoutError, "made no progress for 1s"):
            asyncio.run(silent.generate("p"))

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
