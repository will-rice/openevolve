"""The codex_app_server backend's TurnReport, built from a turn's notifications.

Each call starts a real process: `codex_command` runs tests/fake_codex_app_server.py,
which ends the turn with a recorded turn's notifications. codex_tool_calls_turn.jsonl is
codex 0.161.0's app-server on a local Responses API server answering a fixed script, in
workspace-write with approvalPolicy "never" and an MCP server `qcc` (2026-11-22): a
command that read a missing file, a `touch /usr/...` its sandbox refused (which left no
item), an MCP call codex refused for want of approval, a command that exited 3, an
apply_patch of program.py, an automatic compaction and the final message, from 7 model
responses. The others are real turns on gpt-6.1-sol: codex_retried_turn.jsonl (an error
codex retried, then the reply), codex_refused_bounded_turn.jsonl (a retried error, then
one that ended the turn) and codex_web_search_turn.jsonl (one web search).
"""

import asyncio
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

from openevolve.config import LLMModelConfig
from openevolve.llm.codex_app_server import CodexAppServerLLM

TESTS = Path(__file__).parent
FAKE = TESTS / "fake_codex_app_server.py"
TOOL_CALLS_TURN = TESTS / "codex_tool_calls_turn.jsonl"
RETRIED_TURN = TESTS / "codex_retried_turn.jsonl"
REFUSED_BOUNDED_TURN = TESTS / "codex_refused_bounded_turn.jsonl"
WEB_SEARCH_TURN = TESTS / "codex_web_search_turn.jsonl"


class TestTurnReport(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        self.root = self.tmp / "workspaces"
        self.root.mkdir()

    def llm(self, reply, *program, pace=0.0, **changes) -> CodexAppServerLLM:
        """A codex_app_server model whose app-server is the fake answering with `reply`."""
        command = [sys.executable, str(FAKE), str(self.tmp / "turns.jsonl"), str(pace)]
        values = {
            "name": "gpt-test",
            "timeout": 30,
            "retries": 0,
            "retry_delay": 0,
            "codex_command": [*command, str(reply), *program],
        }
        return CodexAppServerLLM(LLMModelConfig(**{**values, **changes}))

    def test_a_turns_report_counts_its_responses_tools_and_failures(self):
        """The report comes from the notifications; it is logged and appended to the file."""
        report_file = self.tmp / "turn_reports.jsonl"
        llm = self.llm(
            TOOL_CALLS_TURN,
            "def f(x):\n    return x+2\n",
            sandbox_mode="workspace-write",
            workspace_root=str(self.root),
            reply_file="program.py",
            turn_report_file=str(report_file),
        )

        with self.assertLogs("openevolve.llm.codex_app_server", logging.INFO) as logs:
            asyncio.run(llm.generate("p", workspace_files={"program.py": "x\n"}))

        report = llm.last_turn_report
        self.assertEqual(report.agent, "codex")
        self.assertEqual(report.backend, "openai/gpt-test")
        # Nine token usage updates, seven of them raising the total: one per response.
        self.assertEqual(report.requests, 7)
        self.assertEqual(report.compactions, 1)
        self.assertEqual(report.api_retries, {})
        self.assertEqual(
            report.tool_calls, {"commandExecution": 2, "mcp__qcc__check": 1, "fileChange": 1}
        )
        failures = report.tool_failures
        self.assertEqual(
            {reason: failures[reason].count for reason in failures},
            {"missing_file": 1, "permission": 1, "other": 1},
        )
        [missing] = failures["missing_file"].examples
        self.assertEqual(missing.input, "/bin/bash -lc 'cat /nonexistent_qcc_probe'")
        self.assertIn("No such file or directory", missing.error)
        [refused] = failures["permission"].examples
        self.assertEqual(refused.tool, "mcp__qcc__check")
        self.assertEqual(refused.input, '{"path": "program.py"}')
        [exited] = failures["other"].examples
        self.assertEqual(exited.error, "exit code 3")
        self.assertTrue(report.reply_changed)
        self.assertEqual(report.end, "completed")
        self.assertIn(f"turn report: {report.to_json()}", logs.output[-1])
        self.assertEqual(json.loads(report_file.read_text()), json.loads(report.to_json()))

    def test_a_turn_that_left_its_reply_file_reports_it_unchanged(self):
        llm = self.llm(
            "no change",
            sandbox_mode="workspace-write",
            workspace_root=str(self.root),
            reply_file="program.py",
        )

        asyncio.run(llm.generate("p", workspace_files={"program.py": "x\n"}))

        self.assertFalse(llm.last_turn_report.reply_changed)
        self.assertEqual(llm.last_turn_report.end, "completed")

    def test_errors_codex_retries_are_counted_by_kind(self):
        llm = self.llm(RETRIED_TURN)

        asyncio.run(llm.generate("p"))

        self.assertEqual(llm.last_turn_report.api_retries, {"responseStreamDisconnected": 1})
        self.assertIsNone(llm.last_turn_report.reply_changed)

    def test_a_web_search_is_a_tool_call(self):
        llm = self.llm(WEB_SEARCH_TURN, web_search="live")

        asyncio.run(llm.generate("p"))

        self.assertEqual(llm.last_turn_report.tool_calls, {"webSearch": 1})
        self.assertEqual(llm.last_turn_report.tool_failures, {})

    def test_a_failed_turns_report_is_set_and_written_before_it_raises(self):
        report_file = self.tmp / "turn_reports.jsonl"
        llm = self.llm(REFUSED_BOUNDED_TURN, turn_report_file=str(report_file))

        with self.assertRaisesRegex(RuntimeError, "reported an error"):
            asyncio.run(llm.generate("p"))

        report = llm.last_turn_report
        self.assertEqual(report.end, "failed")
        self.assertEqual(report.error, "Connection failed: error sending request")
        self.assertEqual(report.api_retries, {"responseStreamDisconnected": 1})
        self.assertEqual(report_file.read_text(), report.to_json() + "\n")

    def test_a_turn_past_its_timeout_reports_it(self):
        """The fake sleeps 2s before it answers the turn, past the call's 1s timeout."""
        llm = self.llm("late", pace=2, timeout=1)

        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(llm.generate("p"))

        self.assertEqual(llm.last_turn_report.end, "timeout")
        self.assertGreaterEqual(llm.last_turn_report.wall_seconds, 1)

    def test_a_turn_without_progress_reports_its_idle_timeout(self):
        """Each notification comes 2s after the last, past the 1s idle_timeout."""
        llm = self.llm(WEB_SEARCH_TURN, pace=2, timeout=1, idle_timeout=1)

        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(llm.generate("p"))

        self.assertEqual(llm.last_turn_report.end, "idle_timeout")


if __name__ == "__main__":
    unittest.main()
