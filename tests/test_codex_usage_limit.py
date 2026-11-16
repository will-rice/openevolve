"""The codex_app_server backend on a turn that hits the account's usage limit.

The call starts a real process: `codex_command` runs tests/fake_codex_app_server.py,
which ends the turn with codex_usage_limit_turn.jsonl: the ErrorNotification and
failed turn/completed codex 0.161.0's app-server sends for the usage-limit error a
real turn ended on (2026-11-11), mapped as its event handler does
(codex-rs/app-server/src/bespoke_event_handling.rs:1030-1052, 1610-1628).
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from openevolve.config import LLMModelConfig
from openevolve.llm.codex_app_server import CodexAppServerLLM, CodexUsageLimitError

FAKE = Path(__file__).parent / "fake_codex_app_server.py"
USAGE_LIMIT_TURN = Path(__file__).parent / "codex_usage_limit_turn.jsonl"


class TestUsageLimit(unittest.TestCase):
    def test_a_usage_limit_raises_at_once_with_codexs_message(self):
        """Retries would only meet the same limit: one turn, then the error."""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "turns.jsonl"
            command = [sys.executable, str(FAKE), str(log), "0", str(USAGE_LIMIT_TURN)]
            llm = CodexAppServerLLM(
                LLMModelConfig(
                    name="gpt-test",
                    timeout=30,
                    retries=3,
                    retry_delay=0,
                    codex_command=command,
                )
            )

            with self.assertRaises(CodexUsageLimitError) as raised:
                asyncio.run(llm.generate("p"))

            self.assertEqual(len(log.read_text().splitlines()), 1)
        self.assertEqual(
            str(raised.exception),
            "You’ve hit your usage limit. Visit https://chatgpt.com/settings/usage to "
            "purchase more credits or try again at Nov 16th, 2026 1:40 PM.",
        )


if __name__ == "__main__":
    unittest.main()
