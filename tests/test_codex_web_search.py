"""The codex_app_server backend on a turn that searched the web.

The call starts a real process: `codex_command` runs tests/fake_codex_app_server.py,
which ends the turn with codex_web_search_turn.jsonl: the item/completed and
turn/completed notifications codex 0.161.0's app-server sent for a real turn on
gpt-6.1-sol with the thread's web_search "live" (2026-11-17), one of them a
WebSearchThreadItem.
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from openevolve.config import LLMModelConfig
from openevolve.llm.codex_app_server import CodexAppServerLLM

FAKE = Path(__file__).parent / "fake_codex_app_server.py"
WEB_SEARCH_TURN = Path(__file__).parent / "codex_web_search_turn.jsonl"


class TestWebSearch(unittest.TestCase):
    def test_a_turns_web_searches_are_logged_with_its_reply(self):
        """The reply is the final agent message; the search's query is logged."""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "turns.jsonl"
            command = [sys.executable, str(FAKE), str(log), "0", str(WEB_SEARCH_TURN)]
            llm = CodexAppServerLLM(
                LLMModelConfig(
                    name="gpt-test",
                    timeout=30,
                    retries=0,
                    retry_delay=0,
                    codex_command=command,
                    web_search="live",
                )
            )

            with self.assertLogs("openevolve.llm.codex_app_server", "INFO") as logs:
                reply = asyncio.run(llm.generate("p"))

        self.assertEqual(reply, "[GPT‑6.2 Astra and Sol](https://news.ycombinator.com/)")
        self.assertIn(
            "Codex turn on openai/gpt-test completed with 1 web searches: "
            "['https://news.ycombinator.com/']",
            logs.output[-1],
        )


if __name__ == "__main__":
    unittest.main()
