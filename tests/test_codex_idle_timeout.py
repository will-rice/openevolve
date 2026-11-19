"""The codex_app_server backend's idle_timeout: a turn ends on no progress, not on time.

Each call starts a real process: `codex_command` runs tests/fake_codex_app_server.py,
which answers turn/start at once and then replays a recorded turn's notifications,
<sleep> seconds before each. codex_web_search_turn.jsonl is a real completed turn
(codex 0.161.0, 2026-11-17): four item/completed notifications, then turn/completed.
codex_unreachable_provider_turn.jsonl is a real turn on a server that refused
connections (2026-11-19): its user message's item/completed, then an
ErrorNotification codex retries itself, then nothing.
"""

import asyncio
import sys
import tempfile
import time
import unittest
from pathlib import Path

from openevolve.config import LLMModelConfig
from openevolve.llm.codex_app_server import CodexAppServerLLM

FAKE = Path(__file__).parent / "fake_codex_app_server.py"
SLOW_TURN = Path(__file__).parent / "codex_web_search_turn.jsonl"
STALLED_TURN = Path(__file__).parent / "codex_unreachable_provider_turn.jsonl"
# Seconds before each recorded notification of the slow turn: under IDLE, so the turn
# progresses, while its five notifications take three times IDLE in all.
PACE = 0.6
IDLE = 1


def run_turn(turn: Path, pace: float, **changes) -> tuple[str, float]:
    """One call on the fake replaying `turn`; its reply and how long it took."""
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "turns.jsonl"
        values = {
            "name": "gpt-test",
            "timeout": 1,
            "retries": 0,
            "retry_delay": 0,
            "codex_command": [sys.executable, str(FAKE), str(log), str(pace), str(turn)],
            **changes,
        }
        llm = CodexAppServerLLM(LLMModelConfig(**values))
        started = time.monotonic()
        reply = asyncio.run(llm.generate("p"))
        return reply, time.monotonic() - started


class TestIdleTimeout(unittest.TestCase):
    def test_a_turn_that_keeps_progressing_outlasts_both_limits(self):
        """Each gap is under the idle limit; the turn outruns it and `timeout` too."""
        reply, took = run_turn(SLOW_TURN, PACE, idle_timeout=IDLE)

        self.assertEqual(reply, "[GPT‑6.2 Astra and Sol](https://news.ycombinator.com/)")
        self.assertGreater(took, 2 * IDLE)

    def test_without_idle_timeout_the_same_turn_times_out(self):
        """The old whole-call deadline: `timeout` ends the slow turn."""
        with self.assertRaises(asyncio.TimeoutError):
            run_turn(SLOW_TURN, PACE)

    def test_a_stalled_turn_fails_after_the_idle_limit(self):
        """A retried error is no progress: the turn ends one idle limit after it."""
        with self.assertRaisesRegex(asyncio.TimeoutError, "made no progress for 1s"):
            run_turn(STALLED_TURN, 0, idle_timeout=IDLE, timeout=30)

    def test_turn_timeout_bounds_a_progressing_turn(self):
        with self.assertRaises(asyncio.TimeoutError):
            run_turn(SLOW_TURN, PACE, idle_timeout=IDLE, turn_timeout=1)

    def test_turn_timeout_needs_an_idle_timeout(self):
        with self.assertRaisesRegex(ValueError, "needs an idle_timeout"):
            CodexAppServerLLM(LLMModelConfig(name="gpt-test", turn_timeout=10))


if __name__ == "__main__":
    unittest.main()
