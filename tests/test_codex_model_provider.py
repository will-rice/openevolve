"""The codex_app_server backend on a model provider other than OpenAI's.

The thread-parameter and auth tests drive the in-process fake app-server of
test_codex_app_server_llm. The retry tests start a real process: `codex_command`
runs tests/fake_codex_app_server.py, which answers the turn with recorded
notifications. codex_unreachable_provider_turn.jsonl is what codex 0.161.0's
app-server sent (2026-11-19) for a turn on a custom provider whose base_url refused
connections: an ErrorNotification with willRetry true, after which codex goes on
reconnecting and sends nothing more. codex_retried_turn.jsonl is that notification
followed by the end of a real completed turn (codex_web_search_turn.jsonl's).
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from openevolve.config import LLMModelConfig
from openevolve.llm.codex_app_server import (
    CodexAppServerLLM,
    CodexAuthError,
)
from test_codex_app_server_llm import _make_cfg, run_generate

FAKE = Path(__file__).parent / "fake_codex_app_server.py"
UNREACHABLE_TURN = Path(__file__).parent / "codex_unreachable_provider_turn.jsonl"
RETRIED_TURN = Path(__file__).parent / "codex_retried_turn.jsonl"

LOCAL = {"base_url": "http://127.0.0.1:8000/v1", "wire_api": "responses", "name": "local"}


def local_overrides(**changes):
    return {"model_provider": {"local": LOCAL}, "web_search": "disabled", **changes}


class TestThreadParams(unittest.TestCase):
    def test_a_custom_provider_is_the_threads_provider_and_config(self):
        """modelProvider is the id; its definition is the model_providers.<id> override."""
        outcome, fake, _ = run_generate(cfg_overrides=local_overrides(context_window=131072))
        self.assertIsNone(outcome["error"])
        params = next(p for m, p in fake.requests if m == "thread/start")
        self.assertEqual(params["modelProvider"], "local")
        self.assertEqual(
            params["config"],
            {
                "web_search": "disabled",
                "model_context_window": 131072,
                "model_providers.local": LOCAL,
            },
        )

    def test_openais_provider_sends_no_provider(self):
        _, fake, _ = run_generate(cfg_overrides={"web_search": "live"})
        params = next(p for m, p in fake.requests if m == "thread/start")
        self.assertNotIn("modelProvider", params)
        self.assertEqual(params["config"], {"web_search": "live"})


class TestAuthRule(unittest.TestCase):
    def test_a_custom_provider_needs_no_chatgpt_login(self):
        """No account/read: a real app-server answers it with no account
        ({"account": null, "requiresOpenaiAuth": true}) on a login-free home."""
        outcome, fake, _ = run_generate(scenario="unauthenticated", cfg_overrides=local_overrides())
        self.assertIsNone(outcome["error"])
        self.assertEqual(outcome["result"], "hello world")
        self.assertNotIn("account/read", [m for m, _ in fake.requests])

    def test_openais_provider_still_requires_the_login(self):
        outcome, fake, _ = run_generate(scenario="unauthenticated")
        self.assertIsInstance(outcome["error"], CodexAuthError)
        self.assertNotIn("thread/start", [m for m, _ in fake.requests])


class TestRefusedProviders(unittest.TestCase):
    def assert_refused(self, pattern, **overrides):
        with self.assertRaisesRegex(ValueError, pattern):
            CodexAppServerLLM(_make_cfg(**overrides))

    def test_a_built_in_id_is_refused(self):
        self.assert_refused("codex's own", **local_overrides(model_provider={"openai": LOCAL}))

    def test_an_id_that_is_not_one_config_segment_is_refused(self):
        self.assert_refused("must match", **local_overrides(model_provider={"a.b": LOCAL}))

    def test_two_providers_are_refused(self):
        self.assert_refused(
            "one provider", **local_overrides(model_provider={"a": LOCAL, "b": LOCAL})
        )

    def test_a_provider_without_base_url_is_refused(self):
        self.assert_refused("base_url", **local_overrides(model_provider={"local": {}}))

    def test_the_removed_chat_wire_api_is_refused(self):
        chat = {**LOCAL, "wire_api": "chat"}
        self.assert_refused("wire_api", **local_overrides(model_provider={"local": chat}))

    def test_a_custom_provider_without_web_search_is_refused(self):
        self.assert_refused("web_search", **local_overrides(web_search=None))

    def test_a_non_positive_context_window_is_refused(self):
        self.assert_refused("context_window", context_window=0)


def local_llm(log, turn, timeout):
    """A custom-provider model whose app-server answers each turn with `turn`."""
    command = [sys.executable, str(FAKE), str(log), "0", str(turn)]
    return CodexAppServerLLM(
        LLMModelConfig(
            name="local-model",
            timeout=timeout,
            retries=0,
            retry_delay=0,
            codex_command=command,
            model_provider={"local": LOCAL},
            web_search="disabled",
        )
    )


class TestRetriedErrors(unittest.TestCase):
    def test_a_retried_error_does_not_end_the_turn(self):
        """willRetry: codex retries itself; the turn's reply is its final message."""
        with tempfile.TemporaryDirectory() as directory:
            llm = local_llm(Path(directory) / "turns.jsonl", RETRIED_TURN, 30)

            with self.assertLogs("openevolve.llm.codex_app_server", "WARNING") as logs:
                reply = asyncio.run(llm.generate("p"))

        self.assertEqual(reply, "[GPT\u20116.2 Astra and Sol](https://news.ycombinator.com/)")
        self.assertEqual(llm.last_backend, "local/local-model")
        self.assertIn("responseStreamDisconnected", logs.output[0])

    def test_an_unreachable_provider_runs_to_the_calls_timeout(self):
        """codex reconnects without end; the call ends at its timeout, not a usage limit."""
        with tempfile.TemporaryDirectory() as directory:
            llm = local_llm(Path(directory) / "turns.jsonl", UNREACHABLE_TURN, 2)

            with self.assertRaises(asyncio.TimeoutError):
                asyncio.run(llm.generate("p"))
        self.assertIsNone(llm.last_backend)


if __name__ == "__main__":
    unittest.main()
