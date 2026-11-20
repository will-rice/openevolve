"""
Claude Code CLI interface for LLMs.

Runs the Claude Code CLI headless (`claude -p`,
https://code.claude.com/docs/en/headless) as an agent backend, one process per call,
on Anthropic's API through the CLI's own login, or on any Anthropic-compatible
endpoint its environment or `claude_args` name (e.g. `--settings` holding an "env"
table with ANTHROPIC_BASE_URL). It shares CodexAppServerLLM's call handling: the
per-call workspace (`workspace_root`, `workspace_files`), `reply_file`, the retries,
`timeout`, `idle_timeout` and `turn_timeout`, `last_message` and `last_backend`; only
the turn differs.

Usage in config.yaml:
    llm:
      provider: "claude_code"
      models:
        - name: "sonnet"
          weight: 1.0
          timeout: 300

Or inject programmatically:
    from openevolve.llm.claude_code import init_claude_code_client
    for model_cfg in config.llm.models:
        model_cfg.init_client = init_claude_code_client

Notes:
- The command is `claude_command` (["claude"] when unset; a launcher may go first),
  then `-p --output-format stream-json --verbose --model <name>`, then
  `--max-budget-usd` when `max_budget_usd` is set, then `claude_args` (tools,
  permission mode, settings). The system message and the prompt go on stdin as one
  text, as the Codex provider sends them, which `-p` reads when no prompt argument
  is given, so their size is not bounded by Linux's 128 KiB argument limit; Claude
  Code's own system prompt, which tells the model its tools, stays.
- Each stdout line is one stream-json event. The turn ends at the `result` event,
  whose `result` is the agent's final message; `is_error`, or a subtype other than
  "success" (error_max_turns, error_during_execution, error_max_budget_usd, ...:
  SDKResultMessage, https://code.claude.com/docs/en/agent-sdk/typescript), fails
  the call.
- With `idle_timeout`, every event but `system/api_retry`, which Claude Code emits
  before it retries a failed request, restarts it: a server that keeps refusing is
  not progress.
- `last_backend` is "claude-code/<model>".
"""

import asyncio
import logging
import math
import subprocess
import time
from typing import List, Optional

from openevolve.llm.codex_app_server import CodexAppServerLLM, _AppServerSession, _cfg_value

logger = logging.getLogger(__name__)

# The provider id of `last_backend`.
CLAUDE_CODE_PROVIDER_ID = "claude-code"
# The stream-json event Claude Code emits before it retries a failed API request.
API_RETRY = ("system", "api_retry")


class ClaudeCodeLLM(CodexAppServerLLM):
    """LLM interface that generates through a headless Claude Code run.

    Requires the `claude` CLI, logged in or pointed at an endpoint with a key.
    """

    def __init__(self, model_cfg=None):
        super().__init__(model_cfg)
        self.model = _cfg_value(model_cfg, "name", "sonnet")
        self.claude_command = list(_cfg_value(model_cfg, "claude_command", ["claude"]))
        self.claude_args = list(_cfg_value(model_cfg, "claude_args", []))
        self.max_budget_usd = getattr(model_cfg, "max_budget_usd", None)

    @property
    def backend(self) -> str:
        """The backend this provider's turns run on: "claude-code/<model>"."""
        return f"{CLAUDE_CODE_PROVIDER_ID}/{self.model}"

    def _build_command(self) -> List[str]:
        budget = (
            [] if self.max_budget_usd is None else ["--max-budget-usd", str(self.max_budget_usd)]
        )
        return [
            *self.claude_command,
            *("-p", "--output-format", "stream-json", "--verbose"),
            *("--model", self.model),
            *budget,
            *self.claude_args,
        ]

    def _run_turn(self, prompt: str, timeout: int, cwd: Optional[str]) -> str:
        proc = subprocess.Popen(
            self._build_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=cwd,
        )
        session = _AppServerSession(proc)
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
            start = time.monotonic()
            if self.idle_timeout is None:
                deadline, idle_until = start + timeout, math.inf
            else:
                limit = math.inf if self.turn_timeout is None else self.turn_timeout
                deadline, idle_until = start + limit, start + self.idle_timeout
            while True:
                try:
                    event = session.next_message(min(deadline, idle_until))
                except asyncio.TimeoutError:
                    if idle_until < deadline:
                        raise asyncio.TimeoutError(
                            f"Claude Code turn on {self.backend} made no progress for "
                            f"{self.idle_timeout}s"
                        ) from None
                    raise
                if (event.get("type"), event.get("subtype")) == API_RETRY:
                    logger.warning(f"Claude Code on {self.backend} retries a request: {event}")
                elif self.idle_timeout is not None:
                    idle_until = time.monotonic() + self.idle_timeout
                if event.get("type") != "result":
                    continue
                if event.get("is_error") or event.get("subtype") != "success":
                    raise RuntimeError(
                        f"Claude Code turn on {self.backend} failed: {event.get('subtype')}: "
                        f"{event.get('result') or event.get('errors')}"
                    )
                text = event.get("result") or ""
                logger.info(
                    f"Claude Code turn on {self.backend} completed in "
                    f"{event.get('num_turns')} model turns"
                )
                if self.reply_file is None and not text.strip():
                    raise RuntimeError(
                        f"Empty response from Claude Code. stderr: {session.stderr_summary()}"
                    )
                return text
        finally:
            session.close()


def init_claude_code_client(model_cfg):
    """Factory function compatible with OpenEvolve's init_client config hook."""
    return ClaudeCodeLLM(model_cfg)
