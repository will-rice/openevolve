"""
Codex app-server interface for LLMs.

Uses the Codex CLI's app-server (`codex app-server`) as a non-interactive LLM
backend, enabling OpenEvolve to run with models available through a ChatGPT
(Codex) subscription without requiring direct API keys.

Authentication is delegated entirely to the app-server's managed ChatGPT login
(browser or device-code flow): the app-server persists the tokens itself and
refreshes them automatically. This adapter never reads the CLI's token files.
Before running a turn it asks the app-server for the current account state
(`account/read`) and requires a ChatGPT-plan login — if Codex is not
authenticated, it fails with actionable instructions instead of falling back
to an API key.

Usage in config.yaml:
    llm:
      provider: "codex_app_server"
      models:
        - name: "gpt-5.3-codex"
          weight: 1.0
          timeout: 300
          reasoning_effort: "medium"

Or inject programmatically:
    from openevolve.llm.codex_app_server import init_codex_app_server_client
    for model_cfg in config.llm.models:
        model_cfg.init_client = init_codex_app_server_client

Notes:
- `temperature`, `top_p` and `max_tokens` have no equivalent app-server option
  in this first pass and are therefore ignored.
- Each generation call spawns a fresh `codex app-server` process (from
  `codex_command`, ["codex"] by default), speaks newline-delimited JSON-RPC over
  its stdio (the protocol's default transport), and shuts it down again. Threads
  are non-interactive: tool approvals are set to "never". The sandbox is
  `sandbox_mode`, read-only by default, so the model can answer but not mutate
  the tree while generating a response.
- With `workspace_root`, each call runs Codex in a fresh directory under it
  holding the call's `workspace_files` (the evolution loop passes the parent
  program, `program<file_suffix>`, and each program the prompt may show,
  `<id><file_suffix>`), removed when the call ends. A sandbox that
  can write ("workspace-write", "danger-full-access") requires it, so calls that
  run at once never write one cwd.
- With `reply_file` (a writable sandbox only), the reply is that file of the
  call's workspace as the completed turn left it, in place of the agent's message,
  which is kept as `last_message` (as it is without reply_file):
  the model edits the program in place, in as many tool calls as it needs, and the
  file is the child, taken whole by OpenEvolve's full-rewrite path
  (utils/code_utils.py:parse_full_rewrite, whose reply without a code fence is the
  program verbatim). Only a turn that completed is read; a file that is gone, is not
  UTF-8 or holds a code fence (which the full-rewrite path would cut) raises
  CodexReplyFileError, never retried: another turn starts over from the parent.
- A turn that ends on the account's usage limit (the app-server's `error`
  notification with `codexErrorInfo` "usageLimitExceeded") raises
  CodexUsageLimitError at once, never retried: another call only meets the same
  limit. The caller decides when to try again.
- The stdio transport is unsupported on Windows; the backend expects a POSIX
  shell there.
"""

import asyncio
import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)

# Union of reasoning efforts documented for the GPT-5.x / Codex model families.
VALID_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")

# ThreadStartParams.sandbox: the app-server's SandboxMode string enum.
VALID_SANDBOX_MODES = ("read-only", "workspace-write", "danger-full-access")
# ThreadStartParams.config.web_search: the app-server's WebSearchMode string enum.
VALID_WEB_SEARCH_MODES = ("disabled", "cached", "indexed", "live")

_CLIENT_INFO = {"name": "openevolve", "title": "OpenEvolve", "version": "1.0.0"}

# TurnError.codexErrorInfo of a turn that hit the account's usage limit: codex maps
# its UsageLimitReached, QuotaExceeded and UsageNotIncluded errors to it
# (codex-rs/protocol/src/error.rs:470-472, rust-v0.161.0).
USAGE_LIMIT_EXCEEDED = "usageLimitExceeded"


class CodexAuthError(RuntimeError):
    """Raised when the Codex app-server reports no usable ChatGPT login.

    The Codex app-server owns the ChatGPT browser/device-code login and
    refreshes tokens itself; OpenEvolve never reads or writes the CLI's token
    files, so the only remedy is to log in via the Codex CLI first.
    """

    def __init__(self, message: str):
        super().__init__(
            f"{message} Run `codex login` (browser flow) or `codex login --device-code` "
            "to authenticate with your ChatGPT account first."
        )


class CodexReplyFileError(RuntimeError):
    """Raised when a completed turn's `reply_file` cannot be the reply.

    The file is gone, is not UTF-8, or holds a code fence. The turn is not retried:
    a retry starts over from the parent in a fresh workspace, at another turn's cost.
    """


class CodexUsageLimitError(RuntimeError):
    """Raised when a turn ends on the account's usage limit.

    Its message is codex's, which names when the limit resets when codex knows
    ("You've hit your usage limit. ... try again at Nov 16th, 2026 1:40 PM.").
    """


def _cfg_value(model_cfg: Any, attr: str, default: Any) -> Any:
    value = getattr(model_cfg, attr, None)
    return default if value is None else value


class _AppServerSession:
    """Client for one `codex app-server` process speaking JSONL JSON-RPC.

    Requests are written as one JSON object per line to the process's stdin;
    responses (which echo the request id) and notifications (which omit it) are
    read line by line from stdout. A background thread feeds a queue so reads
    can honor a wall-clock deadline instead of blocking forever on a hung
    app-server.
    """

    def __init__(self, proc: "subprocess.Popen", stderr_tail: int = 500):
        self.proc = proc
        self._next_id = 0
        self._messages: "queue.Queue[Any]" = queue.Queue()
        self._pending: "deque[Any]" = deque()
        self._stderr_tail = stderr_tail
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._messages.put(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON line from codex app-server: {line[:200]}")
        except (ValueError, OSError):
            pass
        finally:
            self._messages.put(None)  # EOF sentinel

    def stderr_summary(self) -> str:
        try:
            raw = (self.proc.stderr.read() or "") if self.proc.stderr else ""
        except (ValueError, OSError):
            raw = ""
        return raw.strip()[-self._stderr_tail :]

    def request(self, method: str, params: Optional[Dict[str, Any]], deadline: float) -> Any:
        """Send a request and block until its response arrives or time runs out."""
        self._next_id += 1
        request_id = self._next_id
        self._send({"id": request_id, "method": method, "params": params or {}})
        # Non-matching messages must not be parked in self._pending here:
        # next_message() drains _pending first, so a parked message would
        # be returned again on the next iteration and the loop would spin
        # forever without ever reading the actual response from the queue
        # (a real codex app-server emits notifications before responses).
        # Stash non-matching messages locally during the request and
        # restore them to the front of _pending, in order, when the
        # response arrives.
        stash: List[Dict[str, Any]] = []
        try:
            while True:
                message = self.next_message(deadline)
                if message.get("id") != request_id:
                    # A notification (e.g. an early item/completed) or a late
                    # response: stash it instead of dropping it.
                    stash.append(message)
                    continue
                if "error" in message:
                    error = message["error"] or {}
                    raise RuntimeError(
                        f"codex app-server rejected {method}: "
                        f"{error.get('code')}: {error.get('message')}"
                    )
                return message.get("result")
        finally:
            for stashed in reversed(stash):
                self._pending.appendleft(stashed)

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._send({"method": method, "params": params or {}})

    def next_message(self, deadline: float) -> Dict[str, Any]:
        """Return the next raw JSON-RPC message, honoring the wall-clock deadline."""
        if self._pending:
            return self._pending.popleft()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError("codex app-server did not respond in time")
        try:
            message = self._messages.get(timeout=remaining)
        except queue.Empty:
            raise asyncio.TimeoutError("codex app-server did not respond in time")
        if message is None:  # EOF: the process exited
            raise RuntimeError(
                f"codex app-server exited unexpectedly. stderr: {self.stderr_summary()}"
            )
        return message

    def _send(self, payload: Dict[str, Any]) -> None:
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except (ValueError, OSError, BrokenPipeError) as e:
            raise RuntimeError(f"Failed to write to codex app-server: {e}. {self.stderr_summary()}")

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        # Closing stdout while the reader is blocked in it can wait forever for
        # the stream's lock. Stop and reap the child before closing its read pipes.
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
        for stream in (self.proc.stdout, self.proc.stderr):
            try:
                if stream:
                    stream.close()
            except (OSError, ValueError):
                pass


class CodexAppServerLLM(LLMInterface):
    """LLM interface that generates through the Codex app-server.

    Requires the `codex` CLI to be installed and logged in with a ChatGPT
    account (the app-server manages that login and token refresh itself).
    """

    def __init__(self, model_cfg=None):
        self.model = _cfg_value(model_cfg, "name", "gpt-5.3-codex")
        self.system_message = getattr(model_cfg, "system_message", None)
        self.timeout = _cfg_value(model_cfg, "timeout", 300)
        self.weight = _cfg_value(model_cfg, "weight", 1.0)
        self.retries = _cfg_value(model_cfg, "retries", 3)
        self.retry_delay = _cfg_value(model_cfg, "retry_delay", 5)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.codex_command = list(_cfg_value(model_cfg, "codex_command", ["codex"]))
        self.sandbox_mode = _cfg_value(model_cfg, "sandbox_mode", "read-only")
        self.workspace_root = getattr(model_cfg, "workspace_root", None)
        self.reply_file = getattr(model_cfg, "reply_file", None)
        self.web_search = getattr(model_cfg, "web_search", None)
        # The final agent message of the latest turn that completed, the reply or,
        # with reply_file, beside it; the evolution loop reads it from the ensemble
        # (LLMEnsemble.last_message) for the evaluator (ChildContext.llm_message).
        self.last_message: Optional[str] = None
        if self.web_search is not None and self.web_search not in VALID_WEB_SEARCH_MODES:
            raise ValueError(
                f"Invalid web_search: {self.web_search}. "
                f"Expected one of {', '.join(VALID_WEB_SEARCH_MODES)}"
            )
        if self.sandbox_mode not in VALID_SANDBOX_MODES:
            raise ValueError(
                f"Invalid sandbox_mode: {self.sandbox_mode}. "
                f"Expected one of {', '.join(VALID_SANDBOX_MODES)}"
            )
        if self.sandbox_mode != "read-only" and self.workspace_root is None:
            raise ValueError(
                f"sandbox_mode {self.sandbox_mode} needs a workspace_root, so that "
                "concurrent calls never write one working directory"
            )
        if self.reply_file is not None and self.sandbox_mode == "read-only":
            raise ValueError(
                f"reply_file {self.reply_file} needs a writable sandbox_mode, in which "
                "the model can edit it"
            )
        if (
            self.reasoning_effort is not None
            and self.reasoning_effort not in VALID_REASONING_EFFORTS
        ):
            raise ValueError(
                f"Invalid reasoning_effort: {self.reasoning_effort}. "
                f"Expected one of {', '.join(VALID_REASONING_EFFORTS)}"
            )
        logger.info(f"Initialized CodexAppServerLLM with model: {self.model}")

    async def generate(self, prompt: str, **kwargs) -> str:
        sys_msg = kwargs.pop("system_message", self.system_message) or ""
        return await self.generate_with_context(
            system_message=sys_msg,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        user_content = "\n\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        prompt = f"{system_message}\n\n{user_content}" if system_message else user_content

        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        files = kwargs.get("workspace_files", {})

        loop = asyncio.get_event_loop()
        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: self._run_app_server(prompt, timeout, files)
                    ),
                    timeout=timeout + 30,
                )
            except FileNotFoundError:
                logger.error(
                    "The `codex` CLI was not found. Install it and run `codex login` first."
                )
                raise
            except CodexAuthError as e:
                # Retrying cannot help until the user logs in; fail immediately.
                logger.error(f"Codex app-server authentication error (not retried): {e}")
                raise
            except CodexUsageLimitError:
                # Retrying at once meets the same limit; the caller waits it out.
                raise
            except CodexReplyFileError as e:
                logger.error(f"Codex app-server reply file unusable (not retried): {e}")
                raise
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        f"Codex app-server timeout on attempt {attempt + 1}/{retries + 1}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Codex app-server error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {e}")
                    raise

    def _build_command(self) -> List[str]:
        return [*self.codex_command, "app-server"]

    def _run_app_server(self, prompt: str, timeout: int, files: Dict[str, str]) -> str:
        """Run one turn, in a fresh workspace holding `files` when one is configured."""
        self.last_message = None
        if self.workspace_root is None:
            self.last_message = self._run_turn(prompt, timeout, None)
            return self.last_message
        with tempfile.TemporaryDirectory(dir=self.workspace_root) as workspace:
            for name, text in files.items():
                # newline="": the program's bytes as written, CRLF included.
                with open(os.path.join(workspace, name), "w", encoding="utf-8", newline="") as f:
                    f.write(text)
            # Returns only once the turn completed; a timeout or a failed turn raises
            # first, so a half-edited reply_file is never read.
            message = self._run_turn(prompt, timeout, workspace)
            self.last_message = message
            if self.reply_file is None:
                return message
            return self._read_reply_file(os.path.join(workspace, self.reply_file))

    def _read_reply_file(self, path: str) -> str:
        """The completed turn's `reply_file`, byte for byte, as the reply."""
        try:
            with open(path, encoding="utf-8", newline="") as f:
                text = f.read()
        except FileNotFoundError:
            raise CodexReplyFileError(f"the turn left no {self.reply_file}") from None
        except UnicodeDecodeError as e:
            raise CodexReplyFileError(f"{self.reply_file} is not UTF-8: {e}") from None
        if "```" in text:
            raise CodexReplyFileError(
                f"{self.reply_file} holds a code fence (```), which the full-rewrite "
                "parser would take as the end of the program"
            )
        return text

    def _run_turn(self, prompt: str, timeout: int, cwd: Optional[str]) -> str:
        deadline = time.monotonic() + timeout
        session = self._start_session(cwd)
        try:
            # JSON-RPC handshake (any request before this is rejected by the server).
            session.request("initialize", {"clientInfo": _CLIENT_INFO}, deadline)
            session.notify("initialized")

            # Auth boundary: the app-server owns the ChatGPT login and token
            # refresh; verify it has a usable subscription account up front.
            self._ensure_authenticated(session, deadline)

            thread_params: Dict[str, Any] = {
                "model": self.model,
                "approvalPolicy": "never",
                # ThreadStartParams.sandbox is a SandboxMode string enum
                # ("read-only" | "workspace-write" | "danger-full-access"),
                # not a {"mode": ..., "networkAccess": ...} map — current
                # app-server versions reject the map form with -32600.
                "sandbox": self.sandbox_mode,
            }
            if cwd is not None:
                thread_params["cwd"] = cwd
            if self.web_search is not None:
                thread_params["config"] = {"web_search": self.web_search}
            thread_result = session.request("thread/start", thread_params, deadline)
            thread_id = (thread_result or {}).get("thread", {}).get("id")
            if not thread_id:
                raise RuntimeError(f"codex app-server returned no thread id: {thread_result}")

            turn_params: Dict[str, Any] = {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}],
            }
            if self.reasoning_effort is not None:
                turn_params["effort"] = self.reasoning_effort
            session.request("turn/start", turn_params, deadline)

            return self._collect_response(session, deadline)
        finally:
            session.close()

    def _ensure_authenticated(self, session: _AppServerSession, deadline: float) -> None:
        result = session.request("account/read", {}, deadline) or {}
        # Current app-server versions return requiresOpenaiAuth=true
        # together with a valid ChatGPT account — the flag describes the
        # auth mode, not a missing login. The account presence/type check
        # below is the real auth boundary.
        if result.get("account") is None:
            raise CodexAuthError("Codex app-server reports the account is not authenticated.")
        account = result.get("account") or {}
        if account.get("type") != "chatgpt":
            raise CodexAuthError(
                "Codex app-server is not signed in with a ChatGPT account "
                f"(found auth type '{account.get('type')}'). The codex_app_server "
                "provider bills a ChatGPT subscription; to use an API key instead, "
                "keep the default OpenAI provider."
            )

    def _collect_response(self, session: _AppServerSession, deadline: float) -> str:
        """Stream notifications until the turn finishes; return the agent message.

        The agent text arrives as `item/completed` notifications
        (`item.type == "agentMessage"`); `turn/completed` closes the turn.
        """
        text: Optional[str] = None
        # The queries of the turn's web searches (WebSearchThreadItem.query), logged
        # when it completes so a caller can see whether and how the model searched.
        searches: List[str] = []
        while True:
            message = session.next_message(deadline)
            if message.get("id") is not None:
                continue  # a response, not a notification
            method = message.get("method")
            params = message.get("params") or {}
            if method == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "agentMessage":
                    text = item.get("text") or ""
                elif item.get("type") == "webSearch":
                    searches.append(item["query"])
            elif method == "turn/completed":
                status = (params.get("turn") or {}).get("status")
                if status and status != "completed":
                    raise RuntimeError(f"codex app-server turn did not complete: {status}")
                logger.info(f"Codex turn completed with {len(searches)} web searches: {searches}")
                break
            elif method == "error":
                # ErrorNotification: a TurnError and whether codex retries the turn
                # itself (codex-rs/app-server-protocol/src/protocol/v2/notification.rs:62-69).
                error = params["error"]
                if error.get("codexErrorInfo") == USAGE_LIMIT_EXCEEDED:
                    raise CodexUsageLimitError(error["message"])
                raise RuntimeError(f"codex app-server reported an error: {params}")
        if self.reply_file is None and (not text or not text.strip()):
            raise RuntimeError(
                f"Empty response from codex app-server. stderr: {session.stderr_summary()}"
            )
        # With reply_file the message is not the reply, and may be absent.
        return text or ""

    def _start_session(self, cwd: Optional[str]) -> _AppServerSession:
        proc = subprocess.Popen(
            self._build_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=cwd,
        )
        return _AppServerSession(proc)


def init_codex_app_server_client(model_cfg):
    """Factory function compatible with OpenEvolve's init_client config hook."""
    return CodexAppServerLLM(model_cfg)
