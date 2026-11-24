"""
What one agent turn did: the report the agent providers (claude_code, codex_app_server)
build from the events they already parse, log at the turn's end and keep for the caller.

Each provider fills a TurnReport while its turn runs, then logs it as one INFO line,
`turn report: <compact JSON>`, sets it as `last_turn_report` and, with the model
config's `turn_report_file`, appends it to that file as one JSON line. It is set and
written on every path, a turn that raised included, so a caller that catches the
error reads the report of the turn that raised.

A failed tool call is put under one reason by its error text alone, the
texts Claude Code and codex 0.161.0 sent in recorded turns:
- "permission": Claude Code's permission layer refused the call ("Permission to use
  Bash has been denied because Claude Code is running in don't ask mode. ..."), or
  codex declined it (status "declined") or refused an MCP call it may not ask
  approval for ("MCP tool call requires approval, but approval policy is never").
- "outside_workdirs": Claude Code's file tools refused a path ("/etc/hostname is
  outside /workspace; the permissions.blockReadsOutsideWorkingDirectories setting
  blocks reads outside the working directories. ...").
- "sandbox": a shell call whose error output names a refused file system
  operation ("Read-only file system", "Permission denied").
- "missing_file": a shell call whose error output names a path that does not exist
  ("No such file or directory"): mostly a command that looked for a file no one
  wrote, which no sandbox refused.
- "other": any other failed call (a command that exited non-zero, an MCP tool that
  returned an error, a file tool that found no file).
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# The error texts of a refused permission: Claude Code's ("Permission to use <tool>
# has been denied ..."; Claude Code 2.1.321) and codex's for an MCP call under
# approvalPolicy "never" (codex 0.161.0, both recorded 2026-11-22).
PERMISSION_PATTERN = re.compile(
    r"Permission to use \S+ has been denied|requires approval, but approval policy is never"
)
# Claude Code's refusal of a file tool's path (permissions.blockReadsOutsideWorkingDirectories).
OUTSIDE_WORKDIRS_TEXT = "outside the working directories"
# A shell's errors for a file system operation the sandbox or the OS refused.
SANDBOX_TEXTS = ("Read-only file system", "Permission denied")
# A shell's error for a path that does not exist.
MISSING_TEXT = "No such file or directory"
# Examples kept per reason, and how much of a call's input and error each keeps.
MAX_EXAMPLES = 3
INPUT_CHARS = 120
ERROR_CHARS = 160


@dataclass
class ToolFailureExample:
    """One failed tool call: its tool, and the start of its input and its error."""

    tool: str
    input: str
    error: str


@dataclass
class ToolFailures:
    """The failed tool calls of one reason: how many, and the first MAX_EXAMPLES."""

    count: int = 0
    examples: List[ToolFailureExample] = field(default_factory=list)


@dataclass
class TurnReport:
    """What one agent turn did.

    Attributes:
        agent: The agent that ran the turn, "claude-code" or "codex".
        backend: The provider's `backend`, "<provider id>/<model>".
        wall_seconds: The turn's wall time, from its process's start to its end.
        requests: Model replies the turn received (see each provider's notes for
            what it counts).
        compactions: Context compactions during the turn.
        tool_calls: Tool calls by tool name (MCP tools as mcp__<server>__<tool>).
        tool_failures: Failed tool calls by reason: "permission", "outside_workdirs",
            "sandbox" or "other" (the module's notes say which is which).
        api_retries: Requests the agent retried itself, by error (an HTTP status for
            Claude Code, codex's error kind for codex).
        reply_changed: Whether the turn changed `reply_file` from what the call wrote
            to its workspace (a file the turn deleted or created counts); None without
            a reply_file.
        end: How the turn ended: "completed" (the agent finished), "max_turns"
            (Claude Code stopped at its --max-turns), "failed" (the run ended on an
            error), "interrupted" (codex interrupted the turn), "timeout" or
            "idle_timeout" (the provider's limits ended it), or "aborted" (the turn
            raised before it reached an end: the process exited, a request failed).
        error: The start of the error the turn ended on, when it failed.
    """

    agent: str
    backend: str
    wall_seconds: float = 0.0
    requests: int = 0
    compactions: int = 0
    tool_calls: Dict[str, int] = field(default_factory=dict)
    tool_failures: Dict[str, ToolFailures] = field(default_factory=dict)
    api_retries: Dict[str, int] = field(default_factory=dict)
    reply_changed: Optional[bool] = None
    end: Optional[str] = None
    error: Optional[str] = None

    def add_tool_call(self, tool: str) -> None:
        """Count one call of `tool`."""
        self.tool_calls[tool] = self.tool_calls.get(tool, 0) + 1

    def add_tool_failure(self, reason: str, tool: str, call_input: str, error: str) -> None:
        """Count one failed call under `reason`, keeping it as an example while room is left."""
        failures = self.tool_failures.setdefault(reason, ToolFailures())
        failures.count += 1
        if len(failures.examples) < MAX_EXAMPLES:
            failures.examples.append(
                ToolFailureExample(tool, call_input[:INPUT_CHARS], error_excerpt(error))
            )

    def add_api_retry(self, error: str) -> None:
        """Count one request the agent retried after `error`."""
        self.api_retries[error] = self.api_retries.get(error, 0) + 1

    def to_json(self) -> str:
        """The report as one line of compact JSON."""
        return json.dumps(asdict(self), separators=(",", ":"), ensure_ascii=False)

    def emit(self, logger: logging.Logger, path: Optional[str]) -> None:
        """Log the report as one INFO line and, with `path`, append it there as a JSON line."""
        line = self.to_json()
        logger.info(f"turn report: {line}")
        if path is not None:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def failure_reason(shell: bool, error: str) -> str:
    """The reason a tool call failed, from its error text (the module's notes).

    Args:
        shell: Whether the call ran a shell command (Claude Code's Bash, codex's
            commandExecution), whose file system errors are "sandbox" or
            "missing_file".
        error: The call's error text.
    """
    if PERMISSION_PATTERN.search(error):
        return "permission"
    if OUTSIDE_WORKDIRS_TEXT in error:
        return "outside_workdirs"
    if shell and any(text in error for text in SANDBOX_TEXTS):
        return "sandbox"
    if shell and MISSING_TEXT in error:
        return "missing_file"
    return "other"


def error_excerpt(error: str) -> str:
    """The ERROR_CHARS of `error` an example keeps: the line that gave its reason.

    A shell's output can run long before the line `failure_reason` read (a command
    that printed a file, then listed a missing one), so the example keeps the first
    line naming a text of the module's notes, or else the error's start.
    """
    texts = (OUTSIDE_WORKDIRS_TEXT, *SANDBOX_TEXTS, MISSING_TEXT)
    for line in error.splitlines():
        if PERMISSION_PATTERN.search(line) or any(text in line for text in texts):
            return line[:ERROR_CHARS]
    return error[:ERROR_CHARS]
