"""A fake `claude -p` process for tests: standard library only.

Started as `python fake_claude_code.py <log> <stream> <sleep> [<program>] -p ...` through
the provider's `claude_command`. It reads its prompt from stdin and appends one JSON line
to <log>: its arguments after <program> (or <sleep>), its cwd, the prompt, and the files
of its cwd (.py) with their text. With <program>, it then writes it to program.py, as a model
editing the program in place does. Then it prints <stream>'s lines, a recorded run's
stream-json events, <sleep> seconds before each.
"""

import json
import os
import sys
import time

log_path, stream, sleep = sys.argv[1], sys.argv[2], float(sys.argv[3])
rest = sys.argv[4:]
program = None if rest[0] == "-p" else rest.pop(0)
prompt = sys.stdin.read()
files = {}
for name in sorted(os.listdir()):
    # A workspace's program files; a run outside one starts in the caller's cwd.
    if name.endswith(".py") and os.path.isfile(name):
        with open(name, encoding="utf-8") as f:
            files[name] = f.read()
with open(log_path, "a", encoding="utf-8") as log:
    log.write(
        json.dumps({"args": rest, "cwd": os.getcwd(), "prompt": prompt, "files": files}) + "\n"
    )
if program is not None:
    with open("program.py", "w", encoding="utf-8") as f:
        f.write(program)
with open(stream, encoding="utf-8") as recorded:
    for line in recorded:
        time.sleep(sleep)
        print(line, end="", flush=True)
