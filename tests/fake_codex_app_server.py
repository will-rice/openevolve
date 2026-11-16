"""A fake `codex app-server` process for tests: standard library only.

Started as `python fake_codex_app_server.py <log> <sleep> <reply> app-server` through the
provider's `codex_command`. It answers the requests the provider sends with the
shapes a real app-server returns (initialize, account/read with a ChatGPT login,
thread/start, turn/start, then item/completed and turn/completed notifications)
and appends one JSON line per turn to <log>: its process cwd, the thread's cwd and
sandbox, the turn's prompt, and the files of the thread's cwd, if it has one, with
their text. It sleeps <sleep> seconds
before answering a turn, and answers it with <reply>.
"""

import json
import os
import sys
import time

log_path = sys.argv[1]
sleep = float(sys.argv[2])
reply = sys.argv[3]
thread = {}


def send(message):
    print(json.dumps(message), flush=True)


for line in sys.stdin:
    request = json.loads(line)
    method, params = request["method"], request.get("params") or {}
    if method == "initialize":
        send({"id": request["id"], "result": {"userAgent": "fake"}})
    elif method == "account/read":
        send({"id": request["id"], "result": {"account": {"type": "chatgpt"}}})
    elif method == "thread/start":
        thread = params
        send({"id": request["id"], "result": {"thread": {"id": "thread-1"}}})
    elif method == "turn/start":
        time.sleep(sleep)
        files = {}
        for name in sorted(os.listdir(thread["cwd"]) if thread.get("cwd") else []):
            with open(os.path.join(thread["cwd"], name), encoding="utf-8", newline="") as f:
                files[name] = f.read()
        record = {
            "process_cwd": os.getcwd(),
            "thread_cwd": thread.get("cwd"),
            "sandbox": thread.get("sandbox"),
            "prompt": params["input"][0]["text"],
            "files": files,
        }
        with open(log_path, "a") as log:
            log.write(json.dumps(record) + "\n")
        send({"id": request["id"], "result": {}})
        item = {"id": "item-1", "type": "agentMessage", "text": reply}
        send({"method": "item/completed", "params": {"item": item}})
        turn = {"id": "turn-1", "status": "completed"}
        send({"method": "turn/completed", "params": {"turn": turn}})
