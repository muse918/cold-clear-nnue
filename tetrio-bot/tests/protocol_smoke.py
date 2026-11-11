#!/usr/bin/env python3
"""Exercise the real JSONL process, including observed-only queue replenishment."""
import json
import pathlib
import random
import subprocess
import sys

binary = str(pathlib.Path(sys.argv[1]).resolve())
process = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, bufsize=1)
requests = 0


def send(request, ok=True):
    global requests
    requests += 1
    process.stdin.write((request if isinstance(request, str) else json.dumps(request)) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    assert line, "JSONL process unexpectedly exited"
    reply = json.loads(line)
    assert reply["ok"] is ok, reply
    return reply


try:
    send("invalid json", ok=False)
    send({"cmd": "missing_command", "id": 17}, ok=False)
    assert send({"cmd": "info"})["rules"]["spin_mode"] == "all-mini+"
    send({"cmd": "configure", "search": {"max_depth": 2, "beam_width": 8,
         "max_nodes": 1000, "time_budget_ms": 0}})
    rng = random.Random(918)
    pieces = []
    for _ in range(12):
        bag = list("IOTLJSZ")
        rng.shuffle(bag)
        pieces.extend(bag)
    current = 6
    state = send({"cmd": "state", "state": {"current": pieces[0], "queue": pieces[1:6]}})["state"]
    cleared = 0
    for turn in range(24):
        if turn == 8:
            state = send({"cmd": "enqueue", "packet": {"amount": 2, "hole": 4}})["state"]
        legal = send({"cmd": "actions"})["actions"]
        found = send({"cmd": "search", "id": turn})
        assert found["id"] == turn and found["action"] in legal, found
        transition = send({"cmd": "apply", "action": found["action"]})
        state = transition["state"]
        assert not state["dead"] and state["pieces_placed"] == turn + 1
        cleared += transition["events"]["lines"]
        assert len(state["queue"]) <= 4
        while len(state["queue"]) < 5:
            state["queue"].append(pieces[current])
            current += 1
        state = send({"cmd": "state", "state": state})["state"]
    assert cleared > 0, "fixture should exercise actual line clears"
    send({"cmd": "configure", "rules": {"spin_mode": "all+"}})
    assert send({"cmd": "info"})["rules"]["spin_mode"] == "all+"
    assert send({"cmd": "search"})["action"] is not None
    process.stdin.close()
    assert process.wait(timeout=10) == 0
    assert process.stderr.read() == "", "unexpected non-protocol diagnostics"
    print(f"protocol_smoke: {requests} requests, 24 legal locks, {cleared} lines, error recovery passed")
finally:
    if process.poll() is None:
        process.kill()
        process.wait()
