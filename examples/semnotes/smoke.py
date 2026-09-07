"""semnotes smoke — wiring, ranking, live fan-out, and loop isolation.

DRAFT. Needs m0serve (`pip install m0serve`, or point M0SERVE at a
binary, e.g. a mojo-http checkout's bin/m0serve):

    python3 examples/semnotes/smoke.py

Starts m0serve on the hash backend (no model, no MAX — deterministic
embeddings are what let the ranking phase assert an exact winner) with
SEMNOTES_SPIN_MS giving every embed a model's worth of GIL-holding
latency, then asserts four things:

  A. the app answers, on the pool, with the expected backend;
  B. search ranks the note whose text is the query first;
  C. a live SSE subscriber hears a POSTed note within seconds, across
     two forked workers (the publish rides the broadcast bus);
  D. while four clients saturate the handler pool with spinning
     searches, the server's own health lane (--health-path, answered on
     the event loop, never a pool job) stays fast.

Phase D is the demonstration phase: it fails on any server that runs the
health check through the same worker threads as the inference views.
"""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time

PORT = 8095
HOST = "127.0.0.1"
SPIN_MS = 80
APP_DIR = os.path.dirname(os.path.abspath(__file__))
M0SERVE = os.environ.get("M0SERVE", "m0serve")


def fail(msg: str, log_path: str) -> None:
    print(f"FAIL: {msg}")
    if os.path.exists(log_path):
        print("=== semnotes.log ===")
        print(open(log_path).read())
    sys.exit(1)


def get(path: str, timeout: float = 10.0):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read() or b"{}")
    finally:
        conn.close()


def post_note(text: str, timeout: float = 10.0):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    try:
        body = json.dumps({"text": text})
        conn.request(
            "POST", "/notes", body, {"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read() or b"{}")
    finally:
        conn.close()


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="semnotes-smoke-")
    log_path = os.path.join(tmp, "semnotes.log")
    env = dict(
        os.environ,
        M0_WORKERS="2",
        SEMNOTES_DB=os.path.join(tmp, "notes.db"),
        SEMNOTES_BACKEND="hash",
        SEMNOTES_SPIN_MS=str(SPIN_MS),
    )
    try:
        proc = subprocess.Popen(
            [
                M0SERVE,
                "notesapp:application",
                "--app-dir", APP_DIR,
                "--port", str(PORT),
                "--realtime",
                "--blocking-threads", "2",
                "--health-path", "/healthz",
            ],
            env=env,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        print(
            f"FAIL: {M0SERVE!r} not found — pip install m0serve, "
            "or set M0SERVE to the binary"
        )
        sys.exit(1)
    try:
        run_phases(log_path)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            fail("server did not exit within 15s of SIGTERM", log_path)
        if code != 0:
            fail(f"server exited {code} after SIGTERM", log_path)
    print("smoke-semnotes OK")


def run_phases(log_path: str) -> None:
    # --- A: up, on the expected backend ------------------------------------
    deadline = time.time() + 30
    status, body = 0, {}
    while time.time() < deadline:
        try:
            status, body = get("/health", timeout=2.0)
            if status == 200:
                break
        except OSError:
            pass
        time.sleep(0.5)
    if status != 200:
        fail("server never answered /health", log_path)
    if body.get("backend") != "hash":
        fail(f"unexpected backend {body.get('backend')!r}", log_path)
    print(f"phase A: up, backend={body['backend']} pid={body['pid']}")

    # --- B: deterministic ranking ------------------------------------------
    notes = [
        "the quick brown fox jumps over the lazy dog",
        "sqlite stores every vector this page ranks",
        "a mojo event loop serves synchronous python",
    ]
    ids = []
    for text in notes:
        status, body = post_note(text)
        if status != 200 or "id" not in body:
            fail(f"POST /notes -> {status} {body}", log_path)
        ids.append(body["id"])
    status, body = get("/search?q=sqlite%20stores%20every%20vector")
    if status != 200 or not body.get("results"):
        fail(f"search answered {status} {body}", log_path)
    top = body["results"][0]
    if top["id"] != ids[1]:
        fail(f"expected note {ids[1]} first, got {top}", log_path)
    scores = [r["score"] for r in body["results"]]
    if scores != sorted(scores, reverse=True):
        fail(f"scores not descending: {scores}", log_path)
    print(f"phase B: ranking OK ({len(body['results'])} results, top {top['score']})")

    # --- C: a held subscriber hears a publish ------------------------------
    sub = socket.create_connection((HOST, PORT), timeout=10)
    sub.sendall(
        b"GET /events HTTP/1.1\r\nHost: smoke\r\nAccept: text/event-stream\r\n\r\n"
    )
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = sub.recv(4096)
        if not chunk:
            fail("SSE connection closed during headers", log_path)
        head += chunk
    if not head.startswith(b"HTTP/1.1 200"):
        fail(f"SSE hold refused: {head[:120]!r}", log_path)
    status, body = post_note("published while a stream is held")
    if status != 200:
        fail(f"phase C POST -> {status}", log_path)
    want = f'"id": {body["id"]}'.replace(" ", "")
    sub.settimeout(10)
    buf = b""
    got = False
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            chunk = sub.recv(4096)
        except socket.timeout:
            break
        if not chunk:
            break
        buf += chunk
        if want.encode() in buf.replace(b" ", b""):
            got = True
            break
    sub.close()
    if not got:
        fail(f"published note {body['id']} never reached the held stream", log_path)
    print(f"phase C: publish {body['id']} reached the held stream (pid {body['pid']})")

    # --- D: the loop's health lane behind a saturated pool ------------------
    stop = threading.Event()

    def storm() -> None:
        while not stop.is_set():
            try:
                get("/search?q=the%20quick%20brown%20fox", timeout=30)
            except OSError:
                return

    threads = [threading.Thread(target=storm, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(1.0)  # let the pool saturate
    samples = []
    try:
        for _ in range(30):
            t0 = time.perf_counter()
            status, _ = get("/healthz", timeout=5.0)
            samples.append((time.perf_counter() - t0) * 1000)
            if status != 200:
                fail(f"/healthz -> {status} under load", log_path)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=35)
    p50 = statistics.median(samples)
    worst = max(samples)
    print(f"phase D: /healthz under pool saturation p50={p50:.1f}ms max={worst:.1f}ms")
    # Every pool thread is inside an 80 ms spin; a health check that became
    # a pool job would queue behind them. The loop lane should be ~1 ms;
    # the bound is generous for shared-runner noise.
    if worst > 500:
        fail(f"health lane stalled behind the pool: max {worst:.1f}ms", log_path)


if __name__ == "__main__":
    main()
