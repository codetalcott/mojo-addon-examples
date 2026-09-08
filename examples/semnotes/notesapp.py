"""semnotes — semantic search that updates live, from bare synchronous WSGI.

DRAFT / prototype — see README.md for status and how to run it.

One process does the whole job: `POST /notes` embeds the note (a
GIL-holding, CPU-bound call — the pool's workload) and stores text + unit
vector in SQLite; `GET /search` embeds the query and ranks by cosine; the
results page holds a Server-Sent Events stream that the *server* keeps
(`M0-Hold`/`M0-Channel`, the m0serve hold contract — the view only
approves the connection) and every insert reaches every open page on
every worker through `m0pub.publish()`. Under a plain WSGI server the
same file still answers every route — the hold headers are ignored and
live updates degrade to refresh-by-hand.

No framework on purpose: when a route misbehaves there is nothing
between the assertion and the server.

Routes:
    GET  /                       the page
    GET  /health                 {"ok": true, "pid": ..., "backend": ...}
    POST /notes {"text": ...}    embed, store, publish -> {"id": ..., "ms": ...}
    GET  /search?q=...           top-10 cosine matches -> {"results": [...]}
    GET  /events                 SSE hold approval (M0-Hold/M0-Channel)

Configuration (all optional):
    SEMNOTES_DB          SQLite path (default semnotes.db in the CWD)
    SEMNOTES_BACKEND     hash | max               (embedder.py)
    SEMNOTES_DEVICE      cpu | gpu, max backend   (embedder.py)
    SEMNOTES_SPIN_MS     emulate model latency    (embedder.py; smoke only)
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
import threading
import time
from urllib.parse import parse_qs

from embedder import get_embedder

# The wheel (`pip install m0serve`) ships the publisher; a flat `m0pub`
# already importable (e.g. PYTHONPATH into a mojo-http checkout) is the
# development fallback. Anything else — uvicorn, gunicorn — gets the
# no-op: every route still answers, live updates degrade, exactly like
# the hold headers.
try:
    from m0serve import m0pub  # type: ignore
except ImportError:
    try:
        import m0pub  # type: ignore
    except ImportError:

        class m0pub:  # type: ignore
            @staticmethod
            def publish(channel: str, payload: str) -> None:
                pass

CHANNEL = "notes"
TOP_K = 10
JSON_H = [("Content-Type", "application/json")]

_embedder = get_embedder()
DIM = _embedder.dim
_PACK = struct.Struct(f"<{DIM}f")

try:  # optional: MAX installs bring numpy; the app never requires it
    import numpy as _np
except ImportError:
    _np = None


# --- store -------------------------------------------------------------------
# One connection per handler thread (views run on pool threads), WAL so the
# prefork workers share one file, and vectors stored unit-normalized so
# ranking is a dot product.

_DB_PATH = os.environ.get("SEMNOTES_DB", "semnotes.db")
_local = threading.local()


def _conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_DB_PATH, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS notes ("
            " id INTEGER PRIMARY KEY,"
            " text TEXT NOT NULL,"
            " vec BLOB NOT NULL,"
            " created REAL NOT NULL)"
        )
        conn.commit()
        _local.conn = conn
    return conn


def _insert(text: str, vec: list[float]) -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO notes (text, vec, created) VALUES (?, ?, ?)",
        (text, _PACK.pack(*vec), time.time()),
    )
    conn.commit()
    return int(cur.lastrowid)


def _rank(qvec: list[float], k: int) -> list[dict]:
    rows = _conn().execute("SELECT id, text, vec FROM notes").fetchall()
    if not rows:
        return []
    if _np is not None:
        mat = _np.frombuffer(b"".join(r[2] for r in rows), dtype="<f4")
        scores = mat.reshape(len(rows), DIM) @ _np.asarray(qvec, dtype="<f4")
        order = scores.argsort()[::-1][:k]
        return [
            {"id": rows[i][0], "text": rows[i][1], "score": round(float(scores[i]), 4)}
            for i in order
        ]
    scored = []
    for note_id, text, blob in rows:
        vec = _PACK.unpack(blob)
        scored.append((math.fsum(a * b for a, b in zip(qvec, vec)), note_id, text))
    scored.sort(reverse=True)
    return [
        {"id": note_id, "text": text, "score": round(score, 4)}
        for score, note_id, text in scored[:k]
    ]


# --- helpers -----------------------------------------------------------------


def _json(start_response, status: str, obj) -> list[bytes]:
    body = json.dumps(obj).encode("utf-8")
    start_response(status, list(JSON_H) + [("Content-Length", str(len(body)))])
    return [body]


def _read_json(environ) -> dict:
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
    except ValueError:
        length = 0
    raw = environ["wsgi.input"].read(length) if length else b""
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _q(environ, name: str, default: str = "") -> str:
    return parse_qs(environ.get("QUERY_STRING", "")).get(name, [default])[0]


# --- routes ------------------------------------------------------------------


def health(environ, start_response):
    return _json(
        start_response,
        "200 OK",
        {"ok": True, "pid": os.getpid(), "backend": _embedder.name},
    )


def add_note(environ, start_response):
    if environ["REQUEST_METHOD"] != "POST":
        return _json(start_response, "405 Method Not Allowed", {"error": "POST only"})
    text = str(_read_json(environ).get("text", "")).strip()
    if not text:
        return _json(start_response, "400 Bad Request", {"error": "empty text"})
    if len(text) > 2000:
        return _json(start_response, "413 Content Too Large", {"error": "too long"})
    t0 = time.perf_counter()
    vec = _embedder.embed(text)
    embed_ms = (time.perf_counter() - t0) * 1000
    note_id = _insert(text, vec)
    m0pub.publish(
        CHANNEL,
        json.dumps({"id": note_id, "preview": text[:80], "pid": os.getpid()}),
    )
    return _json(
        start_response,
        "200 OK",
        {"id": note_id, "embed_ms": round(embed_ms, 3), "pid": os.getpid()},
    )


def search(environ, start_response):
    query = _q(environ, "q").strip()
    if not query:
        return _json(start_response, "200 OK", {"results": [], "ms": 0})
    t0 = time.perf_counter()
    qvec = _embedder.embed(query)
    results = _rank(qvec, TOP_K)
    return _json(
        start_response,
        "200 OK",
        {
            "results": results,
            "ms": round((time.perf_counter() - t0) * 1000, 3),
            "backend": _embedder.name,
            "pid": os.getpid(),
        },
    )


def events(environ, start_response):
    # The quickstart contract: an ordinary buffered response, two M0-
    # headers, and m0serve holds the connection from here. Under a plain
    # WSGI server the headers are ignored and this is a short 200.
    start_response(
        "200 OK",
        [
            ("Content-Type", "text/event-stream"),
            ("Cache-Control", "no-store"),
            ("M0-Hold", "stream"),
            ("M0-Channel", CHANNEL),
        ],
    )
    return [b": connected\n\n"]


def index(environ, start_response):
    body = PAGE.encode("utf-8")
    start_response(
        "200 OK",
        [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


ROUTES = {
    "/": index,
    "/health": health,
    "/notes": add_note,
    "/search": search,
    "/events": events,
}


def application(environ, start_response):
    handler = ROUTES.get(environ.get("PATH_INFO", "/"))
    if handler is None:
        return _json(start_response, "404 Not Found", {"error": "no such route"})
    return handler(environ, start_response)


# --- the page ----------------------------------------------------------------

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>semnotes</title>
<style>
 body{font:16px system-ui;margin:2rem auto;max-width:38rem;padding:0 1rem}
 input{width:100%;box-sizing:border-box;padding:.4rem;font:inherit}
 li{margin:.35rem 0} .score{color:#888;font-size:.85em;margin-left:.5em}
 #status{color:#888;font-size:.85em}
</style>
<h1>semnotes</h1>
<p id="status">connecting&hellip;</p>
<form id="add"><input id="text" autocomplete="off"
  placeholder="add a note, every open tab re-ranks"></form>
<p><input id="q" autocomplete="off" placeholder="search"></p>
<ul id="results"></ul>
<script>
const q = document.getElementById("q");
const results = document.getElementById("results");
const status = document.getElementById("status");
let timer = null;

async function refresh() {
  const r = await fetch(`/search?q=${encodeURIComponent(q.value)}`);
  const data = await r.json();
  results.replaceChildren(...data.results.map(n => {
    const li = document.createElement("li");
    li.textContent = n.text;
    const s = document.createElement("span");
    s.className = "score";
    s.textContent = n.score.toFixed(3);
    li.appendChild(s);
    return li;
  }));
  if (data.backend)
    status.textContent =
      `live · ${data.backend} backend · ${data.ms.toFixed(1)} ms · pid ${data.pid}`;
}

q.addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(refresh, 150);
});

document.getElementById("add").onsubmit = async e => {
  e.preventDefault();
  const text = document.getElementById("text");
  if (!text.value.trim()) return;
  await fetch("/notes", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text: text.value})});
  text.value = "";
};

// The live half: the server holds this stream; any worker's publish lands
// here, and the page answers by re-running its current search.
const es = new EventSource("/events");
es.onmessage = () => refresh();
es.onopen = () => { status.textContent = "live"; refresh(); };
es.onerror = () => { status.textContent = "stream closed (plain WSGI server?)"; };
</script>"""
