# semnotes — semantic search that updates live

**Status: DRAFT / prototype.** An inference example serving this repo's
MiniLM engine ([packages/embed](../../packages/embed)) behind
[m0serve](https://github.com/codetalcott/mojo-http), composing in one
process what usually takes a server + task queue + message broker:
embedding inference on handler threads, SQLite in the views, and live
updates over server-held SSE streams — from one file of bare synchronous
WSGI.

Open the page in two tabs, add a note in either: every open tab re-ranks
its current search within the second. `POST /notes` embeds and stores
text + unit vector; `m0pub.publish()` crosses every worker on m0serve's
broadcast bus; the page's `EventSource` is a connection the *server*
holds — the view only approved it with `M0-Hold`/`M0-Channel` (m0serve's
hold contract; see its QUICKSTART). Under a plain WSGI server the same
file still answers every route; only the live half degrades.

## Run

```bash
pip install m0serve
m0serve notesapp:application --app-dir examples/semnotes \
  --port 8095 --realtime --blocking-threads 2 --health-path /healthz
# then open http://localhost:8095/
```

The same app under uvicorn (for comparison; holds degrade, updates stop):

```bash
uvicorn --app-dir examples/semnotes --interface wsgi notesapp:application --port 8095
```

## Backends

`SEMNOTES_BACKEND` picks the embedder ([embedder.py](embedder.py)):

- `hash` (default) — deterministic feature hashing, 384 dims, stdlib
  only. Lexical rather than semantic, but every server-side property is
  identical, and determinism is what the smoke's ranking assertions
  stand on. `SEMNOTES_SPIN_MS` gives it a model's GIL-holding latency
  for isolation testing.
- `max` — MiniLM-L6-v2 through `packages/embed`'s `EmbeddingEngine` (a
  MAX graph compiled with `InferenceSession`; `SEMNOTES_DEVICE=cpu|gpu`).
  Needs the environment that runs `packages/embed`. Don't mix backends
  in one database file — the embedding spaces are unrelated.

## Smoke

```bash
python3 examples/semnotes/smoke.py            # m0serve on PATH
M0SERVE=/path/to/bin/m0serve python3 examples/semnotes/smoke.py
```

Four phases: the app answers on the expected backend; ranking is correct
and ordered; a POSTed note reaches a held SSE subscriber across two
forked workers; and — the demonstration phase — the server's health lane
(`--health-path`, answered on the event loop, never a handler-pool job)
stays fast while four clients saturate the pool with spinning searches.
Phase D fails on any server that runs the health check through the same
worker threads as the inference views.

## Why this shape

- Inference stays in request-scoped views on m0serve's handler threads,
  and streaming stays on server-held connections — the split m0serve's
  own measurements sanction (an SSE generator that sleeps on a handler
  thread is the documented trap there).
- SQLite is Python's own `sqlite3` (WAL, one connection per handler
  thread); brute-force cosine over unit vectors is milliseconds at demo
  scale and keeps the example dependency-free by default.
- The `max` backend reuses this repo's engine rather than bundling one,
  so the example and `@qkstat/embed` stay in lockstep.
