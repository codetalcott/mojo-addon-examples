# spike/ — embedding-kernel spike

Isolated workspace for the 2-week embedding-kernel spike. See
[../../ideas/embedding-kernel-spike-plan.md](../../ideas/embedding-kernel-spike-plan.md)
for the full plan + gate criteria.

**Status: Day 1 scaffold.** Files compile and round-trip, but `embed_tokens_fn`
is a zero-fill stub. Day 2 replaces it with a real MAX graph forward pass over
MiniLM-L6-v2.

## Layout

```text
spike/
├── src/
│   ├── lib.mojo           # N-API entry point (register_module)
│   └── embed.mojo         # embed_tokens_fn — STUB, returns zeros on Day 1
├── build.sh               # Compiles src/lib.mojo → build/embed.node
├── package.json           # @huggingface/transformers for JS-side tokenization + reference
├── tokenize.js            # JS WordPiece tokenizer → Int32Array token IDs
├── reference.js           # CPU ground-truth generator (runs once on M4)
├── test-roundtrip.js      # Day 1 smoke test (shape only, not correctness)
├── fixtures/              # ground-truth.bin, sanity-set.txt (git-ignored)
└── build/                 # embed.node (git-ignored)
```

## Day 1 — run this sequence

From repo root:

```bash
# Install JS deps (only spike needs @huggingface/transformers — once)
(cd spike && npm install)

# Generate CPU ground truth (run once on M4; git-ignored output)
node spike/reference.js

# Build the Mojo addon
pixi run bash spike/build.sh

# Smoke test: tokenize → embedTokens → assert shape, print first 8 values
node spike/test-roundtrip.js
```

Expected output: `PASS` and a row of eight zeros (stub behavior).
Gate F1 decision happens once `_stub_forward` in
[src/embed.mojo](src/embed.mojo) is replaced with a real MAX graph pass and
the test prints non-zero values close to the reference.

## Lambda Cloud usage

The spike plan's Day 0 section documents the persistent-filesystem +
`GH_TOKEN` bootstrap. Key points for this dir specifically:

- `HF_HOME=/home/ubuntu/persist/model-cache` so model weights download
  once per filesystem, not per instance.
- `npm install` under `spike/` runs against the local
  `node_modules/napi-mojo` — no network needed for the napi framework.
- Commit bench captures to `docs/spike-bench-*.txt` (not `spike/fixtures/`
  which is git-ignored).

## When this dir goes away

If the Day 10 decision is GO: primitives migrate to
[packages/rag/src/kernels.mojo](../packages/rag/src/kernels.mojo), JS glue
to [packages/rag/lib/GpuIndex.js](../packages/rag/lib/GpuIndex.js), and the
`spike/` directory is deleted after merge. If NO-GO: keep `spike/` for the
next person who wants to try — the scaffold is the useful residue.
