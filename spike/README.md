# spike/ — promoted to `packages/embed/`

The embedding-kernel spike completed 2026-04-17 (GO verdict) and has been
productized as **[`@qkstat/embed`](../packages/embed/)**.

- Package source: [`../packages/embed/`](../packages/embed/)
- Historical day-by-day execution log: [`../docs/embedding-kernel-spike-findings.md`](../docs/embedding-kernel-spike-findings.md)
- Decision artifact / writeup: [`../../ideas/embedding-kernel-spike-writeup.md`](../../ideas/embedding-kernel-spike-writeup.md)
- Plan for next steps: [`../../ideas/post-spike-next-steps.md`](../../ideas/post-spike-next-steps.md)

This directory is kept as a stub during the transition. Residual files
(`bench.js`, `demo-reference.js`, `probe_dlpack.py`) are superseded by
equivalents under `packages/embed/` or are day-1 experiments with no
ongoing role. The directory will be removed in a follow-up commit once
`packages/embed/` is verified end-to-end on H100.
