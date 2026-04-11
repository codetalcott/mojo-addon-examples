# Cloud H100 Benchmark Runbook

This runbook walks through benchmarking the Mojo GPU kernels on a rented NVIDIA H100. The M4 Metal numbers are "Tier-3 support, integrated GPU, copy-bound" — they don't reflect what these kernels should do on real discrete hardware with high-bandwidth memory. This runbook produces Tier-1 numbers suitable for publication in the main README.

**Cost**: ~$3/hr for a 1× H100 instance. Full runbook execution is under 30 minutes. Budget ~$2 total.

**Prerequisites**:

- An account at [https://lambda.ai](https://lambda.ai) with a payment method or credits.
- An SSH key added to your Lambda account. If you don't have one:
  ```
  ssh-keygen -t ed25519 -C "lambda-mojo-bench"
  cat ~/.ssh/id_ed25519.pub
  ```
  Paste the public key into the Lambda dashboard at **SSH keys → Add SSH key**.

## 1. Provision an H100

1. Go to [https://cloud.lambda.ai/instances](https://cloud.lambda.ai/instances).
2. Click **Launch instance**.
3. Choose instance type: **1× H100 SXM5 80GB** (or **1× H100 PCIe 80GB** if SXM5 is unavailable — both work for this benchmark).
4. Region: pick whichever has availability (US West / US East are usually cheapest).
5. Filesystem: the default **Ubuntu 22.04 + CUDA** image is fine. Do **not** attach a persistent filesystem for this short benchmark.
6. SSH key: select your uploaded key.
7. Click **Launch**. Wait ~30-60 seconds for the instance to be "Running".
8. Copy the **IPv4 address** shown on the instance card.

## 2. SSH in

```
ssh ubuntu@<INSTANCE_IP>
```

First connection will prompt for fingerprint acceptance. Type `yes`.

## 3. Bootstrap script — copy-paste this entire block

This installs pixi, clones the repo, builds all three GPU-enabled addons, runs the benchmarks, and writes the combined output to `~/bench-output.txt`:

```bash
set -e

# Install pixi (takes ~10s)
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"

# Install Node 22 LTS via nvm (Ubuntu default is too old)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install 22
nvm use 22

# Clone and build
git clone https://github.com/codetalcott/mojo-addon-examples.git
cd mojo-addon-examples
pixi install
npm install

# Verify GPU visibility before the long build
nvidia-smi | head -20

# Build all three GPU-enabled addons (build.sh scripts default to sm_90 on Linux x86_64)
pixi run bash stats/build.sh
pixi run bash image/build.sh
pixi run bash simd-search/build.sh

# Also build the two CPU-only addons so `npm test` works
pixi run bash matmul/build.sh
pixi run bash wyhash/build.sh

# Regression tests
echo "=== REGRESSION TESTS ==="            > ~/bench-output.txt
npm test                                  >> ~/bench-output.txt 2>&1

# GPU benchmarks
echo ""                                   >> ~/bench-output.txt
echo "=== STATS BENCHMARK ==="            >> ~/bench-output.txt
node stats/stats.js 2>&1                  >> ~/bench-output.txt

echo ""                                   >> ~/bench-output.txt
echo "=== IMAGE BENCHMARK ==="            >> ~/bench-output.txt
node image/image.js 2>&1                  >> ~/bench-output.txt

echo ""                                   >> ~/bench-output.txt
echo "=== SIMD-SEARCH BENCHMARK ==="      >> ~/bench-output.txt
node simd-search/search.js 2>&1           >> ~/bench-output.txt

echo ""                                   >> ~/bench-output.txt
echo "=== nvidia-smi FINAL ==="           >> ~/bench-output.txt
nvidia-smi                                >> ~/bench-output.txt 2>&1

echo "DONE. Output in ~/bench-output.txt ($(wc -l < ~/bench-output.txt) lines)"
```

Expected duration: ~10-15 minutes (mostly `pixi install` downloading the Mojo nightly, plus benchmark runs).

## 4. Paste the results back

```
cat ~/bench-output.txt
```

Copy the full output and paste it into the chat. I'll parse the numbers and update the READMEs.

## 5. Terminate the instance — **don't skip this step**

```
exit
```

Then in the Lambda dashboard:

1. Go back to [https://cloud.lambda.ai/instances](https://cloud.lambda.ai/instances).
2. Click the **checkbox** next to your instance.
3. Click **Terminate** at the top.
4. Confirm the termination prompt.

**The instance keeps billing until you terminate it.** A forgotten H100 instance costs ~$75/day. Double-check the instance list shows zero instances after termination.

## Fallbacks

### If Lambda has no H100 availability

Try **RunPod** ($1.99/hr — even cheaper):

1. Sign up at [https://runpod.io](https://runpod.io).
2. **Deploy** → **Community Cloud** → filter by `H100 80GB PCIe` or `H100 80GB SXM5`.
3. Template: **RunPod PyTorch 2.4** (has CUDA + SSH pre-configured).
4. Deploy the pod, open the SSH terminal from the RunPod dashboard.
5. Run the same bootstrap script as above (nvidia-smi, pixi install, etc.).
6. When done, click **Stop** then **Terminate** in the pod dashboard.

### If `--target-accelerator sm_90` fails

Some H100 instances report as `sm_90a` (architecture-specific feature set). If you see an error like `GPU architecture 'sm_90a' is not supported`, re-run the builds with an override:

```
STATS_ACCEL="--target-accelerator sm_90a" pixi run bash stats/build.sh
IMAGE_ACCEL="--target-accelerator sm_90a" pixi run bash image/build.sh
SEARCH_ACCEL="--target-accelerator sm_90a" pixi run bash simd-search/build.sh
```

For other NVIDIA GPUs: A100 → `sm_80`, A10 → `sm_86`, L40 → `sm_89`, B100/B200 → `sm_100a`.

## What the results actually showed

We ran this runbook against an **NVIDIA H100 80GB HBM3** via RunPod on 2026-04-10. The original predictions (kept below for the record) were **wrong in every case** — in every benchmark, Mojo CPU SIMD (Xeon Sapphire Rapids AVX-512) beat Mojo GPU on the same host.

| Kernel | Original Prediction | Actual H100 Result | Why the Prediction Missed |
|---|---|---|---|
| `statsGpu` at 10M | 20-50× JS, ≥ CPU SIMD | 7.6× JS, **< CPU SIMD (8.3×)** | Two-pass algorithm + scalar Float64→Float32 cast loop on host + PCIe-bound H2D |
| `grayscaleGpu` at 4K | 30-100× JS | 1.4× JS, **≪ CPU SIMD (4.3×)** | 33MB H2D + 33MB D2H each call; PCIe-capped |
| `countByteGpu` at 100MB | 50-150× JS, ≥ CPU SIMD | 6.0× JS, **≪ CPU SIMD (35.1×)** | 105MB H2D at ~12 GB/s vs AVX-512 DRAM at ~30 GB/s |

**The core error in the original predictions**: they reasoned about H100 bandwidth (3 TB/s HBM3) without accounting for how the data reaches the device. These addons call the GPU as one-shot N-API functions against host-resident data — every call does `alloc → H2D → kernel → D2H → free`. The data never lives on-device long enough to amortize the PCIe copy. On a PCIe Gen4 x16 link (~12 GB/s effective), PCIe is the bottleneck for any kernel that touches each byte only a few times.

Meanwhile CPU SIMD sits directly on DRAM at ~30 GB/s effective via AVX-512, with no copy step. For low-arithmetic-intensity kernels (reductions, elementwise, byte scanning), **CPU wins on single-shot calls against host-resident data — even on a box that has an H100 in it.**

**What would make H100 actually win these benchmarks**:

1. **Persistent device buffers** across many calls (upload once, scan/reduce many times)
2. **Batched N-API API** that processes N inputs per call with async copy-compute overlap
3. **Higher arithmetic intensity**: matmul (O(n³) compute on O(n²) data), convolution, attention — anything where compute dominates transfer

None of those are in scope for Phase 2. They are candidate Phase 3 work.

**The runbook below still works** — you can reproduce these numbers on any H100 or A100 in ~15 minutes for ~$1, and you can use it to validate your own setups or test Phase 3 improvements.

## Phase 3a validation (2026-04-11, H100 80GB HBM3 via RunPod)

Phase 3a.1 shipped `search_cached.node` — a new simd-search addon exposing a handle-based API (`loadGpu`, `countByteHandle`, `releaseGpu`) that uploads a buffer once and reuses it across many queries. We ran the new benchmark on H100 to test the hypothesis that persistent device buffers flip the Phase 2d result.

**Decision gate**: cached GPU at 17MB ≥ 200× JS AND ≥ 1.5× CPU SIMD.

**Actual result**:

| Size | CPU SIMD | GPU one-shot | **GPU cached** | Cached per-call |
|------|---------:|-------------:|---------------:|----------------:|
| 1 MB   | 20.9× |  9.2× |      51.0× |  ~18 μs |
| 17 MB  | 42.9× | 11.3× | **527.8×** |  ~29 μs |
| 105 MB | 33.9× |  5.3× | **1030.7×** | ~94 μs |

Both gates cleared with huge margins: 17 MB cached hit 527.8× JS (2.6× the gate) and **12.3× CPU SIMD** (8× the gate). **Phase 3a hypothesis validated.**

At 105 MB the cached path hits **1.1 TB/s effective bandwidth** — about 37% of the H100's 3 TB/s HBM3 peak. That's the highest arithmetic efficiency observed anywhere in this project, for a byte-scan kernel that until this phase was assumed to be a bad fit for GPU entirely.

**What Phase 3a validated, narrowly**: persistent device buffers flip the *single-shot PCIe bottleneck*, not any magical GPU speedup. The kernel itself is unchanged from Phase 2c. The entire improvement came from amortizing `loadGpu` across reuses. If you port a kernel to GPU and don't expose a persistent-buffer path, you will see the Phase 2c numbers: GPU loses to CPU SIMD on host-resident one-shot inputs.

**What Phase 3a did not validate**: whether the same pattern ports cleanly to stats (Float64 → Float32 cast on two passes) or grayscale (elementwise with round-trip D2H). Those are Phase 3b work — the template exists but needs to be translated.

### Reproducing Phase 3a

The standard bootstrap block above builds `search_cached.node` automatically only if you add it to the build list. Add these lines to the bootstrap between the existing builds and the benchmark section:

```bash
pixi run bash simd-search/build_cached.sh

echo ""                                               >> ~/bench-output.txt
echo "=== CACHED REGRESSION TEST ==="                 >> ~/bench-output.txt
node simd-search/test_cached.js                       >> ~/bench-output.txt 2>&1

echo ""                                               >> ~/bench-output.txt
echo "=== SEARCH_CACHED BENCHMARK (PHASE 3a) ==="     >> ~/bench-output.txt
node simd-search/search_cached.js                     >> ~/bench-output.txt 2>&1
```

**Paste caveat observed during the 3a run**: RunPod's web terminal dropped several lines in the middle of a large paste block, leaving the bash session in a broken state that silently aborted after `npm test`. If you see the bootstrap stall or the output file stop growing mid-run, don't re-run the whole bootstrap — open a second web terminal and run the remaining commands directly from `/mojo-addon-examples` (the pod's PWD after git clone). The build outputs are persistent.

### Troubleshooting signals

If your results come back dramatically different from the Phase 2d or Phase 3a tables above, something is off:

- **`countByteHandle` cached column much slower than 500× JS at 17MB**: `DeviceContext` is probably being recreated per query, or the partial-sums buffer isn't pinned. Check `_count_byte_cached` in [simd-search/addon_cached.mojo](../simd-search/addon_cached.mojo) — the only per-call work should be `enqueue_function`, `enqueue_copy` of the partial sums, and `synchronize`.
- **One-shot GPU column much slower than above** (e.g. stats 10M < 2× JS): `DeviceContext` is probably being recreated per call in the one-shot addon. Check the instance_data caching path in that addon's `register_module`.
- **`Context leak detected` warnings in stderr**: N-API finalizer for the cached `DeviceContext` is misbehaving. Not fatal but investigate.
- **Cached RSS growing on `test_cached.js` leak smoke**: expected *without* `--expose-gc` — the finalizer only fires during GC. With `--expose-gc` the 3a `search_cached` leak smoke shows zero growth on H100 and ~1 MB/iter on M4. Phase 3b.3 found the `image_cached` and `stats_cached` leak smokes also show growth on H100 (~3.3 MB/iter and ~0.7 MB/iter respectively), in addition to their M4 counterparts — so the "leaks only happen on synthetic load/release loops" pattern is cross-platform for the transform + multi-buffer templates. Production usage (load once, query many, release once) shows zero growth. Do not attempt to fix the cached addon — the template is unchanged from 3a and the extra leak correlates with the kernel-call-inside-release-cycle pattern.

## Phase 3b validation (2026-04-11, H100 80GB HBM3 SXM5 via RunPod)

Phase 3b.1 and 3b.2 shipped two new cached addons: `image_cached.node` (grayscale transform) and `stats_cached.node` (two-pass Float64 reduction + percentiles). Phase 3b.3 validated both on the same H100 SXM5 pod as Phase 3a and re-ran search_cached for a sanity check.

**Decision gate** (per the Phase 3 strategy doc): cached GPU ≥100× JS AND ≥5× CPU SIMD at the top size (4K for grayscale, 10M for stats).

**Actual result**: both kernels Red against the gate; template and correctness both green.

| Kernel    | Top size         | JS   | CPU SIMD | GPU cached | Verdict                    |
| --------- | ---------------- | ---- | -------- | ---------- | -------------------------- |
| grayscale | 4K RGBA (33 MB)  | 1.0× | 6.3×     | 7.4×       | Red (D2H floor)            |
| stats     | 10M Float64      | 1.0× | 3.6×     | 3.5×       | Red (percentile dominance) |

**Phase 3a reproduced on SXM with slightly better numbers** — countByte cached hit 1146.4× at 105 MB (vs 1030.7× on the PCIe variant from 2026-04-11), and CPU SIMD also improved (93.4× at 17 MB vs 42.9× on the earlier PCIe Xeon). SXM has higher HBM3 bandwidth and this particular pod had a faster Xeon than the earlier PCIe run. Sanity check passed.

**Why grayscale is Red**: the strategy doc's risk #1 predicted this precisely. Every `grayscaleHandle` call still pays full D2H for the 33 MB output at 4K — ~3 ms at PCIe Gen4 ~12 GB/s — an irreducible floor. Amortizing `loadImageGpu` eliminates the H2D leg but not the D2H leg. CPU SIMD does 4K grayscale in 5.05 ms on this host; cached GPU does it in 4.29 ms. A 1.2× edge is the ceiling for this workload shape on this hardware. For transforms, the persistent-buffer template works but the absolute win is fundamentally smaller than for reductions. See [image/README.md](../image/README.md#phase-3b1--cached-grayscale-api-nvidia-h100-80gb-hbm3).

**Why stats is Red**: the cached template successfully eliminates the per-call scalar Float64→Float32 cast and the per-call H2D upload. At 100K and 1M, cached ties or slightly beats CPU SIMD (8.1× vs 8.4× / 8.1× vs 7.5× JS). At 10M, everything collapses because CPU-side percentile quickselect (p50/p95/p99) dominates at ~200–300 ms per call, swamping the ~15 ms of GPU-related per-call savings from caching. The JS→CPU-SIMD ratio itself drops from 7.5× at 1M to 3.6× at 10M for the same reason. The benchmark at 10M is measuring quickselect wall-clock, not cached GPU reduction wall-clock. Moving percentiles to the GPU (parallel quickselect / radix partition) is explicitly out of scope for Phase 3b per the strategy doc and is the natural unblock for this Red. See [stats/README.md](../stats/README.md#phase-3b2--cached-stats-api-nvidia-h100-80gb-hbm3).

### Reproducing Phase 3b

The Phase 3b validation shipped a self-contained bootstrap script at [scripts/runpod-bench-3b.sh](../scripts/runpod-bench-3b.sh) that avoids the "paste caveat" by fetching via a single curl line. On a fresh RunPod H100 web terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/codetalcott/mojo-addon-examples/main/scripts/runpod-bench-3b.sh | bash
```

(If the repo is private, pre-clone with an authenticated URL first — see the script header.) The script installs pixi + node 22, clones the repo, builds all three cached addons plus their oracles, runs regression tests and benchmarks, and writes combined output to `~/bench-cached-3b.txt`. Expected duration ~10–15 minutes (dominated by `pixi install`). Paste `cat ~/bench-cached-3b.txt` back and terminate the pod immediately.

**Red is not failure, for the record.** The strategy doc explicitly calls this out: "Red on either kernel doesn't mean Phase 3 failed — it means that specific kernel shape has an additional bottleneck beyond PCIe, and we document it honestly like Phase 2d did. Phase 3c is unaffected by either outcome because matmul has fundamentally different arithmetic intensity." 3b.1 and 3b.2 validated that the template ports. 3c.1 (tensor-core matmul research) is unblocked by this result.
