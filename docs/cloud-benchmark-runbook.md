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

### Troubleshooting signals

If your results come back dramatically different from the table above, something is off:

- **GPU column much faster than CPU SIMD**: you may have accidentally benchmarked device-resident data, or Phase 3 improvements landed and should be documented.
- **GPU column much slower than above** (e.g. stats 10M < 2× JS): `DeviceContext` is probably being recreated per call. Check the instance_data caching path in the addon's `register_module`.
- **`Context leak detected` warnings in stderr**: N-API finalizer for the cached `DeviceContext` is misbehaving. Not fatal but investigate.
