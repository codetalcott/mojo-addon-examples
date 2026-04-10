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

## What the results should show

These are **predictions, not promises** — actual numbers depend on HBM bandwidth, driver version, and kernel launch overhead.

| Kernel | Current M4 Metal | Predicted H100 | Reasoning |
|---|---|---|---|
| `statsGpu` at 10M | 4.2× JS | **20-50× JS**, ≥ Mojo SIMD | Float32 reduction is bandwidth-bound; 3TB/s HBM dominates CPU SIMD |
| `grayscaleGpu` at 4K | 0.8× JS | **30-100× JS** | Elementwise on 33MB — H100 does this in ~10μs |
| `countByteGpu` at 100MB | 2.3× JS | **50-150× JS**, ≥ CPU SIMD | Tree reduction + 100MB DMA is HBM-bound; expected to beat CPU |

If any of these come back *close to* or *below* the CPU numbers, something's wrong with the kernel or the DeviceContext is being recreated per call — flag it and investigate.
