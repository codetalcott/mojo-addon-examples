#!/usr/bin/env bash
# scripts/debug-image-linux.sh — TEMPORARY: localize the Linux-only segfault in
# examples/image/test.js. Runs each op in its own process, then progressive
# prefixes of the test sequence, then the full test under gdb for a backtrace.
# macOS arm64 runs the identical sequence clean 5/5, so this only exists to
# interrogate Linux x86_64. Delete once the bug is found.
set -uo pipefail

NODE_BIN="$(command -v node)"
run_case() {  # run_case <label> <js>
  local label="$1" js="$2"
  "$NODE_BIN" -e "
const addon = require('$PWD/examples/image/build/image.node');
const px = new Uint8Array([100,150,200,255]);
const hot = new Uint8Array([200,200,200,255]);
const mid = new Uint8Array([128,128,128,255]);
const white = new Uint8Array(4*4*4).fill(255);
$js
console.log('done');
" > /tmp/case.log 2>&1
  local rc=$?
  echo "  [$rc] $label"
  [ $rc -ne 0 ] && tail -3 /tmp/case.log | sed 's/^/      /'
}

echo "### per-op isolation"
run_case "load only"        ""
run_case "grayscale 1x1"    "addon.grayscale(px,1,1);"
run_case "brightness 0.5"   "addon.brightness(px,1,1,0.5);"
run_case "brightness 2.0"   "addon.brightness(hot,1,1,2.0);"
run_case "threshold 0"      "addon.threshold(mid,1,1,0);"
run_case "threshold 255"    "addon.threshold(mid,1,1,255);"
run_case "blur 4x4 r3"      "addon.blur(white,4,4,3);"
run_case "blur 4x4 r1"      "addon.blur(white,4,4,1);"

echo "### progressive sequence (same process)"
run_case "seq: gray"                    "addon.grayscale(px,1,1);"
run_case "seq: gray+bright"             "addon.grayscale(px,1,1); addon.brightness(px,1,1,0.5);"
run_case "seq: gray+bright+bright"      "addon.grayscale(px,1,1); addon.brightness(px,1,1,0.5); addon.brightness(hot,1,1,2.0);"
run_case "seq: +threshold x2"           "addon.grayscale(px,1,1); addon.brightness(px,1,1,0.5); addon.brightness(hot,1,1,2.0); addon.threshold(mid,1,1,0); addon.threshold(mid,1,1,255);"
run_case "seq: full (=test.js)"         "addon.grayscale(px,1,1); addon.brightness(px,1,1,0.5); addon.brightness(hot,1,1,2.0); addon.threshold(mid,1,1,0); addon.threshold(mid,1,1,255); addon.blur(white,4,4,3);"
run_case "repeat gray x50"              "for (let i=0;i<50;i++) addon.grayscale(px,1,1);"

echo "### full test.js under gdb (backtrace on fault)"
sudo apt-get install -y -q gdb > /dev/null 2>&1 || echo "(gdb install failed; skipping)"
if command -v gdb > /dev/null; then
  cd examples/image
  gdb -batch -quiet \
      -ex "run" \
      -ex "bt 25" \
      -ex "info sharedlibrary image" \
      --args "$NODE_BIN" test.js 2>&1 | tail -45
fi
