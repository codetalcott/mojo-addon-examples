## Tier 2 — GPU via std.gpu.host only: hand-rolled kernel, raw Pointer
## kernel args, no `layout`, no `linalg`. Mirrors the pattern in
## examples/image/addon.mojo's _gpu_kernel_grayscale.
##
## This is the maximally-stripped GPU build. If `layout` and `linalg` bring in
## MAX-licensed dylibs, this tier will *not* link them — leaving only the
## std.gpu.host runtime, which we hypothesize is core Mojo (Apache-2-style
## license) rather than MAX-restricted.

from std.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys import has_accelerator


comptime BLOCK = 256


def _double_kernel(
    src: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    n_i64: Int64,
):
    # Int/UInt are not DevicePassable as of Mojo 26.6 — kernel params must
    # be fixed-width. Convert back to Int for indexing.
    var n = Int(n_i64)
    var tid = Int(global_idx.x)
    if tid < n:
        dst[unsafe_offset=tid] = src[unsafe_offset=tid] * 2.0


@export("spike_tier2_gpu_double")
def gpu_double(n: Int) abi("C") -> Float32:
    comptime if not has_accelerator():
        return -1.0

    try:
        var ctx = DeviceContext()
        var dev_src = ctx.enqueue_create_buffer[DType.float32](n)
        var dev_dst = ctx.enqueue_create_buffer[DType.float32](n)
        dev_src.enqueue_fill(3.0)

        var grid = ceildiv(n, BLOCK)
        ctx.enqueue_function[_double_kernel, _double_kernel](
            dev_src.unsafe_ptr(),
            dev_dst.unsafe_ptr(),
            Int64(n),
            grid_dim=grid,
            block_dim=BLOCK,
        )

        var host_dst = ctx.enqueue_create_host_buffer[DType.float32](n)
        ctx.enqueue_copy(host_dst, dev_dst)
        ctx.synchronize()

        return host_dst.unsafe_ptr()[0]
    except:
        return -2.0
