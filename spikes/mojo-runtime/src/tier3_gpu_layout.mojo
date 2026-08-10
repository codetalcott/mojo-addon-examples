## Tier 3 — GPU + `layout` package (TileTensor), still no `linalg`.
##
## Adds the `layout` package on top of std.gpu.host. Same kernel functionally,
## but uses TileTensor wrappers instead of raw Pointer args. Tests the
## marginal cost of `layout` in dynamic dependencies — does the package live
## inside the MAX runtime or sit on top of std.gpu?

from std.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys import has_accelerator
from layout import Coord, Idx, TileTensor, TensorLayout, row_major


comptime BLOCK = 256


def _double_kernel[Layout: TensorLayout](
    src: TileTensor[DType.float32, Layout, MutAnyOrigin],
    dst: TileTensor[DType.float32, Layout, MutAnyOrigin],
    n_i64: Int64,
):
    # Int/UInt are not DevicePassable as of Mojo 26.6 — kernel params must
    # be fixed-width. Convert back to Int for indexing.
    var n = Int(n_i64)
    comptime assert src.flat_rank == 1, "expected 1D tensor"
    var tid = Int(global_idx.x)
    if tid < n:
        dst[tid] = src[tid] * 2.0


@export("spike_tier3_gpu_double")
def gpu_double(n: Int) abi("C") -> Float32:
    comptime if not has_accelerator():
        return -1.0

    try:
        var ctx = DeviceContext()
        var dev_src = ctx.enqueue_create_buffer[DType.float32](n)
        var dev_dst = ctx.enqueue_create_buffer[DType.float32](n)
        dev_src.enqueue_fill(3.0)

        var layout = row_major(Coord(Idx(n)))
        var t_src = TileTensor[DType.float32](dev_src, layout)
        var t_dst = TileTensor[DType.float32](dev_dst, layout)

        var grid = ceildiv(n, BLOCK)
        comptime kernel = _double_kernel[type_of(layout)]
        ctx.enqueue_function[kernel, kernel](
            t_src,
            t_dst,
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
