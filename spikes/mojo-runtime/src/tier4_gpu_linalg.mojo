## Tier 4 — GPU + `layout` + `linalg.matmul[target="gpu"]` — current
## packages/rag baseline. Tests the marginal cost of `linalg` over `layout`.

from std.gpu.host import DeviceContext
from std.sys import has_accelerator
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul import matmul as linalg_matmul


comptime dtype = DType.float32


@export("spike_tier4_gpu_matmul", ABI="C")
def gpu_matmul(m: Int, k: Int, n: Int) -> Float32:
    comptime if not has_accelerator():
        return -1.0

    try:
        var ctx = DeviceContext()
        var dev_a = ctx.enqueue_create_buffer[dtype](m * k)
        var dev_b = ctx.enqueue_create_buffer[dtype](k * n)
        var dev_c = ctx.enqueue_create_buffer[dtype](m * n)
        dev_a.enqueue_fill(1.0)
        dev_b.enqueue_fill(2.0)

        var tt_a = TileTensor[dtype](dev_a, row_major(Coord(Idx(m), Idx(k))))
        var tt_b = TileTensor[dtype](dev_b, row_major(Coord(Idx(k), Idx(n))))
        var tt_c = TileTensor[dtype](dev_c, row_major(Coord(Idx(m), Idx(n))))

        linalg_matmul[target="gpu"](tt_c, tt_a, tt_b, Optional(ctx))

        var host_c = ctx.enqueue_create_host_buffer[dtype](m * n)
        ctx.enqueue_copy(host_c, dev_c)
        ctx.synchronize()

        return host_c.unsafe_ptr()[0]
    except:
        return -2.0
