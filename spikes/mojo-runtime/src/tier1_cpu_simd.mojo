## Tier 1 — CPU SIMD + parallelize, still no GPU, no MAX.
##
## Adds std.algorithm (vectorize/parallelize) and std.memory (alloc).
## Tests whether the parallel runtime alone pulls in libAsyncRT* — the user
## hypothesis is that AsyncRT is core Mojo runtime, not MAX-specific.

from std.algorithm.functional import parallelize, vectorize
from std.memory import alloc
from std.sys import simd_width_of


@export("spike_tier1_simd_sum", ABI="C")
def simd_sum(n: Int) -> Float32:
    var ptr = alloc[Float32](n)
    for i in range(n):
        ptr[i] = Float32(i)

    var width = simd_width_of[DType.float32]()
    var sum: Float32 = 0.0
    var i = 0
    while i + width <= n:
        var v = ptr.load[width = simd_width_of[DType.float32]()](i)
        sum += v.reduce_add()
        i += width
    while i < n:
        sum += ptr[i]
        i += 1

    ptr.free()
    return sum


@export("spike_tier1_parallel", ABI="C")
def parallel_op(n: Int) -> Int:
    var counter = alloc[Int](n)

    def worker(idx: Int) capturing:
        counter[idx] = idx * 2

    parallelize[worker](n)

    var total = 0
    for i in range(n):
        total += counter[i]
    counter.free()
    return total
