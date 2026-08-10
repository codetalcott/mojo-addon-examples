## Tier 1 — CPU SIMD + parallelize, still no GPU, no MAX.
##
## Adds std.algorithm (vectorize/parallelize) and std.memory (alloc).
## Tests whether the parallel runtime alone pulls in libAsyncRT* — the user
## hypothesis is that AsyncRT is core Mojo runtime, not MAX-specific.

from std.algorithm.functional import vectorize
from max.algorithm import parallelize
from std.memory.alloc import unsafe_alloc
from std.sys import simd_width_of


@export("spike_tier1_simd_sum")
def simd_sum(n: Int) abi("C") -> Float32:
    var ptr = unsafe_alloc[Float32](n)
    for i in range(n):
        ptr[unsafe_offset=i] = Float32(i)

    var width = simd_width_of[DType.float32]()
    var sum: Float32 = 0.0
    var i = 0
    while i + width <= n:
        var v = ptr.unsafe_load[width = simd_width_of[DType.float32]()](i)
        sum += v.reduce_add()
        i += width
    while i < n:
        sum += ptr[unsafe_offset=i]
        i += 1

    ptr.unsafe_free()
    return sum


@export("spike_tier1_parallel")
def parallel_op(n: Int) abi("C") -> Int:
    var counter = unsafe_alloc[Int](n)

    def worker(idx: Int) capturing:
        counter[unsafe_offset=idx] = idx * 2

    parallelize[worker](n)

    var total = 0
    for i in range(n):
        total += counter[unsafe_offset=i]
    counter.unsafe_free()
    return total
