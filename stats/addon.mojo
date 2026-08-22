## stats/addon.mojo — SIMD statistics on Float64Arrays
##
## Two functions demonstrating SIMD aggregate computation:
##   1. stats(data)           → {mean, stddev, min, max, p50, p95, p99}
##   2. histogram(data, bins) → Float64Array of bin counts
##
## Build:  npm run build:stats
## Run:    node stats/stats.js

from std.algorithm.functional import vectorize
from std.sys import simd_width_of
from std.math import sqrt
from std.memory.alloc import unsafe_alloc

from napi.types import NapiEnv, NapiValue
from napi.error import throw_js_error
from napi.bindings import NapiBindings, Bindings, init_bindings
from napi.framework.js_number import JsNumber
from napi.framework.js_int32 import JsInt32
from napi.framework.js_object import JsObject
from napi.framework.js_typedarray import JsTypedArray
from napi.framework.js_arraybuffer import JsArrayBuffer
from napi.framework.args import CbArgs
from napi.framework.register import fn_ptr, ModuleBuilder
from napi.framework.runtime import init_async_runtime, parallelize_safe


# --- SIMD sum/min/max in one pass ---------------------------------------------

comptime PARALLEL_THRESHOLD = 4096
comptime NUM_WORKERS = 4

def _simd_sum_min_max(
    data: Pointer[Float64, MutAnyOrigin], start: Int, end: Int
) -> Array[Float64, 3]:
    # Returns [sum, min, max]
    var vsum: Float64 = 0.0
    var vmin: Float64 = data[unsafe_offset=start]
    var vmax: Float64 = data[unsafe_offset=start]
    var base = start

    def compute[width: Int](offset: Int) {mut vsum, mut vmin, mut vmax, imm data, imm base}:
        var chunk = data.unsafe_load[width=width](base + offset)
        vsum += chunk.reduce_add()
        var cmin = chunk.reduce_min()
        var cmax = chunk.reduce_max()
        if cmin < vmin:
            vmin = cmin
        if cmax > vmax:
            vmax = cmax

    vectorize[simd_width_of[DType.float64]()](end - start, compute)
    var result = Array[Float64, 3](fill=vsum)
    result[1] = vmin
    result[2] = vmax
    return result^


def _parallel_sum_min_max(
    data: Pointer[Float64, MutAnyOrigin], size: Int
) -> Array[Float64, 3]:
    if size < PARALLEL_THRESHOLD:
        return _simd_sum_min_max(data, 0, size)
    var p_sum = unsafe_alloc[Float64](NUM_WORKERS)
    var p_min = unsafe_alloc[Float64](NUM_WORKERS)
    var p_max = unsafe_alloc[Float64](NUM_WORKERS)

    # chunk_size is recomputed INSIDE the worker on purpose. A `var` local used
    # only from a `capturing` closure is invisible to the compiler's liveness
    # analysis ("assignment was never used"), so its store can be eliminated and
    # the closure then reads a garbage stack slot. Capturing only parameters
    # (data, size) and locals that are also read after the closure (p_*) is safe.
    def worker(wid: Int) capturing:
        var chunk_size = size // NUM_WORKERS
        var s = wid * chunk_size
        var e = s + chunk_size if wid < NUM_WORKERS - 1 else size
        var partial = _simd_sum_min_max(data, s, e)
        p_sum[unsafe_offset=wid] = partial[0]
        p_min[unsafe_offset=wid] = partial[1]
        p_max[unsafe_offset=wid] = partial[2]

    parallelize_safe[worker](NUM_WORKERS)
    var total_sum: Float64 = 0.0
    var total_min: Float64 = p_min[unsafe_offset=0]
    var total_max: Float64 = p_max[unsafe_offset=0]
    for i in range(NUM_WORKERS):
        total_sum += p_sum[unsafe_offset=i]
        if p_min[unsafe_offset=i] < total_min:
            total_min = p_min[unsafe_offset=i]
        if p_max[unsafe_offset=i] > total_max:
            total_max = p_max[unsafe_offset=i]
    p_sum.unsafe_free()
    p_min.unsafe_free()
    p_max.unsafe_free()
    var result = Array[Float64, 3](fill=total_sum)
    result[1] = total_min
    result[2] = total_max
    return result^


# --- SIMD variance pass -------------------------------------------------------

def _simd_sum_sq_diff(
    data: Pointer[Float64, MutAnyOrigin], start: Int, end: Int, mean: Float64
) -> Float64:
    var sum_sq: Float64 = 0.0
    var base = start

    def compute[width: Int](offset: Int) {mut sum_sq, imm data, imm base, imm mean}:
        var chunk = data.unsafe_load[width=width](base + offset)
        var diff = chunk - mean
        sum_sq += (diff * diff).reduce_add()

    vectorize[simd_width_of[DType.float64]()](end - start, compute)
    return sum_sq


def _parallel_sum_sq_diff(
    data: Pointer[Float64, MutAnyOrigin], size: Int, mean: Float64
) -> Float64:
    if size < PARALLEL_THRESHOLD:
        return _simd_sum_sq_diff(data, 0, size, mean)
    var partials = unsafe_alloc[Float64](NUM_WORKERS)

    # See _parallel_sum_min_max: chunk_size must not be a closure-only local.
    def worker(wid: Int) capturing:
        var chunk_size = size // NUM_WORKERS
        var s = wid * chunk_size
        var e = s + chunk_size if wid < NUM_WORKERS - 1 else size
        partials[unsafe_offset=wid] = _simd_sum_sq_diff(data, s, e, mean)

    parallelize_safe[worker](NUM_WORKERS)
    var total: Float64 = 0.0
    for i in range(NUM_WORKERS):
        total += partials[unsafe_offset=i]
    partials.unsafe_free()
    return total


# --- Quickselect for percentiles -----------------------------------------------

def _partition(arr: Pointer[Float64, MutAnyOrigin], lo: Int, hi: Int) -> Int:
    var pivot = arr[unsafe_offset=hi]
    var i = lo
    for j in range(lo, hi):
        if arr[unsafe_offset=j] <= pivot:
            var tmp = arr[unsafe_offset=i]
            arr[unsafe_offset=i] = arr[unsafe_offset=j]
            arr[unsafe_offset=j] = tmp
            i += 1
    var tmp = arr[unsafe_offset=i]
    arr[unsafe_offset=i] = arr[unsafe_offset=hi]
    arr[unsafe_offset=hi] = tmp
    return i


def _quickselect(arr: Pointer[Float64, MutAnyOrigin], size: Int, k: Int) -> Float64:
    var left = 0
    var right = size - 1
    while left < right:
        var pivot_idx = _partition(arr, left, right)
        if pivot_idx == k:
            return arr[unsafe_offset=k]
        elif pivot_idx < k:
            left = pivot_idx + 1
        else:
            right = pivot_idx - 1
    return arr[unsafe_offset=left]


# --- Full stats computation ---------------------------------------------------

def _compute_stats(
    data: Pointer[Float64, MutAnyOrigin], size: Int
) -> Array[Float64, 7]:
    # Returns [mean, stddev, min, max, p50, p95, p99]

    # Pass 1: SIMD sum/min/max
    var smm = _parallel_sum_min_max(data, size)
    var mean = smm[0] / Float64(size)

    # Pass 2: SIMD variance
    var sum_sq = _parallel_sum_sq_diff(data, size, mean)
    var stddev = sqrt(sum_sq / Float64(size))

    # Pass 3: Percentiles via quickselect on a copy
    var copy = unsafe_alloc[Float64](size).as_unsafe_any_origin()
    for i in range(size):
        copy[unsafe_offset=i] = data[unsafe_offset=i]

    var p50_idx = Int(Float64(size - 1) * 0.5)
    var p95_idx = Int(Float64(size - 1) * 0.95)
    var p99_idx = Int(Float64(size - 1) * 0.99)

    var p50 = _quickselect(copy, size, p50_idx)
    var p95 = _quickselect(copy, size, p95_idx)
    var p99 = _quickselect(copy, size, p99_idx)
    copy.unsafe_free()

    var result = Array[Float64, 7](fill=mean)
    result[1] = stddev
    result[2] = smm[1]  # min
    result[3] = smm[2]  # max
    result[4] = p50
    result[5] = p95
    result[6] = p99
    return result^


# --- N-API callbacks ----------------------------------------------------------

def stats_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_one(env, info)
        var b = r.b
        if not JsTypedArray.is_typedarray(b, env, r.arg0):
            throw_js_error(env, "stats requires a Float64Array argument")
            return NapiValue(unsafe_from_address=Int(0))
        var ta = JsTypedArray(r.arg0)
        var size = Int(ta.length(b, env))
        if size == 0:
            throw_js_error(env, "stats requires non-empty array")
            return NapiValue(unsafe_from_address=Int(0))
        var ptr = ta.data_ptr_float64(b, env)
        var s = _compute_stats(ptr, size)

        var obj = JsObject.create(b, env)
        obj.set_property(b, env, "mean",   JsNumber.create(b, env, s[0]).value)
        obj.set_property(b, env, "stddev", JsNumber.create(b, env, s[1]).value)
        obj.set_property(b, env, "min",    JsNumber.create(b, env, s[2]).value)
        obj.set_property(b, env, "max",    JsNumber.create(b, env, s[3]).value)
        obj.set_property(b, env, "p50",    JsNumber.create(b, env, s[4]).value)
        obj.set_property(b, env, "p95",    JsNumber.create(b, env, s[5]).value)
        obj.set_property(b, env, "p99",    JsNumber.create(b, env, s[6]).value)
        return obj.value
    except:
        throw_js_error(env, "stats failed")
        return NapiValue(unsafe_from_address=Int(0))


def histogram_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_two(env, info)
        var b = r.b
        if not JsTypedArray.is_typedarray(b, env, r.arg0):
            throw_js_error(env, "histogram requires a Float64Array and bin count")
            return NapiValue(unsafe_from_address=Int(0))
        var ta = JsTypedArray(r.arg0)
        var size = Int(ta.length(b, env))
        var bins = Int(JsInt32.from_napi_value(b, env, r.arg1))
        if size == 0 or bins <= 0:
            throw_js_error(env, "histogram requires non-empty array and positive bins")
            return NapiValue(unsafe_from_address=Int(0))
        var ptr = ta.data_ptr_float64(b, env)

        # Find min/max via SIMD
        var smm = _parallel_sum_min_max(ptr, size)
        var vmin = smm[1]
        var vmax = smm[2]

        # Bin the data
        var counts = unsafe_alloc[Float64](bins)
        for i in range(bins):
            counts[unsafe_offset=i] = 0.0
        var range_val = vmax - vmin
        if range_val == 0.0:
            counts[unsafe_offset=0] = Float64(size)
        else:
            for i in range(size):
                var idx = Int((ptr[unsafe_offset=i] - vmin) / range_val * Float64(bins))
                if idx >= bins:
                    idx = bins - 1
                counts[unsafe_offset=idx] += 1.0

        # Create Float64Array result
        var ab = JsArrayBuffer.create(b, env, UInt(bins * 8))
        var ab_ptr = ab.data_ptr(b, env).unsafe_bitcast[Float64]()
        for i in range(bins):
            ab_ptr[unsafe_offset=i] = counts[unsafe_offset=i]
        counts.unsafe_free()
        var result_ta = JsTypedArray.create_float64(b, env, ab.value, 0, UInt(bins))
        return result_ta.value
    except:
        throw_js_error(env, "histogram failed")
        return NapiValue(unsafe_from_address=Int(0))


# --- Module entry point -------------------------------------------------------

@export("napi_register_module_v1")
def register_module(env: NapiEnv, exports: NapiValue) abi("C") -> NapiValue:
    try:
        init_async_runtime()
    except:
        pass

    var bindings_ptr = unsafe_alloc[NapiBindings](1)
    try:
        var bindings = NapiBindings()
        init_bindings(bindings)
        bindings_ptr.unsafe_write(bindings^)
    except:
        bindings_ptr.unsafe_free()
        throw_js_error(env, "stats: failed to resolve N-API symbols")
        return exports
    var cb_data = bindings_ptr.unsafe_bitcast[NoneType]().as_unsafe_any_origin()

    var stats_ref = stats_fn
    var hist_ref = histogram_fn

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        m.method("stats", fn_ptr(stats_ref))
        m.method("histogram", fn_ptr(hist_ref))
        m.flush()
    except:
        throw_js_error(env, "stats: failed to register exports")

    return exports
