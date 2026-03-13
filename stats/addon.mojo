## stats/addon.mojo — SIMD statistics on Float64Arrays
##
## Two functions demonstrating SIMD aggregate computation:
##   1. stats(data)           → {mean, stddev, min, max, p50, p95, p99}
##   2. histogram(data, bins) → Float64Array of bin counts
##
## Build:  pixi run bash stats/build.sh
## Run:    node stats/stats.js

from std.algorithm.functional import vectorize, parallelize
from std.sys import simd_width_of
from std.math import sqrt
from std.memory import alloc

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
from napi.framework.runtime import init_async_runtime


# --- SIMD sum/min/max in one pass ---------------------------------------------

comptime PARALLEL_THRESHOLD = 4096
comptime NUM_WORKERS = 4

fn _simd_sum_min_max(
    data: UnsafePointer[Float64, MutAnyOrigin], start: Int, end: Int
) -> InlineArray[Float64, 3]:
    # Returns [sum, min, max]
    var vsum: Float64 = 0.0
    var vmin: Float64 = data[start]
    var vmax: Float64 = data[start]
    var base = start
    fn compute[width: Int](offset: Int) unified {mut}:
        var chunk = data.load[width=width](base + offset)
        vsum += chunk.reduce_add()
        var cmin = chunk.reduce_min()
        var cmax = chunk.reduce_max()
        if cmin < vmin:
            vmin = cmin
        if cmax > vmax:
            vmax = cmax
    vectorize[simd_width_of[DType.float64]()](end - start, compute)
    var result = InlineArray[Float64, 3](fill=vsum)
    result[1] = vmin
    result[2] = vmax
    return result^


fn _parallel_sum_min_max(
    data: UnsafePointer[Float64, MutAnyOrigin], size: Int
) -> InlineArray[Float64, 3]:
    if size < PARALLEL_THRESHOLD:
        return _simd_sum_min_max(data, 0, size)
    var chunk_size = size // NUM_WORKERS
    var p_sum = alloc[Float64](NUM_WORKERS)
    var p_min = alloc[Float64](NUM_WORKERS)
    var p_max = alloc[Float64](NUM_WORKERS)
    fn worker(wid: Int) capturing:
        var s = wid * chunk_size
        var e = s + chunk_size if wid < NUM_WORKERS - 1 else size
        var partial = _simd_sum_min_max(data, s, e)
        p_sum[wid] = partial[0]
        p_min[wid] = partial[1]
        p_max[wid] = partial[2]
    parallelize[worker](NUM_WORKERS)
    var total_sum: Float64 = 0.0
    var total_min: Float64 = p_min[0]
    var total_max: Float64 = p_max[0]
    for i in range(NUM_WORKERS):
        total_sum += p_sum[i]
        if p_min[i] < total_min:
            total_min = p_min[i]
        if p_max[i] > total_max:
            total_max = p_max[i]
    p_sum.free()
    p_min.free()
    p_max.free()
    var result = InlineArray[Float64, 3](fill=total_sum)
    result[1] = total_min
    result[2] = total_max
    return result^


# --- SIMD variance pass -------------------------------------------------------

fn _simd_sum_sq_diff(
    data: UnsafePointer[Float64, MutAnyOrigin], start: Int, end: Int, mean: Float64
) -> Float64:
    var sum_sq: Float64 = 0.0
    var base = start
    fn compute[width: Int](offset: Int) unified {mut}:
        var chunk = data.load[width=width](base + offset)
        var diff = chunk - mean
        sum_sq += (diff * diff).reduce_add()
    vectorize[simd_width_of[DType.float64]()](end - start, compute)
    return sum_sq


fn _parallel_sum_sq_diff(
    data: UnsafePointer[Float64, MutAnyOrigin], size: Int, mean: Float64
) -> Float64:
    if size < PARALLEL_THRESHOLD:
        return _simd_sum_sq_diff(data, 0, size, mean)
    var chunk_size = size // NUM_WORKERS
    var partials = alloc[Float64](NUM_WORKERS)
    fn worker(wid: Int) capturing:
        var s = wid * chunk_size
        var e = s + chunk_size if wid < NUM_WORKERS - 1 else size
        partials[wid] = _simd_sum_sq_diff(data, s, e, mean)
    parallelize[worker](NUM_WORKERS)
    var total: Float64 = 0.0
    for i in range(NUM_WORKERS):
        total += partials[i]
    partials.free()
    return total


# --- Quickselect for percentiles -----------------------------------------------

fn _partition(arr: UnsafePointer[Float64, MutAnyOrigin], lo: Int, hi: Int) -> Int:
    var pivot = arr[hi]
    var i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            var tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp
            i += 1
    var tmp = arr[i]
    arr[i] = arr[hi]
    arr[hi] = tmp
    return i


fn _quickselect(arr: UnsafePointer[Float64, MutAnyOrigin], size: Int, k: Int) -> Float64:
    var left = 0
    var right = size - 1
    while left < right:
        var pivot_idx = _partition(arr, left, right)
        if pivot_idx == k:
            return arr[k]
        elif pivot_idx < k:
            left = pivot_idx + 1
        else:
            right = pivot_idx - 1
    return arr[left]


# --- Full stats computation ---------------------------------------------------

fn _compute_stats(
    data: UnsafePointer[Float64, MutAnyOrigin], size: Int
) -> InlineArray[Float64, 7]:
    # Returns [mean, stddev, min, max, p50, p95, p99]

    # Pass 1: SIMD sum/min/max
    var smm = _parallel_sum_min_max(data, size)
    var mean = smm[0] / Float64(size)

    # Pass 2: SIMD variance
    var sum_sq = _parallel_sum_sq_diff(data, size, mean)
    var stddev = sqrt(sum_sq / Float64(size))

    # Pass 3: Percentiles via quickselect on a copy
    var copy = alloc[Float64](size)
    for i in range(size):
        copy[i] = data[i]

    var p50_idx = Int(Float64(size - 1) * 0.5)
    var p95_idx = Int(Float64(size - 1) * 0.95)
    var p99_idx = Int(Float64(size - 1) * 0.99)

    var p50 = _quickselect(copy, size, p50_idx)
    var p95 = _quickselect(copy, size, p95_idx)
    var p99 = _quickselect(copy, size, p99_idx)
    copy.free()

    var result = InlineArray[Float64, 7](fill=mean)
    result[1] = stddev
    result[2] = smm[1]  # min
    result[3] = smm[2]  # max
    result[4] = p50
    result[5] = p95
    result[6] = p99
    return result^


# --- N-API callbacks ----------------------------------------------------------

fn stats_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_one(env, info)
        var b = r.b
        if not JsTypedArray.is_typedarray(b, env, r.arg0):
            throw_js_error(env, "stats requires a Float64Array argument")
            return NapiValue()
        var ta = JsTypedArray(r.arg0)
        var size = Int(ta.length(b, env))
        if size == 0:
            throw_js_error(env, "stats requires non-empty array")
            return NapiValue()
        var ptr = ta.data_ptr(b, env).bitcast[Float64]()
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
        return NapiValue()


fn histogram_fn(env: NapiEnv, info: NapiValue) -> NapiValue:
    try:
        var r = CbArgs.get_bindings_and_two(env, info)
        var b = r.b
        if not JsTypedArray.is_typedarray(b, env, r.arg0):
            throw_js_error(env, "histogram requires a Float64Array and bin count")
            return NapiValue()
        var ta = JsTypedArray(r.arg0)
        var size = Int(ta.length(b, env))
        var bins = Int(JsInt32.from_napi_value(b, env, r.arg1))
        if size == 0 or bins <= 0:
            throw_js_error(env, "histogram requires non-empty array and positive bins")
            return NapiValue()
        var ptr = ta.data_ptr(b, env).bitcast[Float64]()

        # Find min/max via SIMD
        var smm = _parallel_sum_min_max(ptr, size)
        var vmin = smm[1]
        var vmax = smm[2]

        # Bin the data
        var counts = alloc[Float64](bins)
        for i in range(bins):
            counts[i] = 0.0
        var range_val = vmax - vmin
        if range_val == 0.0:
            counts[0] = Float64(size)
        else:
            for i in range(size):
                var idx = Int((ptr[i] - vmin) / range_val * Float64(bins))
                if idx >= bins:
                    idx = bins - 1
                counts[idx] += 1.0

        # Create Float64Array result
        var ab = JsArrayBuffer.create(b, env, UInt(bins * 8))
        var ab_ptr = ab.data_ptr(b, env).bitcast[Float64]()
        for i in range(bins):
            ab_ptr[i] = counts[i]
        counts.free()
        var result_ta = JsTypedArray.create_float64(b, env, ab.value, 0, UInt(bins))
        return result_ta.value
    except:
        throw_js_error(env, "histogram failed")
        return NapiValue()


# --- Module entry point -------------------------------------------------------

@export("napi_register_module_v1", ABI="C")
fn register_module(env: NapiEnv, exports: NapiValue) -> NapiValue:
    try:
        init_async_runtime()
    except:
        pass

    var bindings_ptr = alloc[NapiBindings](1)
    try:
        var bindings = NapiBindings()
        init_bindings(bindings)
        bindings_ptr.init_pointee_move(bindings^)
    except:
        bindings_ptr.free()
        return exports
    var cb_data = bindings_ptr.bitcast[NoneType]()

    var stats_ref = stats_fn
    var hist_ref = histogram_fn

    try:
        var m = ModuleBuilder(env, exports, cb_data)
        m.method("stats", fn_ptr(stats_ref))
        m.method("histogram", fn_ptr(hist_ref))
        m.flush()
    except:
        pass

    return exports
