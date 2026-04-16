#!/usr/bin/env python3
"""Zero-copy verification: does MAX v26's output tensor expose a GPU
pointer we can hand to Mojo's DeviceBuffer(owning=False)?"""
import sys
import numpy as np

def header(s):
    print(f"\n=== {s} ===")

from max.dtype import DType
from max.graph import Graph, TensorType, DeviceRef, ops
from max.engine import InferenceSession
from max import driver

# --- Pick a device
header("device")
try:
    dev = driver.Accelerator()
    print(f"Accelerator(): {dev}  label={dev.label}  api={dev.api}  arch={dev.architecture_name}")
except Exception as e:
    print(f"Accelerator failed ({e}), using CPU")
    dev = driver.CPU()

# --- Build tiny graph (elementwise mul)
header("graph")
dref = DeviceRef.GPU() if dev.label == 'gpu' else DeviceRef.CPU()
with Graph(
    "tiny",
    input_types=[
        TensorType(DType.float32, shape=[4], device=dref),
        TensorType(DType.float32, shape=[4], device=dref),
    ],
) as g:
    a, b = g.inputs
    g.output(a * b)
print("graph built")

# --- Compile + run
header("compile + run")
session = InferenceSession(devices=[dev])
model = session.load(g)
print(f"Model: {type(model).__name__}")

a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
result = model.execute(a_np, b_np)
print(f"execute returned: {type(result)}  len={len(result) if hasattr(result, '__len__') else 'n/a'}")
r0 = result[0]
print(f"r0: {type(r0).__name__}  module={type(r0).__module__}")

# --- Full attribute dump
header("r0 attributes")
attrs = [a for a in dir(r0) if not a.startswith('_')]
print(f"public: {attrs}")
dunders = [a for a in dir(r0) if a.startswith('__') and ('dlpack' in a.lower() or 'array' in a.lower() or 'cuda' in a.lower())]
print(f"dunder device/interchange: {dunders}")

# --- Device info
header("device of output")
if hasattr(r0, 'device'):
    print(f"r0.device = {r0.device}")
    print(f"r0.device.label = {r0.device.label}")
if hasattr(r0, 'is_host'):
    print(f"r0.is_host = {r0.is_host}")

# --- Try DLPack (primary zero-copy path)
header("DLPack probe")
if hasattr(r0, '__dlpack_device__'):
    try:
        d = r0.__dlpack_device__()
        print(f"__dlpack_device__() = {d}")
    except Exception as e:
        print(f"__dlpack_device__ raised: {e}")
if hasattr(r0, '__dlpack__'):
    try:
        cap = r0.__dlpack__()
        print(f"__dlpack__() returned: type={type(cap).__name__}  repr={repr(cap)[:100]}")
    except Exception as e:
        print(f"__dlpack__ raised: {e}")

# --- Try round-tripping via Buffer.from_dlpack (returns to a MAX Buffer on given device)
header("round-trip via Buffer.from_dlpack")
if hasattr(r0, '__dlpack__'):
    try:
        buf = driver.Buffer.from_dlpack(r0)
        print(f"Buffer.from_dlpack(r0) = {buf}  device={buf.device}  dtype={buf.dtype}  shape={buf.shape}")
        print(f"buf attrs: {[a for a in dir(buf) if not a.startswith('_')][:30]}")
    except Exception as e:
        print(f"from_dlpack failed: {e}")

# --- Sanity: values
header("numeric sanity")
try:
    print(f"r0.to_numpy() = {r0.to_numpy()}")
except Exception as e:
    print(f"to_numpy failed: {e}")

# --- What does a Buffer actually expose re: pointer?
header("Buffer details")
try:
    buf = driver.Buffer.from_numpy(a_np).to(dev)
    print(f"host->device Buffer: device={buf.device}  shape={buf.shape}  dtype={buf.dtype}")
    attrs = [a for a in dir(buf) if not a.startswith('_')]
    print(f"attrs: {attrs}")
    # Check mmap which might give us raw access
    if hasattr(buf, 'mmap'):
        print(f"mmap method: {buf.mmap}")
    # Try DLPack from Buffer
    if hasattr(buf, '__dlpack__'):
        cap = buf.__dlpack__()
        print(f"Buffer.__dlpack__() = type={type(cap).__name__}")
    if hasattr(buf, '__dlpack_device__'):
        print(f"Buffer.__dlpack_device__() = {buf.__dlpack_device__()}")
except Exception as e:
    print(f"Buffer probe failed: {e}")
    import traceback; traceback.print_exc()
