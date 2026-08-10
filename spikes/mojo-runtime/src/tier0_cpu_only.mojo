## Tier 0 — minimal Mojo shared library: pure compute, no GPU, no MAX.
##
## Imports only std.math. Establishes the unconditional baseline of dynamic
## dependencies that any Mojo `--emit shared-lib` build pulls in regardless of
## what the source code does.

from std.math import sqrt


@export("spike_tier0_compute")
def compute(x: Float64) abi("C") -> Float64:
    return sqrt(x) * 2.0
