# Tensor `tla.print` example

This example compiles and launches one single-block kernel that prints a known
16-element prefix from a static or dynamic-shaped GM-resident `float32[8,4]`
tensor, or from a static/dynamic-shaped AIV UB view. The output verifier
requires exactly one native CANN record for ordinary cases and prints only the
stable public fields:

```text
tla.print dtype=float32 subblock=0 shape=[8,4] count=16 values=[0.0, ..., 15.0]
compile_ok=True
launch_ok=True
output_ok=True
```

Kernels use `with tla.vector():` (AIV). Vector-core output includes the logical
C310 vector `subblock` (`0` or `1`). AIC/AIV is inferred from the region at
compile time; there is no Host `--core-type` switch.

Run GM:

```bash
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --device 0 --block-num 1
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --dynamic-shape --device 0 --block-num 1
```

Exercise a tensor print inside runtime control flow. `--enabled 0`
takes the false branch and validly emits no tensor record. With `--enabled 1`,
the runtime-bounded loop emits one record for each executed iteration; this
example verifies each available record rather than requiring a fixed count:

```bash
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --case dynamic-control-flow --enabled 0 --repeats 2 --device 0 --block-num 1
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --case dynamic-control-flow --enabled 1 --repeats 2 --device 0 --block-num 1
```

Exercise raw physical-prefix semantics with column-major GM storage. The
logical output shape remains `[8,4]`, while values follow the contiguous
transposed backing buffer:

```bash
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --layout column-major --device 0 --block-num 1
```

Exercise the exact 262,112-element GM capacity boundary (redirect the large
canonical record when running in automation):

```bash
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --case capacity --device 0 --block-num 1 > capacity.log
```

Run the UB base-address and aligned-offset cases. The kernel explicitly
uses a 32-byte row width and completes the producer-side GM-to-UB copy before
printing:

```bash
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --storage ub --case base --device 0 --block-num 1
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --storage ub --case aligned-offset --device 0 --block-num 1
CATLASS_DSL_FORCE_RECOMPILE=1 python print_tensor.py --run --storage ub --case dynamic --device 0 --block-num 1
```

## Support matrix

| Property | Supported | Rejected |
| --- | --- | --- |
| Core scope | AIV via `tla.vector()` (inferred from IR) | Host `--core-type` / `compile(core_type=...)` |
| Launch grid | One block | Multi-block launches |
| Storage | GM; 32-byte-aligned effective UB address on AIV | L1, L0, or host invocation |
| Dtype | `float32` | Every other dtype |
| Shape | Rank-1/rank-2 static or runtime-shaped tensors | Empty, rank above 2, or mismatched runtime metadata |
| Length | Static or integer-SSA 1–262,112 element prefix, no greater than runtime tensor size | Zero, negative, over 262,112, or over tensor size |
| Dynamic control flow | GM print sites under runtime `if` and `tla.range` | multi-block or dynamic-shape variants of this example case |
| Layout | Row-major, column-major, padded/strided, and packed TLA layouts | Layouts outside the TLA layout enum |
| Baseline | Ascend950PR, CANN 9.1.0 or later | Other device/CANN combinations are not declared |

`tla.print(value, length, /)` derives dtype and the concrete runtime shape from
the tensor. Length is an element count, not a byte count, and may be a Python
integer or integer SSA value. Dynamic-shaped tensors require an explicit
length. Static tensors may omit it when the complete tensor fits the
262,112-element `float32` FIFO capacity. The maximum
comes from the 1 MiB CANN ring, its 48-byte shape TLV, 72-byte tensor TLV, and
32-byte payload alignment.

For a print site nested in dynamic control flow, the disabled path may emit no
record and a loop may emit the same static site repeatedly. The FIFO is
best-effort: once it fills, later records can be absent. The example accepts
those zero/repeated outcomes but rejects every emitted record whose public
dtype, position, subblock, shape, count, or values do not match the call site.
Malformed FIFO data and unknown print identities remain execution errors.

The dynamic examples pass both the first extent and print length as scalar
kernel arguments. Current pointer-only host tensor arguments do not carry
runtime memref extents.

Like Ascend C `DumpTensor`, values are read as a contiguous physical prefix
from the effective address. The logical tensor shape controls display grouping
but does not gather through strides or reorder packed storage. Runtime guards
reject invalid counts and misaligned effective addresses by emitting no native
record, which the host reports as an execution error. `tla.print` does not
insert producer synchronization; callers must complete writes to UB before
printing.
