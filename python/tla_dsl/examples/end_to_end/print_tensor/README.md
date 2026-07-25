# Tensor `tla.print` example

This example compiles and launches one single-block kernel that prints a known
16-element prefix from a GM-resident `float32[8,4]` tensor or an AIV
UB-resident `float32[4,8]` view. The output verifier
requires exactly one native CANN record and prints only the stable public fields:

```text
tla.print dtype=float32 shape=[8,4] count=16 values=[0.0, ..., 15.0]
compile_ok=True
launch_ok=True
output_ok=True
```

Run GM on either supported core scope:

```bash
python print_tensor.py --run --arch-scope aiv.c310 --device 0 --block 1 --force-recompile
python print_tensor.py --run --arch-scope aic.c310 --device 0 --block 1 --force-recompile
```

Run the UB base-address and aligned-offset cases on AIV. The kernel explicitly
uses a 32-byte row width and completes the producer-side GM-to-UB copy before
printing:

```bash
python print_tensor.py --run --storage ub --case base --arch-scope aiv.c310 --device 0 --block 1 --force-recompile
python print_tensor.py --run --storage ub --case aligned-offset --arch-scope aiv.c310 --device 0 --block 1 --force-recompile
```

## Support matrix

| Property | Supported | Rejected |
| --- | --- | --- |
| Core scope | AIV-only `aiv.c310`; AIC-only `aic.c310` | Mixed AIC/AIV; regionless use |
| Launch grid | One block | Multi-block launches |
| Storage | GM; 32-byte-aligned UB on AIV | L1, L0, or host invocation |
| Dtype | `float32` | Every other dtype |
| Shape | Any non-empty static shape | Dynamic or empty |
| Length | 1–16 elements, no greater than tensor size | Zero, negative, over 16, or over tensor size |
| Layout | Contiguous row-major | Strided, tiled, or column-major |
| Baseline | Ascend950PR, CANN 9.1.0-beta.3 or later | Other device/CANN combinations are not declared |

`tla.print(value, length, /)` derives dtype and shape from the tensor. Length is
an element count, not a byte count. Omitting it prints the complete tensor when
the tensor contains at most 16 elements; larger tensors require an explicit
length. Missing, malformed, duplicated, truncated, reordered, or extra native
records fail the example instead of producing synthetic output.

For UB, the effective address—including a pointer or view offset—must be
statically proven 32-byte aligned. `tla.print` does not insert producer
synchronization; callers must complete writes to UB before printing.
