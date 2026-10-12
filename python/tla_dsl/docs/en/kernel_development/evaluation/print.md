---
nav_order: 30
---

# Using the Print Debugging Interface in CATLASS DSL

CATLASS DSL provides the `tla.print` interface for debugging output inside a kernel. `tla.print` can be used in two ways:

| Usage       | Form                                              | Description                                 |
| ----------- | ------------------------------------------------- | ------------------------------------------- |
| Scalar printing | `tla.print(value)` / `tla.print(fmt, *args)`      | Prints a single scalar, plain string, or format string |
| Tensor printing | `tla.print(tensor)` / `tla.print(tensor, length)` | Prints the entire Tensor or its first `length` elements |

`tla.print` accepts positional arguments only. For Tensor printing, `length` specifies the number of elements rather than the number of bytes. A Tensor with a static shape may omit `length`, while a Tensor with a dynamic shape must explicitly provide `length`.

## Usage Examples

The following examples use `basic_mmad_kernel` from `python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py`. When debugging, **choose only one** of the following two approaches.

### Scalar Printing

Scalar printing is suitable for inspecting runtime integers, floating-point values, block scheduling, and execution flow. When using this approach, the kernel must not contain any `tla.print(tensor, ...)` calls.

```diff
# python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py
    with tla.cube():
+       tla.print(123)  # Print a constant
+       tla.print("Hello world")  # Print a string
        tla.set_flag(l1a0_available)
        ...

        for block_linear in block_range:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n
+           tla.print(
+               "tile={} row={} col={}", block_linear, block_row, block_col
+           )  # Print a formatted string
            ...
            for k_l1 in k_l1_range:
                ...
                for k_l0 in k_l0_range:
                    ...
                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
+                   if k_l1 == 0 and k_l0 == 0:
+                       tla.print("first mmad done")
```

This approach demonstrates three types of printing:

1. `tla.print(123)`: Prints a single scalar.
2. `tla.print("Hello world")`: Prints a plain string.
3. `tla.print("tile={} ...", ...)`: Uses formatted printing for runtime values such as `block_linear`, `block_row`, and `block_col`.

### Tensor Printing

Tensor printing is suitable for inspecting data in GM or UB.

```diff
# python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py
    with tla.cube():
+       tla.print(gm_a, 256)
        tla.set_flag(l1a0_available)
        ...
```

`tla.print(gm_a, 256)` prints 256 consecutive elements of GM input A starting from its valid address. With the default Basic MMAD parameters, `gm_a` has dtype `f16` and shape `[256, 1024]`.

### Tensor Printing Support

- Only rank-1 and rank-2 Tensors are supported.
- The supported data types are `f16`, `f32`, `i8`, `i16`, `i32`, `u8`, `u16`, and `u32`.
- A GM Tensor can be printed in a `tla.cube()` or `tla.vector()` region. A UB Tensor can be printed only in a `tla.vector()` region.
- `length` can be a static integer or a runtime integer SSA value. A static value must be between 1 and 262112 and must not exceed the number of elements in the Tensor.
- The interface prints a contiguous physical prefix starting from the valid address. The logical shape of the Tensor is used only for display. The interface neither gathers elements according to stride nor rearranges a packed layout.
- `tla.print` does not automatically insert producer synchronization for a UB Tensor. Ensure that the relevant data transfer or computation has completed before printing.

## Building and Running

Before running, ensure that the kernel uses only one of the preceding printing approaches and contains no remaining calls that print L1, L0A, L0B, or L0C Tensors. The CATLASS DSL compiler selects the corresponding debugging FIFO path based on the type of printing used in the kernel, with no additional compilation options required. Enabling the force-recompilation environment variable is recommended to ensure that the modified kernel and printing helper code are regenerated:

```bash
cd python/tla_dsl/examples/end_to_end/basic_mmad

# Launch a single block to make the output easier to inspect
CATLASS_DSL_FORCE_RECOMPILE=1 python basic_matmul.py --device 0 --block-num 1
```

## Example Output

### Scalar Printing Output

Scalar, plain-string, and formatted output all use the `TLA printf:` prefix:

```text
TLA printf: core=0 block=0 x=123
TLA printf: core=0 block=0 Hello world
TLA printf: core=0 block=0 tile=0 row=0 col=0
TLA printf: core=0 block=0 first mmad done
...
passed=True cache_key=<cache-key>
kernel.o=<cache-dir>/kernel.o
```

### Tensor Printing Output

With the default Basic MMAD shape and default `f16` configuration, once the Tensor FIFO is parsed correctly, the output includes the data type, shape, number of printed elements, and element values:

```text
tla.print dtype=f16 shape=[256,1024] count=256 values=[-1.66015625, -2.26953125, 3.0703125, 0.9375, ...]
passed=True cache_key=<cache-key>
kernel.o=<cache-dir>/kernel.o
```

The actual element values, `cache_key`, and artifact paths depend on the runtime environment. When multiple blocks are launched, the printing order across cores may also vary.

## Notes

- `tla.print` can be used only within a `tla.cube()` or `tla.vector()` region.
- A kernel can use either scalar printing or Tensor printing, but not both. Plain strings and format strings both count as scalar printing.
- `tla.print` may significantly affect kernel performance. Use it only during debugging and remove all debugging code before production runs.
- Tensor printing supports GM, UB, and AIC L1/L0C Tensors on C310. For L1, only a 32-byte-aligned zN/nZ dense prefix is supported, the C0 dimension must occupy 32 bytes, and at most 8 elements can be printed; for L0C, only a 32-byte-aligned L0Clayout 16x16 complete fractal (256 elements) is supported. Both support `f16`, `f32`, `i8`, `i16`, `i32`, `u8`, `u16`, and `u32`. This local-buffer path depends on the verified CANN 9.1 DebugTunnel ABI, and L1 additionally requires the driver to provide `halGetMaxResMapType`, `halResMap`, and `halResUnmap`. L0A/L0B are not currently supported.
- When debugging accuracy with a large shape, use `--block-num 1` to reduce the number of cores printing concurrently and prevent output from different blocks from being interleaved. This does not guarantee a reduction in the total number of lines printed within a tile.
- The underlying implementation of the print debugging interface calls the AscendC debugging interfaces. For more information, see the following documentation:
  - [AscendC::DumpTensor](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/Basic-API/Debugging-APIs/DumpTensor.md)
  - [AscendC::printf](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/Basic-API/Debugging-APIs/printf.md)
