# MultiOpsSimt Example Readme

## Description

- Function: runs every per-thread scalar operation inside one SIMT region, over every supported element type, to verify the behaviour and precision of the `tla.simt_*` family.
- The example contains two kernels, because the per-thread operations split by type:

  | Kernel | Element types | Contents |
  | --- | --- | --- |
  | `multi_ops_float_simt` | `f32` / `f16` / `bf16` | The arithmetic half, plus a math half |
  | `multi_ops_int_simt` | `i32` / `i16` / `i8` | The arithmetic half only, with `//` for the divide |

- Formula of the arithmetic half, the same in both kernels:

  $$
    \begin{aligned}
    t_{i} &= \mathrm{clamp}\left(\frac{(A_{i} + B_{i} - C_{i}) \times D_{i}}{E_{i}},\ -8,\ 8\right) \\
    \mathit{arith}_{i} &= \mathrm{cast}\left(\begin{cases} t_{i} & A_{i} > B_{i} \\ t_{i} + 1 & \text{otherwise} \end{cases}\right)
    \end{aligned}
  $$

  These map in order to `tla.simt_add`, `tla.simt_sub`, `tla.simt_mul`, `tla.simt_div`, `tla.simt_max`, `tla.simt_min`, `tla.simt_cmp`, and `tla.simt_where`, and finally to `tla.simt_cast`, which round-trips the value through the other width, narrowing once and widening once.

- Formula of the math half, in the float kernel only:

  $$
    \mathit{math}_{i} = \sqrt{|\mathit{arith}_{i}|} + e^{G_{i}} + \ln H_{i} + A_{i}^{2}
  $$

  These map in order to `tla.simt_sqrt`, `tla.simt_abs`, `tla.simt_exp`, `tla.simt_log`, and `tla.simt_pow`.

- Supported products: Ascend 950 series products

## Parameters

Float kernel `multi_ops_float_simt`:

| Parameter | Attribute | Shape | dtype | Description |
| --- | --- | --- | --- | --- |
| `gm_a` | Input | `(1024,)` | `f32/f16/bf16` | First operand of the arithmetic half, and the base of `tla.simt_pow` |
| `gm_b` | Input | `(1024,)` | `f32/f16/bf16` | Second operand of the arithmetic half, and the right-hand side of the comparison |
| `gm_c` | Input | `(1024,)` | `f32/f16/bf16` | Subtrahend |
| `gm_d` | Input | `(1024,)` | `f32/f16/bf16` | Multiplier |
| `gm_e` | Input | `(1024,)` | `f32/f16/bf16` | Divisor; fixed to a power of two and never zero in this example |
| `gm_g` | Input | `(1024,)` | `f32/f16/bf16` | Input of `exp`, kept small so that it does not overflow |
| `gm_h` | Input | `(1024,)` | `f32/f16/bf16` | Input of `log`, kept strictly positive |
| `gm_arith` | Output | `(1024,)` | `f32/f16/bf16` | Result of the arithmetic half |
| `gm_math` | Output | `(1024,)` | `f32/f16/bf16` | Result of the math half |

Integer kernel `multi_ops_int_simt`:

| Parameter | Attribute | Shape | dtype | Description |
| --- | --- | --- | --- | --- |
| `gm_a` | Input | `(1024,)` | `i32/i16/i8` | First operand of the arithmetic half, and the left-hand side of the comparison |
| `gm_b` | Input | `(1024,)` | `i32/i16/i8` | Second operand of the arithmetic half, and the right-hand side of the comparison |
| `gm_c` | Input | `(1024,)` | `i32/i16/i8` | Subtrahend |
| `gm_d` | Input | `(1024,)` | `i32/i16/i8` | Multiplier |
| `gm_e` | Input | `(1024,)` | `i32/i16/i8` | Divisor; fixed to 2 and never zero in this example |
| `gm_arith` | Output | `(1024,)` | `i32/i16/i8` | Result of the arithmetic half |

All inputs are built on the Host side, with deliberately small ranges so that every intermediate value fits in `i8` as well.

## Usage Scope

This example runs on a single block (`block_num=1`) with 128 threads, which stride over the array with `tla.range(tid, N_ELE, nthreads)`. It contains no multi-core partitioning and no pipeline optimization. For multi-core partitioning, see [multiple_blocks_simt](../multiple_blocks_simt/README_en.md).

Precision conventions:

- The arithmetic half is compared **exactly**. The divisor is a power of two and the operands are small integers, so the result is bit-identical for every type.
- The math half is compared with a tolerance, since transcendentals cannot be bit-identical between the device and the Host. `f32` uses `rtol=1e-5, atol=1e-4`; `f16` and `bf16` use `rtol=5e-2, atol=5e-2`.

Two lowering behaviours are worth noting:

- `//` lowers to `arith.divsi`, which truncates toward zero, whereas Python's `//` floors, so the reference implementation uses `rounding_mode="trunc"`.
- bf16 has no transcendental unit, so `tla-vector-region` evaluates those operations in f32 and rounds the result back to bf16.

The example only uses the math operations that survive the current pipeline: `sin`/`cos`, `floor`/`ceil`/`round`, and `log2`/`exp2` have no instruction on this target and are therefore not covered.

### Command Line Parameters

```bash
multi_ops_simt.py [--device deviceId] [--dtype dtype] [--cache-dir cacheDir] [--force-recompile] [--no-cache]
```

The command line parameters are described as follows:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `deviceId` | `0` | ID of the device on which the program runs |
| `dtype` | `f32` | Element type: `f32`, `f16`, `bf16`, `i32`, `i16`, `i8`, or `all`. It also selects which kernel runs; `all` runs all six types in one process |
| `cacheDir` | none | Compilation cache directory, equivalent to setting `CATLASS_DSL_CACHE_DIR`; the default cache location is used when it is not given |
| `--force-recompile` | off | Forces recompilation, equivalent to setting `CATLASS_DSL_FORCE_RECOMPILE=1` |
| `--no-cache` | off | Disables the compilation cache, equivalent to setting `CATLASS_DSL_CACHE=0` |

### Example

1. Go to the project root directory and set up the runtime environment for the sample code. For details, see [Quick Start](../../../../docs/zh/quick_start.md).

2. Run the operator sample program from the project root directory.

    ```bash
    python python/tla_dsl/examples/end_to_end/simt/multi_ops_simt/multi_ops_simt.py --dtype all --device 0
    ```

3. If the following result is displayed, the sample is successfully executed and the precision verification passes:

    ```text
    --- multi_ops_simt n=1024 block=128 dtypes=f32,f16,bf16,i32,i16,i8 ---
       f32: arith_ok=True math_ok=True untouched=0/2048 -> PASS
       f16: arith_ok=True math_ok=True untouched=0/2048 -> PASS
      bf16: arith_ok=True math_ok=True untouched=0/2048 -> PASS
       i32: arith_ok=True untouched=0/1024 -> PASS
       i16: arith_ok=True untouched=0/1024 -> PASS
        i8: arith_ok=True untouched=0/1024 -> PASS
    passed=True
    ```
