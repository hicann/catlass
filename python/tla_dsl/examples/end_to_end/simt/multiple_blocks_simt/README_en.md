# MultipleBlocksSimt Example Readme

## Description

- Function: performs a one-dimensional vector addition in SIMT mode, partitioning the data across two levels at once, blocks and threads.
- Formula:

  $$
    C_{i} = A_{i} + B_{i}
  $$

  where $A$, $B$, and $C$ are one-dimensional vectors in the shape of `(n,)`.

- The point of this example is not the arithmetic but the two levels of parallelism:

  | Level | Interface | Where it is queried |
  | --- | --- | --- |
  | Across blocks | `tla.arch.block_idx()` / `tla.arch.block_num()` | *Outside* the SIMT region, captured into it as launch arguments |
  | Within a block | `tla.arch.thread_idx()` / `tla.arch.thread_block_dim()` | *Inside* the SIMT region |

  The thread geometry is the three-dimensional `(64, 4, 2)`, that is 512 threads per block, so that the y and z components are exercised as well. The kernel first linearizes the three-dimensional thread index into `tid`, then walks the array with `gid = block_idx * tdim + tid` and `stride = block_num * tdim`, so that all threads of the whole launch cover the array exactly once instead of every block redundantly computing all of it.

- Supported products: Ascend 950 series products

## Parameters

| Parameter | Attribute | Shape | dtype | Description |
| --- | --- | --- | --- | --- |
| `gm_a` | Input | `(65536,)` | `f32` | Left addend; the layout is `RowMajor` |
| `gm_b` | Input | `(65536,)` | `f32` | Right addend; the layout is `RowMajor` |
| `gm_c` | Output | `(65536,)` | `f32` | Addition result; the layout is `RowMajor` |

The element count `N_ELE` and the thread geometry `THREADS` are constants inside the example; changing them requires a re-run so that the kernel is recompiled.

## Usage Scope

This example is the minimal demonstration of SIMT partitioning across blocks. It contains no optimizations such as tiling or double buffering.

The output buffer is filled with the sentinel value `-999.0` before the launch, and the number of elements still holding the sentinel afterwards is reported separately as `untouched`. A wrong stride shows up as a non-zero `untouched`, which is a partitioning failure rather than an arithmetic one, so it is reported apart from the precision check.

### Command Line Parameters

```bash
multiple_blocks_simt.py [--device deviceId] [--block-dim blockDim] [--atol atol] [--cache-dir cacheDir] [--force-recompile] [--no-cache]
```

The command line parameters are described as follows:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `deviceId` | `0` | ID of the device on which the program runs |
| `blockDim` | `-1` | Number of AIV cores used; `-1` takes the device's `vector_core_num` |
| `atol` | `1e-4` | Absolute tolerance of the precision comparison |
| `cacheDir` | none | Compilation cache directory, equivalent to setting `CATLASS_DSL_CACHE_DIR`; the default cache location is used when it is not given |
| `--force-recompile` | off | Forces recompilation, equivalent to setting `CATLASS_DSL_FORCE_RECOMPILE=1` |
| `--no-cache` | off | Disables the compilation cache, equivalent to setting `CATLASS_DSL_CACHE=0` |

### Example

1. Go to the project root directory and set up the runtime environment for the sample code. For details, see [Quick Start](../../../../docs/zh/quick_start.md).

2. Run the operator sample program from the project root directory.

    ```bash
    python python/tla_dsl/examples/end_to_end/simt/multiple_blocks_simt/multiple_blocks_simt.py --device 0
    ```

3. If the following result is displayed, the sample is successfully executed and the precision verification passes:

    ```text
    untouched=0/65536
    passed=True cache_key=...
    ```
