# MxMatmul Example Readme

## Description

- Function: performs MX (microscaling) matrix multiplication, in which every 32 elements along the k axis share one E8M0 power-of-two scaling factor.
- Formula:

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} (A_{i,k} s^{A}_{i,\lfloor k/32 \rfloor})(B_{k,j} s^{B}_{\lfloor k/32 \rfloor,j})
    \end{aligned}
  $$

  where $A$ and $B$ are input matrices in the shape of `(m, k)` and `(k, n)`, respectively. $C$ is the output matrix in the shape of `(m, n)`. $s^{A}$ and $s^{B}$ are the E8M0 block scale matrices in the shape of `(m, ceil(k/32))` and `(ceil(k/32), n)`, respectively.
- Supported products: Ascend 950PR&950DT products

## Parameters

| Parameter | Attribute | Shape            | dtype                           | Description                                                                                                                                                                         |
|-----------|-----------|------------------|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `A`       | Input     | `(m,k)`          | `f8e4m3fn/f8e5m2/f4e2m1/f4e1m2` | Left matrix; the layout supports `RowMajor` (default) and `ColumnMajor`, and is fixed to `RowMajor` when the data type is fp4                                                       |
| `scale_A` | Input     | `(m,ceil(k/32))` | `f8e8m0`                        | Block scales of the left matrix, one per 32 elements along the k axis; shares the layout of the left matrix                                                                         |
| `B`       | Input     | `(k,n)`          | `f8e4m3fn/f8e5m2/f4e2m1/f4e1m2` | Right matrix; the layout supports `ColumnMajor` (default) and `RowMajor`, and is fixed to `ColumnMajor` when the data type is fp4; the element width is the same as the left matrix |
| `scale_B` | Input     | `(ceil(k/32),n)` | `f8e8m0`                        | Block scales of the right matrix, one per 32 elements along the k axis; shares the layout of the right matrix                                                                       |
| `C`       | Output    | `(m,n)`          | `f32`                           | Matrix multiplication result; only the `RowMajor` layout is supported                                                                                                               |

## Usage Scope

Both input matrices must have the same element width, so an fp8 matrix cannot be multiplied by an fp4 matrix. Within a width the encodings are free to differ: any fp8 encoding may be paired with any fp8 encoding, and any fp4 encoding with any fp4 encoding.

Recommended range:

- `m ≥ 256, n ≥ 256, k ≥ 256`
- The k axis is a multiple of 32, so that no scale group is partially filled.

## Usage Example

### Command Line Parameters

```bash
mx_matmul.py [--m m] [--n n] [--k k] [--layout-a layoutA] [--layout-b layoutB] [--case case] [--device deviceId] [--block-num blockNum]
```

The command line parameters are described as follows:

| Parameter  | Default Value       | Description                                                                           |
|------------|---------------------|---------------------------------------------------------------------------------------|
| `m`        | `256`               | Size of the m axis of matrix A                                                        |
| `n`        | `512`               | Size of the n axis of matrix B                                                        |
| `k`        | `1024`              | Size of the k axis of matrices A and B                                                |
| `layoutA`  | `row`               | Layout of matrix A, `row` or `col`                                                    |
| `layoutB`  | `col`               | Layout of matrix B, `row` or `col`                                                    |
| `case`     | `f8e4m3fn,f8e4m3fn` | Data types of matrices A and B; repeat the option to run several cases in one process |
| `deviceId` | `0`                 | ID of the device on which the program runs                                            |
| `blockNum` | `-1`                | Number of AIC cores used; `-1` auto-detects full occupancy                            |

### Example

1. Go to the project root directory and set up the runtime environment for the sample code. For details, see [Quick Start](../../../docs/zh/quick_start.md).

2. Run the operator sample program from the project root directory.

    ```bash
    python python/tla_dsl/examples/end_to_end/mx_mmad/mx_matmul.py --m 512 --n 512 --k 512 --device 0
    ```

3. If the following result is displayed, the sample is successfully executed and the precision verification passes:

    ```text
    SUMMARY passed=1/1
    ```
