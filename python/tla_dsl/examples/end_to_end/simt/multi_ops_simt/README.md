# MultiOpsSimt Example Readme

## 功能说明

- 算子功能：在一个 SIMT 区域内逐线程执行全部标量运算，覆盖全部受支持的元素类型，用于验证 `tla.simt_*` 系列算子的功能与精度。
- 样例包含两个 kernel，因为逐线程算子按类型分成两组：

  | Kernel | 元素类型 | 计算内容 |
  | --- | --- | --- |
  | `multi_ops_float_simt` | `f32` / `f16` / `bf16` | 算术部分与整型一致，另有数学部分 |
  | `multi_ops_int_simt` | `i32` / `i16` / `i8` | 仅算术部分，除法用 `//` |

- 算术部分的计算公式（两个 kernel 相同）：

  $$
    \begin{aligned}
    t_{i} &= \mathrm{clamp}\left(\frac{(A_{i} + B_{i} - C_{i}) \times D_{i}}{E_{i}},\ -8,\ 8\right) \\
    \mathit{arith}_{i} &= \mathrm{cast}\left(\begin{cases} t_{i} & A_{i} > B_{i} \\ t_{i} + 1 & \text{otherwise} \end{cases}\right)
    \end{aligned}
  $$

  依次对应 `tla.simt_add`、`tla.simt_sub`、`tla.simt_mul`、`tla.simt_div`、`tla.simt_max`、`tla.simt_min`、`tla.simt_cmp`、`tla.simt_where`，最后经 `tla.simt_cast` 在另一位宽上往返一次（窄化一次、加宽一次）。

- 数学部分的计算公式（仅浮点 kernel）：

  $$
    \mathit{math}_{i} = \sqrt{|\mathit{arith}_{i}|} + e^{G_{i}} + \ln H_{i} + A_{i}^{2}
  $$

  依次对应 `tla.simt_sqrt`、`tla.simt_abs`、`tla.simt_exp`、`tla.simt_log`、`tla.simt_pow`。

- 支持产品型号：Ascend 950 系列产品

## 样例参数说明

浮点 kernel `multi_ops_float_simt`：

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `gm_a` | Input | `(1024,)` | `f32/f16/bf16` | 算术部分的第一个操作数，同时是 `tla.simt_pow` 的底数 |
| `gm_b` | Input | `(1024,)` | `f32/f16/bf16` | 算术部分的第二个操作数，同时是比较的右操作数 |
| `gm_c` | Input | `(1024,)` | `f32/f16/bf16` | 减数 |
| `gm_d` | Input | `(1024,)` | `f32/f16/bf16` | 乘数 |
| `gm_e` | Input | `(1024,)` | `f32/f16/bf16` | 除数，样例中固定为 2 的幂且非零 |
| `gm_g` | Input | `(1024,)` | `f32/f16/bf16` | `exp` 的输入，取值较小以免溢出 |
| `gm_h` | Input | `(1024,)` | `f32/f16/bf16` | `log` 的输入，取值严格为正 |
| `gm_arith` | Output | `(1024,)` | `f32/f16/bf16` | 算术部分结果 |
| `gm_math` | Output | `(1024,)` | `f32/f16/bf16` | 数学部分结果 |

整型 kernel `multi_ops_int_simt`：

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `gm_a` | Input | `(1024,)` | `i32/i16/i8` | 算术部分的第一个操作数，同时是比较的左操作数 |
| `gm_b` | Input | `(1024,)` | `i32/i16/i8` | 算术部分的第二个操作数，同时是比较的右操作数 |
| `gm_c` | Input | `(1024,)` | `i32/i16/i8` | 减数 |
| `gm_d` | Input | `(1024,)` | `i32/i16/i8` | 乘数 |
| `gm_e` | Input | `(1024,)` | `i32/i16/i8` | 除数，样例中固定为 2 且非零 |
| `gm_arith` | Output | `(1024,)` | `i32/i16/i8` | 算术部分结果 |

所有输入均在 Host 侧构造，取值范围被刻意压小，保证任何中间结果在 `i8` 上也不溢出。

## 使用范围说明

本样例以单 block（`block_num=1`）、128 线程运行，线程按 `tla.range(tid, N_ELE, nthreads)` 跨步遍历，不含多核切分与流水优化。多核切分请参考 [multiple_blocks_simt](../multiple_blocks_simt/README.md)。

精度约定：

- 算术部分为**精确比对**。除数是 2 的幂、操作数是小整数，因此该部分在任何类型上都可逐位相等。
- 数学部分使用容差比对，超越函数在设备上与 Host 不可能逐位一致；`f32` 用 `rtol=1e-5, atol=1e-4`，`f16`/`bf16` 用 `rtol=5e-2, atol=5e-2`。

两处下降行为需要注意：

- `//` 下降为 `arith.divsi`，向零截断，而 Python 的 `//` 向下取整，因此参考实现使用 `rounding_mode="trunc"`。
- bf16 没有超越函数单元，`tla-vector-region` 会在 f32 上求值再舍入回 bf16。

样例只使用能走通当前流水的数学算子：`sin`/`cos`、`floor`/`ceil`/`round`、`log2`/`exp2` 在该目标上没有对应指令，因此未纳入。

### 命令行参数

```bash
multi_ops_simt.py [--device deviceId] [--dtype dtype] [--cache-dir cacheDir] [--force-recompile] [--no-cache]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `deviceId` | `0` | 指定运行设备 ID |
| `dtype` | `f32` | 元素类型，取值为 `f32`、`f16`、`bf16`、`i32`、`i16`、`i8` 或 `all`；该值同时决定运行哪个 kernel，`all` 表示在一个进程内依次运行全部六种类型 |
| `cacheDir` | 无 | 编译缓存目录，等价于设置 `CATLASS_DSL_CACHE_DIR`；不指定时使用默认缓存位置 |
| `--force-recompile` | 关闭 | 强制重新编译，等价于设置 `CATLASS_DSL_FORCE_RECOMPILE=1` |
| `--no-cache` | 关闭 | 禁用编译缓存，等价于设置 `CATLASS_DSL_CACHE=0` |

### 执行示例

1. 进入项目根目录，参考[快速上手](../../../../docs/zh/quick_start.md)准备样例代码的运行环境。

2. 在项目根目录下执行算子样例程序。

    ```bash
    python python/tla_dsl/examples/end_to_end/simt/multi_ops_simt/multi_ops_simt.py --dtype all --device 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

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
