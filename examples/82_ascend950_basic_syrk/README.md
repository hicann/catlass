# Ascend950 Basic SYRK Example Readme

## 代码组织

```text
├── 82_ascend950_basic_syrk
│   ├── CMakeLists.txt      # CMake 编译文件
│   ├── README.md
│   └── basic_syrk_tla.cpp  # 主文件（host 数据生成、kernel 调度、精度校验）
```

## 使用示例

1. 编译样例（Ascend950 需指定 `CATLASS_ARCH=3510`），可参考[快速入门](../../docs/zh/1_Practice/01_quick_start.md#编译执行)：

    ```bash
    bash scripts/build.sh -DCATLASS_ARCH=3510 82_ascend950_basic_syrk
    ```

2. 切换到可执行文件目录并运行。测试数据随机生成，尺寸由命令行传入：

    ```bash
    cd output/bin
    # 可执行文件名 | m | k | deviceId(可选,默认0)
    ./82_ascend950_basic_syrk 1024 1024 0
    ```

    - `1024`：矩阵 m 轴（$X$ 行数 / $Y$ 边长）
    - `1024`：k 轴（$X$ 列数）
    - `0`：Device ID

3. 执行成功输出：

    ```text
    Compare success.
    ```

## 功能说明

- 算子功能：完成对称秩更新（SYRK），对输入矩阵 $X$ 做自乘，得到对称结果矩阵 $Y$。
- 计算公式：

  $$
  Y = X \cdot X^{T}
  $$

  其中 $X$ 形如 `(M, K)`，$Y$ 形如 `(M, M)` 且满足 $Y = Y^{T}$。

- 本样例面向 Ascend950，在 Basic Matmul 路径上复用 L1/L0 pingpong，并通过 swizzle 调度只计算下三角 Block，再利用 nz2nd / nz2dn 双写补全上三角。

## 参数说明

本样例使用 `SyrkOptions`（定义于 `examples/common/options.hpp`），命令行参数为 `m k [device_id]`：

| 参数名     | 描述                                      | 约束                         |
| ---------- | ----------------------------------------- | ---------------------------- |
| `m`        | 输入矩阵 $X$ 的行数（也即输出 $Y$ 的边长） | 正整数                       |
| `k`        | 输入矩阵 $X$ 的列数                       | 正整数                       |
| `deviceId` | 使用的 NPU 卡 ID（默认 0）                | 在设备 NPU 有效范围内        |

`SyrkOptions::Parse` 会将 problem shape 设为 `(m, m, k)`。

## 算子支持范围

| 参数 | 输入 / 输出 | 数据类型 | 维度 | 数据排布 |
| --- | --- | --- | --- | --- |
| X | 输入 | `bfloat16`，`float16` | `[M, K]` | `layout::RowMajor` |
| Y | 输出 | `bfloat16`，`float16` | `[M, M]` | `layout::RowMajor` |

### 关键模板参数（本样例当前固定值）

| 模板参数   | 说明               | 本样例取值            |
| ---------- | ------------------ | --------------------- |
| `ElementX` | 输入矩阵数据类型   | `bfloat16_t`          |
| `ElementY` | 输出矩阵数据类型   | `bfloat16_t`          |
| `LayoutX`  | 输入 $X$ 排布      | `layout::RowMajor`    |
| `LayoutXt` | $X^{T}$ 视图排布   | `layout::ColumnMajor` |
| `LayoutY`  | 输出 $Y$ 排布      | `layout::RowMajor`    |

## 实现方案

算子整体沿用 Ascend950 Basic Matmul，并采用 `GemmIdentityBlockSwizzle` 调度。由于 $Y$ 为对称阵，只需计算下三角（含对角）基本块，规则如下：

1. `blockCoord.m() < blockCoord.n()` 时，跳过该基本块。
2. `blockCoord.m() == blockCoord.n()` 时，计算后仅用 nz2nd 写入对角位置一次。
3. `blockCoord.m() > blockCoord.n()` 时，计算后双写：

    - 使用 nz2nd 写入 `(blockCoord.m(), blockCoord.n())`；
    - 使用 nz2dn（转置）写入 `(blockCoord.n(), blockCoord.m())`。

L0C→GM 双写路径使用 `M_FIX` 同步，不启用 unitFlag（一次 mmad 的 unitFlag 只能配对一次 Fixpipe）。
