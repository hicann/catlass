# 87_ascend950_batched_matmul_softmax

## 代码组织

```text
├── 87_ascend950_batched_matmul_softmax
│   ├── CMakeLists.txt                  # CMake 编译文件
│   ├── README.md
│   └── batched_matmul_softmax_tla.cpp  # 主文件
```

## 功能介绍

该示例实现 `S = row_softmax(A * B)`，输入按 batch 组织为
`A[batch, M, K]`、`B[batch, K, N]`，输出为 `S[batch, M, N]`。
A 为 FP16 RowMajor，B 为 FP16 ColumnMajor，S 为 FP16 RowMajor。

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../../docs/zh/1_Practice/01_quick_start.md)
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 87_ascend950_batched_matmul_softmax \
    -DCATLASS_ARCH=3510
cd output/bin
# 基本用法：可执行文件名 batch轴 M轴 K轴 N轴 [block_num [verify]]
./87_ascend950_batched_matmul_softmax 2 128 128 128
```

执行结果如下，说明比对成功。

```text
Compare success.
```

## 使用说明

- 可执行文件参数为 `batch M K N [block_num [verify]]`，不传参数时使用默认 shape。
- `block_num` 可选；传入 0 或不传时，示例根据 batch/M/N 自动选择核数。
- `verify` 可选；默认 `1` 执行 CPU golden 精度校验，传 `0` 时只执行
  kernel，用于大 shape 的 `msprof op` 性能采集。
- B 的逻辑形状为 `(batch, K, N)`，示例内部按 ColumnMajor 物理布局生成并校验输入。
