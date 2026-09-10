# PaddingMatmul Example Readme

## 代码组织

```text
├── 04_padding_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── padding_matmul.cpp # 主文件
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 04_padding_matmul
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./04_padding_matmul 256 512 1024 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## 模板推荐场景

推荐 MNK 范围：`M ≥ 256、N ≥ 256、256 < K ≤ 3072`，且 K 非 512B 对齐，具体以 `102_dynamic_optimized_matmul` 泛化工程的路由结论为准。
