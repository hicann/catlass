# BasicMatmulPreloadZN Example Readme

## 代码组织

```text
├── 21_basic_matmul_preload_zN
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── basic_matmul_preload_zN.cpp # 主文件
```

## 功能介绍

该算子在00_basic_matmul基础上支持B矩阵NZ输入（非转置使用zN，转置使用nZ），并使用MmadAtlasA2Preload的DispatchPolicy。

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 21_basic_matmul_preload_zN
cd /output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./21_basic_matmul_preload_zN 256 512 1024 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## 模板推荐场景

本样例与 `06_optimized_matmul` 为同一模板（`MmadAtlasA2Preload` 调度策略、tile 一致），区别是 B 矩阵输入即 zN 分形格式（NZ 布局），GM→L1 读取 B 时无需 ND2NZ 随路转换。推荐 MNK 范围： `M ≥ 256、N ≥ 256、256 < K ≤ 3072`。
