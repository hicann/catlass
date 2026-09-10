# PaddingSplitkMatmul Example Readme

## 代码组织

```text
├── 22_padding_splitk_matmul
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── padding_splitk_matmul.cpp # 主文件
```

## 功能介绍

该算子支持A、B矩阵做完padding后，附加k轴方向切分用于分核，在`m`，`n`较小时提高Cube核利用率。

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 22_padding_splitk_matmul
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./22_padding_splitk_matmul 256 512 1024 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## 模板推荐场景

本样例为 `09_splitk_matmul` 的 padding 版本（A/B 做 padding 后再切 K 分核）。
推荐 MNK 范围：`ceil(M/128)×ceil(N/256) ≤ 12 且 K > 5120`，或 `ceil(M/128)×ceil(N/256) ≤ 2 且 K > 1024`，且 K 非 512B 对齐。具体以 `102_dynamic_optimized_matmul` 泛化工程的路由结论为准。
