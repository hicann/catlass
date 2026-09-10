# SplitkMatmul Example Readme

## 代码组织

```text
├── 09_splitk_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── splitk_matmul.cpp # 主文件
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 09_splitk_matmul
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./09_splitk_matmul 256 512 1024 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## 模板推荐场景

本样例为多核切 K 模板（MultiCoreSplitk），针对深 K 窄 C、分块数不足导致核数吃不满的场景，沿 K 切分提升并行度。
推荐 MNK 范围：`基本块数 ceil(M/128)×ceil(N/256) ≤ 12 且 K > 5120`，或 `基本块数 ≤ 2 且 K > 1024`，输入非 512B 对齐时可参考 `22_padding_splitk_matmul`，具体以 `102_dynamic_optimized_matmul` 泛化工程的路由结论为准。
