# Ascend950 FP8 Epilogue Quant Matmul Example Readme

## 功能介绍

- 演示 Ascend950 上的 **FP8 矩阵乘 + epilogue per-token/per-channel 量化**。
- 计算：`out = (A @ B) * perTokenScale * perChannelScale`。
- A、B 元素类型为 `float8_e4m3_t`，scale 为 `float8_e4m3_t`，输出为 FP32。
- 默认布局为 A `RowMajor`、B `RowMajor`，与 `gen_data.py` 生成的数据一致。

## 代码组织
```
├── 75_ascend950_fp8_epilogue_quant_matmul
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── gen_data.py
│   └── ascend950_fp8_epilogue_quant_matmul.cpp
```

## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考 [quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)。本用例为 Ascend950（3510）算子，编译时需加 `-DCATLASS_ARCH=3510`。
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 75_ascend950_fp8_epilogue_quant_matmul -DCATLASS_ARCH=3510
# 生成测试样例（在 examples/75_ascend950_fp8_epilogue_quant_matmul/data 下生成 input/ 与 golden/）
python3 examples/75_ascend950_fp8_epilogue_quant_matmul/gen_data.py 256 256 128
# 可选：--data-root <DIR> 指定在 DIR/data/ 下生成（默认在脚本所在目录下生成）
# 输入参数分别对应 m, n, k
# 执行测试样例
./output/bin/75_ascend950_fp8_epilogue_quant_matmul 256 256 128 0
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID 可选，默认为 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```
