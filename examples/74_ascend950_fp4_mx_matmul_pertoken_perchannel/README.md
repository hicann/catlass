# MX FP4 Per-Token Per-Channel Matmul Example Readme

## 功能介绍

- 演示 Ascend950 上的 **MX FP4 矩阵乘 + per-token/per-channel 量化 epilogue**。
- 计算：`D = perTokenScale * (MxScaleA * A @ MxScaleB * B) * perChannelScale`，本示例中 MxScale 固定为 1.0。
- A、B 元素类型为 `float4_e2m1x2_t`，per-token/per-channel scale 为 `float8_e4m3_t`，输出为 FP32。
- 默认布局为 A `RowMajor`、B `RowMajor`，与 `gen_data.py` 生成的数据一致。

## 代码组织
```
├── 74_ascend950_fp4_mx_matmul_pertoken_perchannel
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── gen_data.py
│   └── fp4_mx_matmul_pertoken_perchannel.cpp
```

## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考 [quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)。本用例为 Ascend950（3510）算子，编译时需加 `-DCATLASS_ARCH=3510`。
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 74_ascend950_fp4_mx_matmul_pertoken_perchannel -DCATLASS_ARCH=3510
# 生成测试样例（在 examples/74_ascend950_fp4_mx_matmul_pertoken_perchannel/data 下生成 input/ 与 golden/）
python3 examples/74_ascend950_fp4_mx_matmul_pertoken_perchannel/gen_data.py 256 256 128
# 可选：--data-root <DIR> 指定在 DIR/data/ 下生成（默认在脚本所在目录下生成）
# 输入参数分别对应 m, n, k；当前 n、k 需为偶数
# 执行测试样例
./output/bin/74_ascend950_fp4_mx_matmul_pertoken_perchannel 256 256 128 0
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID 可选，默认为 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```
