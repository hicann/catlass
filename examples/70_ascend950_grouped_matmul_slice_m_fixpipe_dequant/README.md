# GroupedMatmulSliceMCubeDequant Example Readme
## 代码组织
```
├── 70_ascend950_grouped_matmul_slice_m_fixpipe_dequant
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── grouped_matmul_slice_m_fixpipe_dequant.cpp # 主文件
```
## 功能介绍
该算子支持A矩阵在m轴切分，然后和B矩阵按照group分组进行矩阵乘。之后进行per_tensor或per_channel的fixpipe随路反量化操作。

A/B矩阵为int8类型，scale为float，输出结果为half。

## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)，本用例为Ascend950算子，编译时需加-DCATLASS_ARCH=3510
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 70_ascend950_grouped_matmul_slice_m_fixpipe_dequant -DCATLASS_ARCH=3510
# cd [代码仓路径]/output/bin
# 可执行文件名|group数量|矩阵m轴|n轴|k轴|量化模式|Device ID
# 量化模式可选0或1，0表示per_tensor，1表示per_channel
# Device ID可选，默认为0
./70_ascend950_grouped_matmul_slice_m_fixpipe_dequant 128 512 1024 2048 0 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```