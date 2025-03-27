# BasicMatmul Example Readme
## 代码组织
```
├── 00_basic_matmul
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   ├── basic_matmul_autotune.py # 自动寻优文件
│   └── basic_matmul.cpp # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# cd [代码仓路径]/build/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./00_basic_matmul 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```
- 运行自动寻优，
```
# python3 basic_matmul_autotune.py
```
可能的执行结果如下，
```
No.1: 24.532μs, {'L1TileShape': 'GemmShape<64, 64, 128>', 'L0TileShape': 'GemmShape<64, 64, 64>'}
No.0: 27.693μs, {'L1TileShape': 'GemmShape<64, 64, 64>', 'L0TileShape': 'GemmShape<64, 64, 64>'}
No.2: 16.986μs, {'L1TileShape': 'GemmShape<64, 128, 128>', 'L0TileShape': 'GemmShape<64, 128, 64>'}
No.3: 20.192μs, {'L1TileShape': 'GemmShape<128, 128, 128>', 'L0TileShape': 'GemmShape<128, 128, 64>'}
No.4: 21.540μs, {'L1TileShape': 'GemmShape<128, 64, 128>', 'L0TileShape': 'GemmShape<128, 64, 64>'}
Best config: No.2
compare success.
```
该结果表示在`optimized_matmul_autotune.py`中预设搜索空间中的最优参数组合为：
L1TileShape: GemmShape<64, 128, 128>
L0TileShape: GemmShape<64, 128, 64>