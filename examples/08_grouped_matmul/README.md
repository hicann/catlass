# GroupedMatmul Example Readme
## 代码组织
```
├── 08_grouped_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   ├── grouped_matmul.cpp # 主文件
│   └── grouped_matmul_autotune.py # 自动寻优示例文件
```
## 示例说明
- 本grouped_matmul为通用kernel，示例内部为切k情况下的使用
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 08_grouped_matmul
# cd [代码仓路径]/build/bin
# 可执行文件名 group数量|m轴|n轴|k轴|Device ID
./08_grouped_matmul 128 512 1024 2048 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```
- 运行自动寻优，
```
# python3 grouped_matmul_autotune.py
```
可能的执行结果如下，
```
No.0: 112.092μs, {'L1TileShape': 'GemmShape<64, 64, 64>', 'L0TileShape': 'GemmShape<64, 64, 64>'}
No.1: 97.674μs, {'L1TileShape': 'GemmShape<64, 64, 128>', 'L0TileShape': 'GemmShape<64, 64, 64>'}
No.3: 68.534μs, {'L1TileShape': 'GemmShape<128, 128, 128>', 'L0TileShape': 'GemmShape<128, 128, 64>'}
No.4: 80.810μs, {'L1TileShape': 'GemmShape<128, 64, 128>', 'L0TileShape': 'GemmShape<128, 64, 64>'}
No.2: 73.630μs, {'L1TileShape': 'GemmShape<64, 128, 128>', 'L0TileShape': 'GemmShape<64, 128, 64>'}
Best config: No.3
compare success.
```
该结果表示在`grouped_matmul_autotune.py`中预设搜索空间中的最优参数组合为：
```
L1TileShape: GemmShape<128, 128, 128>
L0TileShape: GemmShape<128, 128, 64>
```