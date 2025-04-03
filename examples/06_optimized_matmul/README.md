# OptimizedMatmul Example Readme
## 代码组织
```
├── 06_optimized_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   ├── optimized_matmul.cpp # 主文件
│   └── optimized_matmul_autotune.py # 自动寻优示例文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 06_optimized_matmul
# cd [代码仓路径]/build/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./06_optimized_matmul 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```
- 运行自动寻优，
```
# python3 optimized_matmul_autotune.py
```
可能的执行结果如下，
```
No.0: 39.359μs, {'alias1': 'GemmShape<256, 128, 256>, GemmShape<128, 256, 256>', 'alias2': 'GemmShape<256, 128, 64>, GemmShape<128, 256, 64>'}
No.1: 37.281μs, {'alias1': 'GemmShape<64, 256, 256>, GemmShape<256, 64, 256>', 'alias2': 'GemmShape<64, 256, 64>, GemmShape<256, 64, 64>'}
Best config: No.1
compare success.
```
该结果表示在`optimized_matmul_autotune.py`中预设搜索空间中的最优参数组合为：
```
alias1(所标记的代码行): GemmShape<256, 128, 256>, GemmShape<128, 256, 256>
alias2(所标记的代码行): GemmShape<256, 128, 64>, GemmShape<128, 256, 64>
```