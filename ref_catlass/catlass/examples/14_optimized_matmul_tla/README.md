# OptimizedMatmul Example Readme
## 代码组织
```
├── 14_optimized_matmul_tla
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── optimized_matmul_tla.cpp # 主文件
```
## 示例说明
该用例总体设计与06_optimized_matmul相同，区别为使用了TLA相关抽象，故做相关示例说明
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 14_optimized_matmul_tla
# cd [代码仓路径]/build/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./14_optimized_matmul_tla 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```