# OptimizedMatmulTla Example Readme
## 代码组织
```
├── 21_optimized_matmul_ext_tla
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── optimized_matmul_ext_tla.cpp # 主文件
```
## 示例说明
该用例总体设计与14_optimized_matmul_tla相同，区别为增添了B矩阵zN、nZ格式支持，故做相关示例说明
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 21_optimized_matmul_ext_tla
# cd [代码仓路径]/build/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./21_optimized_matmul_ext_tla 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```