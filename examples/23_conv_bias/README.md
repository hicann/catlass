# MatmulBias Example Readme
## 代码组织
```
├── 23_conv_bias
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── conv_bias.cpp # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 23_conv_bias
# cd [代码仓路径]/output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./23_conv_bias 2 128 6 34 34 4 3 3 3 1 1 1 1 1 1 0 0 0 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```