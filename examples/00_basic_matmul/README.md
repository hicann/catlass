# BasicMatmul Example Readme
## 代码组织
```
├── 00_basic_matmul
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── basic_matmul.cpp # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 00_basic_matmul
# cd [代码仓路径]/build/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./00_basic_matmul 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```