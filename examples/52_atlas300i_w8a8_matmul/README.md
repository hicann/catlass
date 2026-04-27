# Atlas300IW8A8Matmul Example Readme
## 代码组织
```
├── 52_atlas300i_w8a8_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── w8a8_matmul.cpp # 主文件
```
## 功能介绍
- 将`int8`类型的A、B矩阵Matmul后，与大小为`(1, n)`的`int32`类型的bias向量相加，最后与大小为`(1, n)`的`float`类型的scale向量相乘将结果量化为`half`类型。
- 当前实现支持RowMajor、ColumnMajor数据排布。对于NZ格式，仅支持A矩阵为zN排布，B矩阵为nZ排布。
## 使用示例
- 获取代码后，编译相应的算子可执行文件，可参考[quickstart](../../docs/1_Practice/01_quick_start.md#算子编译)。本用例为atlas300I算子，编译时需加-DCATLASS_ARCH=2002
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 52_atlas300i_w8a8_matmul -DCATLASS_ARCH=2002
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./52_atlas300i_w8a8_matmul 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```