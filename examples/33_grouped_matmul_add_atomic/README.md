# GroupedMatmulAddAtomic Example Readme
## 代码组织
```
├── 33_grouped_matmul_add_atomic
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── grouped_matmul_add_atomic.cpp   # 主文件
```
## 功能介绍
该算子支持分组矩阵乘法并在结果上进行逐元素加法。
## 使用示例
example使用
- 第一步，编译
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
```
# 编译指定用例
bash scripts/build.sh 33_grouped_matmul_add_atomic
```
- 第二步，执行算子
```
# cd [代码仓路径]/output/bin
# 可执行文件名 group数量|m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./33_grouped_matmul_add_atomic 128 512 1024 2048 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```