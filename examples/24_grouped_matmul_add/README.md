# GroupedMatmulAdd Example Readme
## 代码组织
```
├── 24_grouped_matmul_add
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── grouped_matmul_add.cpp # 主文件
```
## 示例说明
- 本示例演示分组矩阵乘法并在结果上进行逐元素加法。
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 24_grouped_matmul_add
# cd [代码仓路径]/output/bin
# 可执行文件名 group数量|m轴|n轴|k轴|Device ID
./24_grouped_matmul_add 128 512 1024 2048 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```
