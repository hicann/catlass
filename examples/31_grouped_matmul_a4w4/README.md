# GroupedMatmulA4w4 Example Readme
## 代码组织
```
├── 31_grouped_matmul_a4w4
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── grouped_matmul_a4w4.cpp # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 31_grouped_matmul_a4w4
# cd [代码仓路径]/output/bin
# 可执行文件名|group数量|矩阵m轴列表|n轴列表|k轴列表|per-group对应g轴列表(对应k需要能被对应g整除)|Device ID
# Device ID可选，默认为0
./31_grouped_matmul_a4w4 2 "64,64" "128,128" "64,64" "1,1" 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```