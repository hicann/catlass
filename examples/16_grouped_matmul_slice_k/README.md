# GroupedMatmulSliceK Example Readme
## 代码组织
```
├── 16_grouped_matmul_slice_k
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── grouped_matmul_slice_k.cpp # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# cd [代码仓路径]/build/bin
# 可执行文件名 group数量|m轴|n轴|k轴|Device ID
./16_grouped_matmul_slice_k 128 512 1024 2048 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```