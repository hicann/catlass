# QuantMatmul with fixpipe Example Readme
## 代码组织
```
├── 31_quant_matmul_fp
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── quant_matmul_fp.cpp # 主文件
```
## 功能介绍
使用fixbuf实现反量化的随路实现（仅支持fp16）  
**注意： ENABLE_UNIT_FLAG 必须配置为false**
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 31_quant_matmul_fp
# cd [代码仓路径]/output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./31_quant_matmul_fp 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```