# ConvBias Example Readme
## 代码组织
```
├── 24_conv_bias
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
|   ├── gen_data.py   # 输入数据及标杆产生文件
│   └── conv_bias.cpp # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 第一步， 首先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入。
```
# python3 ./examples/24_conv_bias/gen_data.py |batch|cin|di|hi|wi|cout|kd|kh|kw|sD|sH|sW|dD|dH|dW|pD|pH|pW|dtype
# 最后一个参数指明数据类型为**float6**或 **bfloat16**
python3 gen_data.py 32 64 1 32 48 128 1 1 1 1 1 1 1 1 1 0 0 0 float16
```
执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据
```
├── data
│   ├── fmap.bin   # 卷积的featureMap
│   ├── weight.bin  # 卷积的weight
|   ├── bias.bin   # 卷积的bias
│   └── golden.bin # cpu计算卷积的标杆结果
```
- 第二步，执行算子，这里需要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。
```
# 编译指定用例
bash scripts/build.sh 24_conv_bias
# cd [代码仓路径]/output/bin
# 可执行文件名 |batch|di|cin1|hi|wi|cin0|cout|kd|kh|kw|sD|sH|sW|dD|dH|dW|pD|pH|pW|Device ID
# Device ID可选，默认为0
./24_conv_bias 32 1 4 32 48 16 128 1 1 1 1 1 1 1 1 1 0 0 0 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```