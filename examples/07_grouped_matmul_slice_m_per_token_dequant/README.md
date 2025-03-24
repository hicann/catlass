# GroupedMatmul Per Token Dequant Example Readme
## 代码组织
```
├── 07_grouped_matmul_slice_m_per_token_dequant
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── grouped_matmul_slice_m_per_token_dequant.cpp # 主文件
```
## 功能介绍
该算子支持A矩阵在m轴切分，和B矩阵按照group分组进行矩阵乘。之后进行per token的量化操作。
## 使用示例
因为GroupedMatmul参数较多，所以该示例直接在代码中承载输出参数列表`groupList`。
相关输入配置代码如下，具体详见[grouped_matmul_slice_m_per_token_dequant.cpp](grouped_matmul_slice_m_per_token_dequant.cpp)。
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# cd [代码仓路径]/build/bin
# 可执行文件名|group数量|矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./07_grouped_matmul_slice_m_per_token_dequant 128 512 1024 2048 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```