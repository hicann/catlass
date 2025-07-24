# FAG_TND Example Readme
## 代码组织
```
├── 20_fag_tnd
│ ├── CMakeLists.txt # CMake编译文件
│ ├── README.md
│ ├── fag_tnd_kernel.cpp # MLAG模板所需的模块
│ ├── gen_data.py # 测试数据的生成脚本
│ └── fag_tnd.cpp # 主文件
│ └── fag_tnd_tiling.cpp
│ └── softmax_tiling.cpp
```

## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 第一步，首先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入。
python gen_data.py 8 1 128 32768
#输入参数分别对应 nheads，nheads_k，headdim，list_seq

执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据
```
├── data
│ ├── q.bin
│ ├── k.bin
│ ├── v.bin
│ ├── dout.bin
│ ├── atten_mask.bin
│ ├── row_max.bin
│ ├── row_sum.bin
│ ├── out.bin
│ ├── cu_seq_qlen.bin
│ ├── cu_seq_kvlen.bin
│ ├── dq_golden.bin
│ ├── dk_golden.bin
│ └── dv_golden.bin
```

第二步，执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。
cd [代码仓路径]/build/bin
./20_fag_tnd 8 1 128 32768 0

此处的参数和生成数据的参数保持一致
完整参数为 nheads nheads_k headdim seq_list [--device DEVICE_ID]，datapath默认为../../examples/20_fag_tnd/data, device默认为0。
执行结果如下，说明精度比对成功。
Compare dq success.
Compare dk success
Compare dv success