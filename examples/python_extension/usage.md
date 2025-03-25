# python扩展

为方便开发者使用Ascend C Template算子，代码仓基于pybind11和torch提供了使用python调用Ascend C Template算子的示例.

- 注意：建议使用pybind扩展. 目前纯torch扩展不支持NPU，在使用torch扩展时，实际上会把NPU Tensor转换到CPU Tensor，再把CPU Tensor转换回NPU Tensor，因此性能会有较大影响.

## 代码结构

```bash
python_extension
├── CMakeLists.txt              # CMake构建脚本
├── AscendCTKernelWrapper.cpp  # 将at::Tensor类型的输入转换为kernel输入
├── AscendCTKernelWrapper.h    # 头文件
├── PybindBindings.cpp         # pybind绑定
├── set_env.sh                  # 环境变量设定
└── TorchBindings.cpp          # torch绑定
```

## 编译产物结构

```bash
output/python_extension
├── libascendct_torch.so                             # torch动态链接库
└── torch_ascendct.cpython-3xx-aarch64-linux-gnu.so  # pybind11动态链接库
```

## 使用说明

- 假设你已经在shared_lib中，增加了所需算子的实现和入口.

### pybind接口实现

由于pybind传入参数为at::Tensor而非AscendC中的GM地址指针，所以需要对python侧传来的数据进行处理.
主要步骤为根据输入tensor的信息填充运行信息参数，申请输出内存.
此部分较为灵活，与算子本身参数较为相关，可参考已有的BasicMatmul实现.

### 编译

各部分代码完成后，使用`bash scripts/build.sh python_extension`编译.

编译环境如下：

| 依赖              | 版本               | 安装指令                      |
| ----------------- | ------------------ | ----------------------------- |
| `build-essential` | gcc版本为9以上     | `apt install build-essential` |
| `cmake`           | `>=3.16`           | `apt install cmake`           |
| `CANN`            | `>=8.0.0.alpha002` | run包安装                     |
| `pybind11`        | 无要求             | `pip install pybind11`        |
| `torch`           | 无要求             | `pip install torch`           |
| `torch_npu`       | 无要求             | `pip install torch_npu`       |

- 虽然`torch`的版本没有要求，可沿用现有环境，但是为保证能够调用`torch_npu`的接口，`torch_npu`需要安装大版本下的**最新post**版本。你可以根据`CANN`版本和`torch`版本，在[Ascend/pytorch](https://gitee.com/ascend/pytorch)查询适合的`torch_npu`版本。

### 设置环境变量

- 多数情况下，你需要在运行前使用`output/python_extension/set_env.sh`补全系统的链接路径变量，这样程序才能找到torch的cpp库.

### 运行

```python
import sys
sys.path.append("../../output/python_extension") # 确保编译出的pybind so文件在path内
import torch_ascendct
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

class AscendCTTest(TestCase):
    def test_basic_matmul(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        result = torch_ascendct.basic_matmul(a, b, "float16")
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)
    def test_basic_matmul_torch_lib(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        torch.ops.load_library("../../output/python_extension/libascendct_torch.so")
        result = torch.ops.AscendCTTorch.basic_matmul(a, b, "float16")
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)
        
if __name__ == "__main__":
    run_tests()
```
