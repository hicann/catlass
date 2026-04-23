# Torch Catlass

基于 Catlass 的 PyTorch NPU 算子库，支持多架构动态加载。

## 项目结构

```
optest/
├── CMakeLists.txt              # 主构建文件
├── build.sh                    # 构建脚本
├── todo.md                     # 待办事项和疑问
├── README.md                   # 本文档
├── pyproject.toml              # Python 项目配置
│
├── kernels/                    # Kernel 源码目录（NPU 实现）
│   ├── CMakeLists.txt
│   ├── README.md
│   └── 00_basic_matmul/        # 基础矩阵乘法算子
│       ├── CMakeLists.txt
│       └── basic_matmul.cpp
│
├── src/                        # C++ 封装层（Torch 接口）
│   ├── CMakeLists.txt
│   ├── catlass_torch.cpp       # Torch 绑定入口
│   ├── common/                 # 通用工具
│   │   └── workspace.cpp
│   └── include/
│       └── common/
│           ├── register.h      # Torch 算子注册
│           ├── workspace.h
│           └── run_npu_func.h
│
├── torch_catlass/              # Python 包
│   ├── __init__.py             # 包入口，库加载逻辑
│   ├── lib/                    # 编译后的库文件
│   │   ├── libcatlass_torch.so # 主库
│   │   ├── 2201/               # 2201 架构的 kernel 库
│   │   │   ├── libcatlass_kernel_2201.so
│   │   │   ├── libcatlass_kernel_2201_slice_0.so
│   │   │   └── ...
│   │   └── 3510/               # 3510 架构的 kernel 库
│   │       ├── libcatlass_kernel_3510.so
│   │       └── ...
│   └── ops/                    # 算子 Python 接口
│       ├── __init__.py
│       └── basic_matmul.py
│
└── tests/                      # 测试文件
    └── test_basic_matmul.py
```

## 架构设计

### 1. 多架构支持

项目支持多种 NPU 架构（2201, 3510），通过动态检测当前设备自动加载对应的库：

- **架构检测**: `get_npu_arch()` 函数根据设备名称判断架构
- **库加载**: 根据架构加载对应的 kernel 库
- **切片管理**: 每个 kernel 库按 `CATLASS_KERNEL_SLICE_COUNT` 切分，控制单个库大小

### 2. Torch 算子封装

使用 `torch.ops` 机制注册算子：

```cpp
// C++ 层（src/）
#define REGISTER_TORCH_FUNC(opFunc) \
    catlass_torch::_register::RegisterFunc<decltype(&opFunc)> register_##opFunc(#opFunc, &opFunc)
```

```python
# Python 层
def basic_matmul(a, b, out=None, use_nz_b=False):
    return torch.ops.catlass.basic_matmul(a, b, out, use_nz_b)
```

### 3. 构建系统

使用 CMake 构建，支持：

- **选择性编译算子**: 通过 `-DCATLASS_KERNEL_LIST` 指定编译哪些算子
- **选择性编译架构**: 通过 `-DCATLASS_ARCH_LIST` 指定编译哪些架构
- **库切分**: 自动将大库切分为多个小库

### 4. 代码分层

- **src/**: Torch 接口层，提供 Python 绑定
- **kernels/**: NPU 实现层，包含具体的 kernel 代码

## 快速开始

### 1. 环境要求

- Python >= 3.11
- PyTorch >= 2.0
- torch-npu >= 2.9.0
- CMake >= 3.16
- ASCEND 开发环境

### 2. 编译

```bash
# 设置环境变量
export ASCEND_HOME_PATH=/path/to/ascend

# 编译所有架构和算子
bash build.sh

# 只编译特定算子
CATLASS_KERNEL_LIST="basic_matmul" bash build.sh

# 只编译特定架构
CATLASS_ARCH_LIST="2201" bash build.sh

# 组合使用
CATLASS_KERNEL_LIST="basic_matmul" CATLASS_ARCH_LIST="2201;3510" bash build.sh
```

### 3. 使用

```python
import torch
import torch_npu
import torch_catlass

# 自动检测架构并加载库
a = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
b = torch.randn(1024, 1024, dtype=torch.float16, device="npu")

# 调用算子
result = torch_catlass.ops.basic_matmul(a, b)
```

### 4. 测试

```bash
# 安装测试依赖
pip install pytest

# 运行测试
pytest tests/ -v
```

## 开发指南

### 添加新算子

1. **创建 kernel 目录**:
   ```bash
   mkdir kernels/01_new_op
   ```

2. **实现 kernel**:
   ```cpp
   // kernels/01_new_op/new_op.cpp
   __global__ __aicore__ void new_op_kernel(...) {
       // kernel 实现
   }
   ```

3. **注册到 CMakeLists.txt**:
   ```cmake
   # kernels/01_new_op/CMakeLists.txt
   add_kernel(new_op "2201;3510" new_op.cpp)
   ```

4. **添加 Torch 封装**:
   ```cpp
   // src/kernels/new_op.cpp
   at::Tensor new_op(const at::Tensor& input) {
       // 调用 kernel
   }
   REGISTER_TORCH_FUNC(new_op)
   ```

5. **添加 Python 接口**:
   ```python
   # torch_catlass/ops/new_op.py
   def new_op(input):
       return torch.ops.catlass.new_op(input)
   ```

### 选择性编译

只编译特定算子和架构：

```bash
# 只编译 basic_matmul 算子的 2201 架构版本
export CATLASS_KERNEL_LIST="basic_matmul"
export CATLASS_ARCH_LIST="2201"
bash build.sh
```

## 错误处理

- **不支持的架构**: 直接抛出 RuntimeError 并结束进程
- **库加载失败**: 抛出详细错误信息
- **Kernel 执行失败**: 返回错误码

## 依赖管理

- **catlass**: 使用项目根目录的 catlass 库（相对路径）
- **fmt**: 不使用，使用 printf 替代
- **torch/torch_npu**: 通过 pip 安装

## 待办事项

详见 [todo.md](todo.md)，包括：

- [x] 库命名和组织优化
- [x] 多架构支持
- [x] 选择性编译架构
- [ ] 更多算子支持
- [ ] 完善错误处理

## 参考项目

- [torch_catlass](../../catlass-test/torch_catlass): 多平台选择性加载方式
- [ascend_ops](../../ascend_ops): torch.ops 封装方式

## 许可证

Copyright (c) 2026 Huawei Technologies Co., Ltd.
