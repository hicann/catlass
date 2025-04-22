# 打包为共享库

有时我们希望在已有的成熟工程中添加模板库算子，实现加速计算的效果，但又不希望大幅度改变构建工程，为此我们可以将模板库算子编译成共享库，以方便在已有工程中调用.

## 代码结构

```bash
examples/shared_lib
├── act_kernel.cpp      # host算子入口实现
├── act_kernel.h        # host使用的参数结构体、算子入口
├── build.sh                # 编译脚本
└── impl                    # 算子核函数实现
        ├── BasicMatmul.h  # 示例：basic_matmul
        └── ...
```

## 编译产物结构

```bash
output/shared_lib
├── act_kernel.h        # 动态链接库头文件
├── libact_kernel.a     # 静态链接库
└── libact_kernel.so    # 动态链接库
```

## 使用说明

假设待添加算子为`custom_matmul`

### 算子kernel实现

在`shared_lib/impl`文件夹中创建`CustomMatmul.h`，内容可参考`BasicMatmul.h`，大致如下：

```cpp
#include "act/act.hpp"
// act头文件...

using namespace Act;

template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
ACT_GLOBAL
void custom_matmul(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC
    // 按需定义输入参数...
)
{
    // 使用Act api定义算子...
}
```
- 你可以在模板参数中传入数据类型，但目前版本编译器暂不支持在内核调用符上使用bfloat16_t. 若需要通过模板特化bfloat16_t相关的核函数，可参考下面的示例：
```cpp
template<typename T>
ACT_DEVICE void real_kernel(...){
    //...
}
template<aclDataType T>
ACT_GLOBAL void kernel(...){
    if constexpr (T == ACL_BF16){
        real_kernel<bfloat16_t>(...);
    }
}
void kernel_host(...){
    kernel<ACL_BF16><<<blockNum, nullptr, stream>>>(...);
}
```
即：device侧的特化要在device侧实现.
### 算子host接口实现

参考`shared_lib/act_kernel.cpp`增加host接口.
推荐参数列表如下：

| 参数名           | 类型             | 作用               |
| ---------------- | ---------------- | ------------------ |
| `blockNum`       | `uint32_t`       | 设定aiCore个数     |
| `stream`         | `aclrtStream`    | NPU流              |
| `kernelInfo` | `KernelInfo` | 算子执行的数据地址和输入详细情况，如mnk等维度的大小 |

同时，更新`shared_lib/act_kernel.h`中的对外接口.

### 编译

```bash
bash scripts/build.sh shared_lib
```

## 注意事项
- 我们目前提供了三种典型算子作为示例：
  - `BasicMatmul`：基本矩阵乘法，并实现了类型模板的实现方法
  - `GroupedMatmul`：分组矩阵乘法，提供分组输入输出示例
  - `OptimizedMatmul`：优化矩阵乘法，提供CV融合的示例
- 本节是算子打包成动态库的一个示例，可根据需要自行扩展功能，并不仅局限于已有的代码.

## 版权声明
Copyright (c) 2025 Huawei Technologies Co., Ltd.

This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY, OR FITNESS FOR A PARTICULAR   PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.

## 许可证
[CANN Open Software License Agreement Version 1.0](../../LICENSE)