# Block Dispatch Policies说明
DispatchPolicy是BlockMmad的一个重要模板参数，各个DispatchPolicy定义在include/catlass/gemm/dispatch_policy中。本文档对下列四个DispatchPolicy的功能、参数以及使用的example进行简单介绍。
- MmadAtlasA2Pingpong
- MmadAtlasA2Preload
- MmadAtlasA2PreloadAsync
- MmadAtlasA2PreloadAsyncWithCallBack
## MmadAtlassA2PingPong
功能：在A2架构上采用L1和L0A/B Buffer上pingpong Buffer。

参数说明：
- `STAGES`：多buffer场景的buffer片数。
- `ENABLE_UINT_FLAG`：用于表示是否启用uintflag优化，启用Mmad运算与L0C结果拷贝到全局内存的细粒度并行。

示例代码：
```c++
struct MmadAtlasA2PingPong {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UINT_FLAG = true;
}
```


当前使用该DispatchPolicy的examples有`00_basic_matmul`、`01_batched_matmul`、`03_matmul_add`、`04_padding_matmul`、`09_split_matmul`。

## MmadAtlassA2Preload
功能：在A2架构上采用L1和L0A/B Buffer上pingpong Buffer，同时支持shufflek策略与block间的预加载。

参数说明：
- `STAGES`：多buffer场景的buffer片数。
- `ENABLE_UINT_FLAG`：用于表示是否启用uintflag优化，启用Mmad运算与L0C结果拷贝到全局内存的细粒度并行。
- `ENABLE_SHUFFLE_K`：用于表示是否启用shufflek策略。

示例代码：
```c++
struct MmadAtlasA2Preload {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UINT_FLAG = true;
    static constexpr bool ENABLE_SHUFFLE_K = true;
}
```


当前使用该DispatchPolicy的examples有`06_optimized_matmul`。

## MmadAtlassA2PreloadAsync
功能：在A2架构上采用L1 Buffer和L0A/L0B/L0C Buffer上的nBuffer，同时支持shufflek策略、block间的预加载以及group间的预加载。

参数说明：
- `PRELOAD_STAGES`：用于表示经过PRELOAD_STAGES次GM到L1的数据读取后，启动L1到L0的数据搬运和Mmad计算，取值要求小于L1_STAGES。
- `L1_STAGES`：用于表示L1开的buffer数量，需要满足L1TileShape的(M\*K\*矩阵A元素数据类型字节数+K\*N\*矩阵B元素数据类型字节数)<=L1大小。
- `L0A_STAGES`：用于表示L0A开的buffer数量，需要满足L0TileShape的M\*K\*矩阵A元素数据类型字节数<=L0A大小。
- `L0B_STAGES`：用于表示L0B开的buffer数量，需要满足L0TileShape的K\*N\*矩阵B元素数据类型字节数<=L0B大小。
- `L0C_STAGES`：用于表示L0C开的buffer数量，需要满足L0TileShape的M\*N\*Mmad计算数据类型字节数<=L0C大小。
- `ENABLE_UINT_FLAG`：用于表示是否启用uintflag优化，启用Mmad运算与L0C结果拷贝到全局内存的细粒度并行。
- `ENABLE_SHUFFLE_K`：用于表示是否启用shufflek策略。

示例代码：
```c++
struct MmadAtlasA2PreloadAsync {
    static constexpr uint32_t PRELOAD_STAGES = 1;
    static constexpr uint32_t L1_STAGES = 2;
    static constexpr uint32_t L0A_STAGES = 2;
    static constexpr uint32_t L0B_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = 1;
    static constexpr bool ENABLE_UINT_FLAG = false;
    static constexpr bool ENABLE_SHUFFLE_K = true;
}
```

当前使用该DispatchPolicy的examples有`02_grouped_matmul_slice_m`、`05_grouped_matmul_slice_k`、`11_grouped_matmul_slice_k_per_token_dequant`。

## MmadAtlassA2PreloadAsyncWithCallback
功能：在A2架构上采用L1 Buffer和L0A/L0B/L0C Buffer上的nBuffer，同时支持shufflek策略、block间的预加载以及group间的预加载。同时支持用户将aic和aiv之间的同步命令以callback的形式传入block层，由block层决定调用的时机。

参数说明：
- `PRELOAD_STAGES`：用于表示经过PRELOAD_STAGES次GM到L1的数据读取后，启动L1到L0的数据搬运和Mmad计算，取值要求小于L1_STAGES。
- `L1_STAGES`：用于表示L1开的buffer数量，需要满足L1TileShape的(M\*K\*矩阵A元素数据类型字节数+K\*N\*矩阵B元素数据类型字节数)<=L1大小。
- `L0A_STAGES`：用于表示L0A开的buffer数量，需要满足L0TileShape的M\*K\*矩阵A元素数据类型字节数<=L0A大小。
- `L0B_STAGES`：用于表示L0B开的buffer数量，需要满足L0TileShape的K\*N\*矩阵B元素数据类型字节数<=L0B大小。
- `L0C_STAGES`：用于表示L0C开的buffer数量，需要满足L0TileShape的M\*N\*Mmad计算数据类型字节数<=L0C大小。
- `ENABLE_UINT_FLAG`：用于表示是否启用uintflag优化，启用Mmad运算与L0C结果拷贝到全局内存的细粒度并行。
- `ENABLE_SHUFFLE_K`：用于表示是否启用shufflek策略。

示例代码：
```c++
struct MmadAtlasA2PreloadAsyncWithCallback {
    static constexpr uint32_t PRELOAD_STAGES = 1;
    static constexpr uint32_t L1_STAGES = 2;
    static constexpr uint32_t L0A_STAGES = 2;
    static constexpr uint32_t L0B_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = 1;
    static constexpr bool ENABLE_UINT_FLAG = false;
    static constexpr bool ENABLE_SHUFFLE_K = true;
}
```
当前使用该DispatchPolicy的examples有`10_grouped_matmul_slice_m_per_token_dequant`、`12_quant_matmul`。

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
[CANN Open Software License Agreement Version 1.0](../LICENSE)