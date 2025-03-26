# 推荐使用配置
当前代码仓提供了6个examples样例，展示了如何基于算子模板库快速完成自定义算子开发。如果使用当前样例进行相关场景性能测试，推荐使用如下配置。
## 算子推荐使用输入范围
<table>
    <tr>
        <th>算子名称</th>
        <th>推荐输入范围</th>
    </tr>
    <tr>
        <td>basic_matmul</td>
        <td>A、B矩阵列方向<sup id="fn">[1]</sup>512字节对齐</td>
    </tr>
    <tr>
        <td>batched_matmul</td>
        <td>A、B矩阵列方向512字节对齐</td>
    </tr>
    <tr>
        <td rowspan="2">grouped_matmul</td>
        <td>专家数expertNum = 8；k = 2048 n = 2816 或 k = 1408 n = 2048；m = [20000, 28000]</td>
    </tr>
    <tr>
        <td>专家数expertNum = 20；k = 5120 n = 3072 或 k = 1536 n = 5120；m = [68000, 72000]</td>
    </tr>
    <tr>
        <td>matmul_add</td>
        <td>A、B矩阵列方向512字节对齐</td>
    </tr>
    <tr>
        <td rowspan="2">padding_matmul</td>
        <td>A、B矩阵列方向非512字节对齐</td>
    </tr>
    <tr>
        <td>2000 < m, n, k < 65536</td>
    </tr>
    <tr>
        <td>fa</td>
        <td>batch <= 64; headnum = 8, 32, 128; headdim = 72, 80, 128; seq_len = [256,16384] </td>
    </tr>
    <tr>
        <td>grouped_matmul_per_token_dequant</td>
        <td>GroupNum = 160; m=280; n = 4096 k = 512 或 n = 512 k = 4096</td>
    </tr>
</table>



注[1]：列方向定义。A矩阵transA=False时，列方向为k轴；transA=True时，列方向为m轴。B矩阵transB=False时，列方向为n轴；transB=True时，列方向为k轴。

## 其他配置
Matmul
- BLOCK_NUM的值设置为与AI_CORE的数目相等<sup>[2]</sup>。
- Swizzle配置
m>n时，使用GemmIdentityBlockSwizzle<3,0>。
m<=n时，使用GemmIdentityBlockSwizzle<3,1>。
Swizzle详细使用方法，请参考[Swizzle策略说明](swizzle_explanation.md)。

FA
- `faInfo.isTriuMask`设置为1时，开启下三角优化。当前默认开启，可参考[fa_kernel.cpp](../examples/05_fa/fa_kernel.cpp)中isTriuMask的配置。
- 当前fa样例暂不支持moe的相关配置。

注[2]：例如AI_CORE数目为20，则代码中BLOCK_NUM设置为20。AI_CORE数量可在`${CANN_INSTALL_PATH}/${arch}/data/platform_config/<chip_type>.ini`文件中的`ai_core_cnt`字段找到。`CANN_INSTALL_PATH`为cann包安装路径；`arch`为服务器cpu架构；`chip_type`为npu芯片型号，可通过执行`npu-smi info`查询。
## 版权声明
Copyright (c) 2024 Huawei Technologies Co., Ltd.

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