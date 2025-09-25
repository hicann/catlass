# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import itertools

from utils.config import Config

class AivMatmulTemplate:

    TEMPLATE = """
#include "kernel/aiv_matmul_kernel.h"
void {launch_kernel_func_name}(aclrtStream& stream, uint64_t fftsAddr,
    uint8_t* dA, uint8_t* dB, uint8_t* dC, uint8_t* dW, uint8_t* dTilingParams, TilingParams& tilingParams)
{{
    using ElementA = {element_a};
    using ElementB = {element_b};
    using ElementC = {element_c};
    DispatchPolicyTag dispatchPolicyTag = {dispatch_policy_tag}
    LaunchAivMatmulKernel<ElementA, ElementB, ElementC, dispatchPolicyTag>(
        stream, fftsAddr, dA, dB, dC, dTilingParams, tilingParams);
}}

size_t {get_workspace_func_name}(TilingParams& tilingParams)
{{
    using ElementA = {element_a};
    using ElementB = {element_b};
    return AivMatmulKernelGetWorkspaceSize<ElementA, ElementB, ElementC, dispatchPolicyTag>(tilingParams);
}}
"""
    DISPATCH_POLICY_TAG_MAP = {
        0: "DispatchPolicyTag::DEFAULT",
        1: "DispatchPolicyTag::MATMUL_AIV_SIMPLE",
        2: "DispatchPolicyTag::MATMUL_AIV_TRANS"
    }

    @staticmethod
    def gen_code(kernel_name, base_file_name, kernel_serial, dtype, kernel_info):
        DISPATCH_POLICY_SET = [0, 1, 2]
        combinations = list(
            itertools.product(DISPATCH_POLICY_SET)
        )
        for d_tag in combinations:
            # kernel_fun_name can be CommonMatmulKernelHalfLayout00
            kernel_func_name = (
                kernel_name
                + dtype.capitalize()
                + "Policy"
                + str(d_tag)
            )
            # store tilingKey and kernel name
            kernel_info[
                Config.get_tiling_key(kernel_serial, dtype, 0, 0, 0, 0, 0, 0)
            ] = kernel_func_name
            # launch_kernel_fun_name can be LaunchCommonMatmulKernelHalfLayout00
            launch_kernel_func_name = "Launch" + kernel_func_name
            # get_workspace_fun_name can be CommonMatmulKernelHalfLayout00GetWorkspaceSize
            get_workspace_func_name = kernel_func_name + "GetWorkspaceSize"
            # file name can be common_matmul_kernel_half_layout_00.cpp
            file_name = (
                base_file_name
                + "_"
                + dtype
                + "_policy"
                + str(d_tag)
                + ".cpp"
            )

            element_a = dtype
            element_b = dtype
            element_c = dtype

            content = AivMatmulTemplate.TEMPLATE.format(
                launch_kernel_func_name=launch_kernel_func_name,
                get_workspace_func_name=get_workspace_func_name,
                element_a=element_a,
                element_b=element_b,
                element_c=element_c,
            )

            with open(os.path.join(Config.WRAPPER_CODE_PATH, file_name), "w") as f:
                f.write(content)
