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

class PaddingMutiCoreSplitkMatmulTemplate:

    TEMPLATE = """
#include "kernel/padding_muti_core_splitk_matmul_kernel.h"
void {launch_kernel_func_name}(aclrtStream& stream, uint64_t fftsAddr,
    uint8_t* dA, uint8_t* dB, uint8_t* dC, uint8_t* dW, uint8_t* dTilingParams, TilingParams& tilingParams)
{{
    using ElementA = {element_a};
    using ElementB = {element_b};
    using ElementC = {element_c};
    using LayoutA = {layout_a};
    using LayoutB = {layout_b};
    using LayoutC = {layout_c};
    constexpr PaddingTag paddingTagA = {padding_tag_a};
    constexpr PaddingTag paddingTagB = {padding_tag_b};
    LaunchPaddingMutiCoreSplitkMatmulKernel<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
        paddingTagA, paddingTagB>(
        stream, fftsAddr, dA, dB, dC, dW, dTilingParams, tilingParams);
}}

size_t {get_workspace_func_name}(TilingParams& tilingParams)
{{
    using ElementA = {element_a};
    using ElementB = {element_b};
    using ElementC = {element_c};
    using LayoutA = {layout_a};
    using LayoutB = {layout_b};
    using LayoutC = {layout_c};
    constexpr PaddingTag paddingTagA = {padding_tag_a};
    constexpr PaddingTag paddingTagB = {padding_tag_b};
    return PaddingMutiCoreSplitkMatmulKernelGetWorkspaceSize<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
        paddingTagA, paddingTagB>(tilingParams);
}}
"""

    @staticmethod
    def gen_code(kernel_name, base_file_name, kernel_serial, dtype, kernel_info):

        PADDING_TAG_SET_A = [0, 3]
        PADDING_TAG_SET_B = [0, 3]
        combinations = list(
            itertools.product(
                Config.LAYOUT_TAG_SET, Config.LAYOUT_TAG_SET,
                PADDING_TAG_SET_A, PADDING_TAG_SET_B
            )
        )
        for l_tag_a, l_tag_b, p_tag_a, p_tag_b in combinations:
            # kernel_fun_name can be PaddingMutiCoreSplitkMatmulKernelHalfLayout00
            kernel_func_name = (
                kernel_name
                + dtype.capitalize()
                + "Layout"
                + str(l_tag_a)
                + str(l_tag_b)
                + "Padding"
                + str(p_tag_a)
                + str(p_tag_b)
            )
            # store tilingKey and kernel name
            kernel_info[
                Config.get_tiling_key(kernel_serial, dtype, l_tag_a, l_tag_b, 0, p_tag_a, p_tag_b, 0)
            ] = kernel_func_name
            # launch_kernel_fun_name can be LaunchPaddingMutiCoreSplitkMatmulKernelHalfLayout00
            launch_kernel_func_name = "Launch" + kernel_func_name
            # get_workspace_fun_name can be PaddingMutiCoreSplitkMatmulKernelHalfLayout00GetWorkspaceSize
            get_workspace_func_name = (
                kernel_name
                + dtype.capitalize()
                + "Layout"
                + str(l_tag_a)
                + str(l_tag_b)
                + "Padding"
                + str(p_tag_a)
                + str(p_tag_b)
                + "GetWorkspaceSize"
            )
            # file name can be padding_matmul_kernel_half_layout_00.cpp
            file_name = (
                base_file_name
                + "_"
                + dtype
                + "_layout"
                + str(l_tag_a)
                + str(l_tag_b)
                + "_padding"
                + str(p_tag_a)
                + str(p_tag_b)
                + ".cpp"
            )

            element_a = dtype
            element_b = dtype
            element_c = dtype
            layout_a = Config.LAYOUT_TAG_MAP[l_tag_a]
            layout_b = Config.LAYOUT_TAG_MAP[l_tag_b]
            layout_c = "Catlass::layout::RowMajor"
            padding_tag_a = Config.PADDING_TAG_MAP[p_tag_a]
            padding_tag_b = Config.PADDING_TAG_MAP[p_tag_b]

            content = PaddingMutiCoreSplitkMatmulTemplate.TEMPLATE.format(
                launch_kernel_func_name=launch_kernel_func_name,
                get_workspace_func_name=get_workspace_func_name,
                element_a=element_a,
                element_b=element_b,
                element_c=element_c,
                layout_a=layout_a,
                layout_b=layout_b,
                layout_c=layout_c,
                padding_tag_a=padding_tag_a,
                padding_tag_b=padding_tag_b,
            )

            with open(os.path.join(Config.WRAPPER_CODE_PATH, file_name), "w") as f:
                f.write(content)
