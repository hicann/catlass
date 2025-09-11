import os
import itertools

from utils.config import Config

class CommonMatmulTemplate:

    TEMPLATE = """
#include "kernel/common_matmul_kernel.h"
void {launch_kernel_func_name}(aclrtStream& stream, uint64_t fftsAddr,
    uint8_t* dA, uint8_t* dB, uint8_t* dC, uint8_t* dW, uint8_t* dTilingParams, TilingParams& tilingParams)
{{
    using ElementA = {element_a};
    using ElementB = {element_b};
    using ElementC = {element_c};
    using LayoutA = {layout_a};
    using LayoutB = {layout_b};
    using LayoutC = {layout_c};
    LaunchCommonMatmulKernel<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
        stream, fftsAddr, dA, dB, dC, dTilingParams, tilingParams);
}}

size_t {get_workspace_func_name}(TilingParams& tilingParams)
{{
    using ElementA = {element_a};
    using ElementB = {element_b};
    using ElementC = {element_c};
    using LayoutA = {layout_a};
    using LayoutB = {layout_b};
    using LayoutC = {layout_c};
    return CommonMatmulKernelGetWorkspaceSize<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(tilingParams);
}}
"""
    @staticmethod
    def gen_code(self, kernel_name, base_file_name, kernel_serial, dtype, kernel_info):
        combinations = list(itertools.product(Config.LAYOUT_TAG_SET, Config.LAYOUT_TAG_SET))
        for l_tag_a, l_tag_b in combinations:
            kernel_func_name = (
                kernel_name
                + dtype.capitalize()
                + "Layout"
                + str(l_tag_a)
                + str(l_tag_b)
            )
            kernel_info[Config.get_tiling_key(kernel_serial, dtype, l_tag_a, l_tag_b, 0, 0, 0)] = kernel_func_name
            launch_kernel_func_name = "Launch" + kernel_func_name
            get_workspace_func_name = (
                kernel_name
                + dtype.capitalize()
                + "Layout"
                + str(l_tag_a)
                + str(l_tag_b)
                + "GetWorkspaceSize"
            )
            file_name = (
                base_file_name
                + "_"
                + dtype
                + "_layout"
                + str(l_tag_a)
                + str(l_tag_b)
                + ".cpp"
            )

            element_a = dtype
            element_b = dtype
            element_c = dtype
            layout_a = Config.LAYOUT_TAG_MAP[l_tag_a]
            layout_b = Config.LAYOUT_TAG_MAP[l_tag_b]
            layout_c = "Catlass::layout::RowMajor"

            content = self.TEMPLATE.format(
                launch_kernel_func_name=launch_kernel_func_name,
                get_workspace_func_name=get_workspace_func_name,
                element_a=element_a,
                element_b=element_b,
                element_c=element_c,
                layout_a=layout_a,
                layout_b=layout_b,
                layout_c=layout_c,
            )

            with open(os.path.join(Config.WRAPPER_CODE_PATH, file_name), "w") as f:
                f.write(content)
