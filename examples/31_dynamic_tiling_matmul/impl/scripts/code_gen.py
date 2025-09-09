import os
import sys
import itertools

WRAPPER_CODE_PATH = "./warpper"

LAYOUT_TAG_SET = [0, 1] # 0 is RowMajor, 1 is ColumnMajor
LAYOUT_TAG_MAP = {0: "Catlass::layout::RowMajor", 1: "Catlass::layout::ColumnMajor"}

DTYPE_MAP = {"half": 0, "float": 1}

common_matmul_template = """
#include "kernel/common_matmul_kernel.h"
void {launch_kernel_func_name}(aclrtStream& stream, uint64_t fftsAddr,
    uint8_t* dA, uint8_t* dB, uint8_t* dC, uint8_t* dW, uint8_t* dTilingParams, TilingParams& tilingParams)
{{
    using ElementA = {elememt_a};
    using ElementB = {elememt_b};
    using ElementC = {elememt_c};
    using LayoutA = {layout_a};
    using LayoutB = {layout_b};
    using LayoutC = {layout_c};
    LaunchCommonMatmulKernel<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
        stream, fftsAddr, dA, dB, dC, dTilingParams, tilingParams);
}}

size_t {get_workspace_func_name}(TilingParams& tilingParams)
{{
    using ElementA = {elememt_a};
    using ElementB = {elememt_b};
    using ElementC = {elememt_c};
    using LayoutA = {layout_a};
    using LayoutB = {layout_b};
    using LayoutC = {layout_c};
    return CommonMatmulKernelGetWorkspaceSize<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(tilingParams);
}}
"""

def gen_common_matmul_code(kernel_name, base_file_name, kernel_serial, dtype):
    combinations = list(itertools.product(LAYOUT_TAG_SET, LAYOUT_TAG_SET))
    for l_tag_a, l_tag_b in combinations:
        kernel_func_name = (kernel_name + dtype.capitialize() + "Layout" + str(l_tag_a) + str(l_tag_b))
        file_name = (base_file_name + "_" + dtype + "_layout" + str(l_tag_a) + str(l_tag_b) + ".cpp")

        element_a = dtype
        element_b = dtype
        element_c = dtype
        layout_a = LAYOUT_TAG_MAP[l_tag_a]
        layout_b = LAYOUT_TAG_MAP[l_tag_b]
        layout_c = "Catlass::layout::RowMajor"

        template = common_matmul_template.format(element_a, element_b, element_c, layout_a, layout_b, layout_c)

        with open(os.path.join(WRAPPER_CODE_PATH, file_name), "w") as f:
            f.write(template)


if __name__ == "__main__":
    str_dtype = str(sys.argv[1])

    gen_common_matmul_code(
        "CommonMatmulKernel",
        "common_matmul_kernel",
        0,
        str_dtype
    )