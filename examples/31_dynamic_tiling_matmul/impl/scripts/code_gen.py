import os
import sys
import itertools

WRAPPER_CODE_PATH = "../wrapper"

LAYOUT_TAG_SET = [0, 1]  # 0 is RowMajor, 1 is ColumnMajor
LAYOUT_TAG_MAP = {0: "Catlass::layout::RowMajor", 1: "Catlass::layout::ColumnMajor"}

DTYPE_MAP = {"half": 0, "float": 1}

common_matmul_template = """
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

launch_map_template = """
#ifndef LAUNCH_MAP_H
#define LAUNCH_MAP_H

#include <unordered_map>
#include <string>

#include "tiling_params.h"

union TilingKey {{
    uint64_t value;
    struct {{
        uint64_t layoutTagB : 4;  // 0-3
        uint64_t layoutTagA : 4;  // 4-7
        uint64_t paddingTagA : 4; // 8-11
        uint64_t paddingTagB : 4; // 12-15
        uint64_t paddingTagC : 4; // 16-19
        uint64_t reserveBit : 32; // 20-51 May be used in the future
        uint64_t dtype : 4;       // 52-55
        uint64_t templateKernelSerial : 8; // 56-63
    }} bits;

    TilingKey() : value(0) {{}}
    TilingKey(uint64_t v) : value(v) {{}}

    uint8_t GetLayoutTagA() const {{return bits.layoutTagA;}}
    uint8_t GetLayoutTagB() const {{return bits.layoutTagB;}}
    uint8_t GetPaddingTagTagA() const {{return bits.paddingTagA;}}
    uint8_t GetPaddingTagTagB() const {{return bits.paddingTagB;}}
    uint8_t GetPaddingTagTagC() const {{return bits.paddingTagC;}}
    uint8_t GetDtype() const {{return bits.dtype;}}
    uint8_t GetKernelSerial() const {{return bits.templateKernelSerial;}}

    void SetKernelSerial(uint8_t kernelSerial) {{ bits.templateKernelSerial = kernelSerial;}}
    void SetLayoutTagA(uint8_t layoutTagA) {{ bits.layoutTagA = layoutTagA & 0xF; }}
    void SetLayoutTagB(uint8_t layoutTagB) {{ bits.layoutTagB = layoutTagB & 0xF; }}
    void SetPaddingTagA(uint8_t paddingTagA) {{ bits.paddingTagA = paddingTagA & 0xF; }}
    void SetPaddingTagB(uint8_t paddingTagB) {{ bits.paddingTagB = paddingTagB & 0xF; }}
    void SetPaddingTagC(uint8_t paddingTagC) {{ bits.paddingTagC = paddingTagC & 0xF; }}
    void SetDtype(uint8_t dtype) {{ bits.dtype = dtype & 0xF; }}

    void SetTilingKey(uint8_t kernelSerial, uint8_t layoutTagA, uint8_t layoutTagB,
        uint8_t paddingTagA, uint8_t paddingTagB, uint8_t paddingTagC, uint8_t dtype = 0)
    {{
        SetKernelSerial(kernelSerial);
        SetLayoutTagA(layoutTagA);
        SetLayoutTagB(layoutTagB);
        SetPaddingTagA(paddingTagA);
        SetPaddingTagB(paddingTagB);
        SetPaddingTagC(paddingTagC);
        SetDtype(dtype);
    }}
}};

#define DECLARE_KERNEL_FUNC(kernelName) \\
    void Launch##kernelName(aclrtStream&, uint64_t, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, TilingParams&); \\
    size_t kernelName##GetWorkspaceSize(TilingParams&);

{declare_list}

std::unordered_map<uint64_t, void(*)(aclrtStream&, uint64_t, 
    uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, TilingParams&)> launchKernelFuncMap = {{ 
{launch_func_list}
}};

using GetWorkspaceFunc = size_t(*)(TilingParams& tilingParams);
std::unordered_map<uint64_t, GetWorkspaceFunc> getWorkspaceFuncMap = {{
{workspace_func_list}
}};

// only for print kernel Info
std::unordered_map<uint64_t, std::string> funcNameMap = {{
{func_name_list}
}};


#endif // LAUNCH_MAP_H
"""

def get_tiling_key(kernel_serial, dtype, l_tag_a, l_tag_b, p_tag_a, p_tag_b, p_tag_c):
    part1 = kernel_serial # 56-63
    part2 = DTYPE_MAP[dtype] << 4 # 48-55
    part3 = 0 # 40-47
    part4 = 0 # 32-39
    part5 = 0 # 24-31
    part6 = 0 | p_tag_c # 16-23
    part7 = (p_tag_a << 4) | p_tag_b # 8-15
    part8 = (l_tag_a << 4) | l_tag_b # 0-7
    hex_str = f"0x{part1:02x}{part2:02x}{part3:02x}{part4:02x}{part5:02x}{part6:02x}{part7:02x}{part8:02x}"
    return hex_str


def gen_common_matmul_code(kernel_name, base_file_name, kernel_serial, dtype, kernel_info):
    combinations = list(itertools.product(LAYOUT_TAG_SET, LAYOUT_TAG_SET))
    for l_tag_a, l_tag_b in combinations:
        kernel_func_name = (
            kernel_name
            + dtype.capitalize()
            + "Layout"
            + str(l_tag_a)
            + str(l_tag_b)
        )
        kernel_info[get_tiling_key(kernel_serial, dtype, l_tag_a, l_tag_b, 0, 0, 0)] = kernel_func_name
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
        layout_a = LAYOUT_TAG_MAP[l_tag_a]
        layout_b = LAYOUT_TAG_MAP[l_tag_b]
        layout_c = "Catlass::layout::RowMajor"

        template = common_matmul_template.format(
            launch_kernel_func_name=launch_kernel_func_name,
            get_workspace_func_name=get_workspace_func_name,
            element_a=element_a,
            element_b=element_b,
            element_c=element_c,
            layout_a=layout_a,
            layout_b=layout_b,
            layout_c=layout_c,
        )

        with open(os.path.join(WRAPPER_CODE_PATH, file_name), "w") as f:
            f.write(template)

def gen_launch_map_code(kernel_info):
    declare_list = "\n".join("DECLARE_KERNEL_FUNC({})".format(value) for value in kernel_info.values())
    launch_func_list = ",\n".join("{{ {}, Launch{} }}".format(key, value) for key, value in kernel_info.items())
    workspace_func_list = ",\n".join("{{ {}, {}GetWorkspaceSize }}".format(key, value) for key, value in kernel_info.items())
    func_name_list = ",\n".join("{{ {}, \"{}\" }}".format(key, value) for key, value in kernel_info.items())
    template = launch_map_template.format(declare_list=declare_list, launch_func_list=launch_func_list, workspace_func_list=workspace_func_list, func_name_list=func_name_list)
    with open(os.path.join("../../include", "launch_map.h"), "w") as f:
            f.write(template)


if __name__ == "__main__":

    kernel_info = {}
    os.makedirs(WRAPPER_CODE_PATH, exist_ok=True)
    gen_common_matmul_code("CommonMatmulKernel", "common_matmul_kernel", 0, "half", kernel_info)
    gen_common_matmul_code("CommonMatmulKernel", "common_matmul_kernel", 0, "float", kernel_info)
    gen_launch_map_code(kernel_info)