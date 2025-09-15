# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os

class LaunchMapTemplate:
    TEMPLATE = """
#ifndef LAUNCH_MAP_H
#define LAUNCH_MAP_H

#include <unordered_map>
#include <string>

#include "tiling_params.h"

/*
 * Bit field layout description (little-endian):
 * -------------------------------------------------------------------------
 * | Bit Range | Size | Field Name            | Description                |
 * |-----------|------|-----------------------|----------------------------|
 * | 0-3       | 4    | layoutTagB            | Layout tag for B matrix    |
 * | 4-7       | 4    | layoutTagA            | Layout tag for A matrix    |
 * | 8-11      | 4    | paddingTagA           | Padding tag for A matrix   |
 * | 12-15     | 4    | paddingTagB           | Padding tag for B matrix   |
 * | 16-19     | 4    | paddingTagC           | Padding tag for C matrix   |
 * | 20-51     | 32   | reserveBit            | Reserved for future use    |
 * | 52-55     | 4    | dtype                 | Data type specification    |
 * | 56-63     | 8    | templateKernelSerial  | Template kernel serial ID  |
 * -------------------------------------------------------------------------
 */

union TilingKey {{
    uint64_t value;
    struct {{
        uint64_t layoutTagC : 4;  // 0-3
        uint64_t layoutTagB : 4;  // 4-7
        uint64_t layoutTagA : 4;  // 8-11
        uint64_t paddingTagC : 4; // 12-15
        uint64_t paddingTagB : 4; // 16-19
        uint64_t paddingTagA : 4; // 20-23
        uint64_t reserveBit : 28; // 24-51 May be used in the future
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

    @staticmethod
    def gen_code(kernel_info):
        declare_list = "\n".join(
            "DECLARE_KERNEL_FUNC({})".format(value) for value in kernel_info.values()
        )
        launch_func_list = ",\n".join(
            "{{ {}, Launch{} }}".format(key, value)
            for key, value in kernel_info.items()
        )
        workspace_func_list = ",\n".join(
            "{{ {}, {}GetWorkspaceSize }}".format(key, value)
            for key, value in kernel_info.items()
        )
        func_name_list = ",\n".join(
            '{{ {}, "{}" }}'.format(key, value) for key, value in kernel_info.items()
        )
        content = LaunchMapTemplate.TEMPLATE.format(
            declare_list=declare_list,
            launch_func_list=launch_func_list,
            workspace_func_list=workspace_func_list,
            func_name_list=func_name_list,
        )
        with open(os.path.join("../../include", "launch_map.h"), "w") as f:
            f.write(content)
