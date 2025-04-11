# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import numpy as np
import mskpp
from helper.act_type import GemmCoord, RowMajor, ColumnMajor
from helper.helper import get_ascendc_sync_base_addr, AIC_CORE_NUM_ATLAS_A2_910B4

device_id = 1

def get_kernel():
    kernel_file = "../06_optimized_matmul/optimized_matmul.cpp"
    kernel_name = "OptimizedMatmul"
    build_script = "./helper/jit_build.sh" # kernel compile script

    config = mskpp.KernelInvokeConfig(kernel_file, kernel_name)
    gen_file = mskpp.Launcher(config).code_gen()
    kernel = mskpp.compile(build_script=build_script, launch_src_file=gen_file)
    return kernel


"""
To enable the autotune feature, it is required to adjust line breaks and add the "// tunable: alias" marker to
the code lines in "optimized_matmul.cpp". The marked line will be entirely replaced, e.g.
    ...
    95    using L1TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
    96        std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 256>,
    97        GemmShape<128, 256, 256> // tunable: L1TileShape
    98        >;
    99    using L0TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
    100       std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 64>,
    101       GemmShape<128, 256, 64> // tunable: L0TileShape
    102       >;
"""
@mskpp.autotune(configs=[
    {'L1TileShape': 'GemmShape<128, 256, 256>', 'L0TileShape': 'GemmShape<128, 256, 64>'}, #0 the same config as in optimized_matmul.cpp
    {'L1TileShape': 'GemmShape<256, 128, 256>', 'L0TileShape': 'GemmShape<256, 128, 64>'},
    {'L1TileShape': 'GemmShape<128, 128, 256>', 'L0TileShape': 'GemmShape<128, 128, 64>'},
    {'L1TileShape': 'GemmShape<128, 128, 512>', 'L0TileShape': 'GemmShape<128, 128, 64>'},
    {'L1TileShape': 'GemmShape<64, 256, 128>', 'L0TileShape': 'GemmShape<64, 256, 64>'},
    {'L1TileShape': 'GemmShape<64, 256, 256>', 'L0TileShape': 'GemmShape<64, 256, 64>'},
    {'L1TileShape': 'GemmShape<64, 128, 256>', 'L0TileShape': 'GemmShape<64, 128, 64>'},
    {'L1TileShape': 'GemmShape<128, 128, 256>', 'L0TileShape': 'GemmShape<128, 128, 128>'},
    {'L1TileShape': 'GemmShape<128, 128, 512>', 'L0TileShape': 'GemmShape<128, 128, 128>'},
    {'L1TileShape': 'GemmShape<64, 128, 256>', 'L0TileShape': 'GemmShape<64, 128, 128>'},
    {'L1TileShape': 'GemmShape<64, 128, 512>', 'L0TileShape': 'GemmShape<64, 128, 128>'},
    {'L1TileShape': 'GemmShape<128, 64, 512>', 'L0TileShape': 'GemmShape<128, 64, 128>'},
    {'L1TileShape': 'GemmShape<64, 64, 256>', 'L0TileShape': 'GemmShape<64, 64, 256>'},
    {'L1TileShape': 'GemmShape<64, 64, 512>', 'L0TileShape': 'GemmShape<64, 64, 256>'},
    {'L1TileShape': 'GemmShape<64, 64, 1024>', 'L0TileShape': 'GemmShape<64, 64, 256>'},
], warmup=10000, repeat=5, device_ids=[device_id]) # set kernel warmup 1000us, avg of repeat 10 times
def optimized_matmul(ffts_addr, problem_shape, a, layout_a, b, layout_b, c, layout_c,
        workspace_a, workspace_b):
    # This function's input arguments must exactly match the kernel function.
    kernel = get_kernel()
    blockdim = AIC_CORE_NUM_ATLAS_A2_910B4 # choose the aic number that matches your hardware in helper/helper.py
    return kernel[blockdim](ffts_addr, problem_shape, a, layout_a, b, layout_b, c, layout_c,
        workspace_a, workspace_b) # invoke the kernel


if __name__ == "__main__":

    def round_up(val, align):
        return (val + align - 1) // align * align

    m = 256
    n = 512
    k = 1024

    # 创建kernel入参
    problem_shape = GemmCoord(m, n, k)
    layout_a = RowMajor(m, k)
    layout_b = ColumnMajor(k, n)
    layout_c = RowMajor(m, n)

    align = 256
    is_need_padding_a = True if layout_a.stride[0] < 65536 else (layout_a.stride[0] % align) != 0
    is_need_padding_b = True if layout_b.stride[1] < 65536 else (layout_b.stride[1] % align) != 0

    # assume m, n, k of L1TileShape are not larger than 1024
    sizeWA = round_up(layout_a.shape[0], 1024) * round_up(layout_a.shape[1], 1024)
    sizeWB = round_up(layout_b.shape[0], 1024) * round_up(layout_b.shape[1], 1024)

    a = np.random.randint(-5, 5, [m, k]).astype(np.half)
    b = np.random.randint(-5, 5, [k, n]).astype(np.half)
    c = np.zeros([m * n]).astype(np.half)
    
    workspace_a = np.random.randint(-5, 5, [sizeWA]).astype(np.half) if is_need_padding_a else a
    workspace_b = np.random.randint(-5, 5, [sizeWB]).astype(np.half) if is_need_padding_b else b
    ffts_addr, _ = get_ascendc_sync_base_addr(device_id)

    optimized_matmul(ffts_addr, problem_shape, a, layout_a, b, layout_b, c, layout_c,
        workspace_a, workspace_b)

    def data_compare(a, b):
        rtol = 1.0 / 256
        bool_matrix = np.abs(a - b) < rtol
        result = "success" if bool_matrix.all() else "failed"
        print("compare {}.".format(result))


    gen_golden = np.matmul(a, b.reshape(n, k).T).reshape([m * n])
    data_compare(gen_golden, c)