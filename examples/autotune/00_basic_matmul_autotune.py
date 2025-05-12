# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import numpy as np
import mskpp
from helper.act_type import GemmCoord, RowMajor
from helper.helper import AIC_CORE_NUM_ATLAS_A2_910B4, check_autotune_avalible, assert_kernel_run_one_time


check_autotune_avalible()


def get_kernel():
    kernel_file = "../00_basic_matmul/basic_matmul.cpp"
    kernel_name = "BasicMatmul"
    build_script = "./helper/jit_build.sh" # kernel compile script

    config = mskpp.KernelInvokeConfig(kernel_file, kernel_name)
    gen_file = mskpp.Launcher(config).code_gen()
    kernel = mskpp.compile(build_script=build_script, launch_src_file=gen_file)
    return kernel


"""
To enable the autotune feature, it is required to add the "// tunable" marker to
the code lines in "basic_matmul.cpp", e.g.
    ...
    44    using L1TileShape = GemmShape<128, 256, 256>; // tunable
    45    using L0TileShape = GemmShape<128, 256, 64>; // tunable
"""
@mskpp.autotune(configs=[
    {'L1TileShape': 'GemmShape<128, 256, 256>', 'L0TileShape': 'GemmShape<128, 256, 64>'}, #0 the same config as in basic_matmul.cpp
    {'L1TileShape': 'GemmShape<128, 256, 128>', 'L0TileShape': 'GemmShape<128, 256, 64>'},
    {'L1TileShape': 'GemmShape<128, 128, 256>', 'L0TileShape': 'GemmShape<128, 128, 64>'},
    {'L1TileShape': 'GemmShape<64, 128, 128>', 'L0TileShape': 'GemmShape<64, 128, 128>'},
    {'L1TileShape': 'GemmShape<64, 128, 256>', 'L0TileShape': 'GemmShape<64, 128, 128>'},
    {'L1TileShape': 'GemmShape<64, 128, 512>', 'L0TileShape': 'GemmShape<64, 128, 128>'},
    {'L1TileShape': 'GemmShape<64, 64, 128>', 'L0TileShape': 'GemmShape<64, 64, 128>'},
    {'L1TileShape': 'GemmShape<64, 64, 256>', 'L0TileShape': 'GemmShape<64, 64, 128>'},
    {'L1TileShape': 'GemmShape<64, 64, 512>', 'L0TileShape': 'GemmShape<64, 64, 128>'},
    {'L1TileShape': 'GemmShape<128, 128, 128>', 'L0TileShape': 'GemmShape<128, 128, 128>'},
    {'L1TileShape': 'GemmShape<128, 128, 256>', 'L0TileShape': 'GemmShape<128, 128, 128>'},
    {'L1TileShape': 'GemmShape<128, 128, 512>', 'L0TileShape': 'GemmShape<128, 128, 128>'},
], warmup=1000, repeat=10, device_ids=[0])
def basic_matmul(problem_shape, a, layout_a, b, layout_b, c, layout_c):
    # This function's input arguments must exactly match the kernel function.
    kernel = get_kernel()
    blockdim = AIC_CORE_NUM_ATLAS_A2_910B4 # choose the aic number that matches your hardware in helper/helper.py
    assert_kernel_run_one_time()
    return kernel[blockdim](problem_shape, a, layout_a, b, layout_b, c, layout_c) # invoke the kernel


if __name__ == "__main__":

    m = 256
    n = 512
    k = 1024

    problem_shape = GemmCoord(m, n, k)
    layout_a = RowMajor(m, k)
    layout_b = RowMajor(k, n)
    layout_c = RowMajor(m, n)

    a = np.random.randint(1, 2, [m, k]).astype(np.half)
    b = np.random.randint(1, 2, [k, n]).astype(np.half)
    c = np.zeros([m, n]).astype(np.half)

    basic_matmul(problem_shape, a, layout_a, b, layout_b, c, layout_c)

    # check if the output tensor c is consistent with the golden data
    golden = np.matmul(a, b)
    is_equal = np.array_equal(c, golden)
    result = "success" if is_equal else "failed"
    print("compare {}.".format(result))