# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import sys
import numpy as np
import mskpp

sys.path.append("../")
from common.act_type import GemmCoord, RowMajor, ColumnMajor


def get_kernel():
    kernel_file = "grouped_matmul.cpp"
    kernel_name = "GroupedMatmul"
    build_script = "../../scripts/jit_build.sh" # kernel compile script

    config = mskpp.KernelInvokeConfig(kernel_file, kernel_name)
    gen_file = mskpp.Launcher(config).code_gen()
    kernel = mskpp.compile(build_script=build_script, launch_src_file=gen_file)
    return kernel


"""
To enable the autotune feature, it is required to add the "// tunable" marker to
the code lines in "grouped_matmul.cpp", e.g.
    ...
    57    using L1TileShape = GemmShape<128, 256, 256>; // tunable
    58    using L0TileShape = GemmShape<128, 256, 64>; // tunable
"""
@mskpp.autotune(configs=[
    {'L1TileShape': 'GemmShape<64, 64, 64>', 'L0TileShape': 'GemmShape<64, 64, 64>'},
    {'L1TileShape': 'GemmShape<64, 64, 128>', 'L0TileShape': 'GemmShape<64, 64, 64>'},
    {'L1TileShape': 'GemmShape<64, 128, 128>', 'L0TileShape': 'GemmShape<64, 128, 64>'},
    {'L1TileShape': 'GemmShape<128, 128, 128>', 'L0TileShape': 'GemmShape<128, 128, 64>'},
    {'L1TileShape': 'GemmShape<128, 64, 128>', 'L0TileShape': 'GemmShape<128, 64, 64>'},
], warmup=1000, repeat=10, device_ids=[0])
def grouped_matmul(problem_count, problem_shape_list,
        a, layout_a_list,
        b, layout_b_list,
        c, layout_c_list,
        template_param=[RowMajor, ColumnMajor, RowMajor]
        # template_param=["Act::layout::RowMajor", "Act::layout::ColumnMajor",
        #                 "Act::layout::RowMajor"] # alternative approach
    ):
    # This function's input arguments must exactly match the kernel function.
    kernel = get_kernel()
    blockdim = 20
    return kernel[blockdim](problem_count, problem_shape_list,
                            a, layout_a_list,
                            b, layout_b_list,
                            c, layout_c_list) # invoke the kernel


def gen_golden_data(problem_shape_list, a, b):
    prev = 0
    results = []
    b = b.reshape(problem_count, n, k)
    b = np.transpose(b, (0, 2, 1))
    for i, coord in enumerate(problem_shape_list):
        start = prev
        end = start + coord.m
        a_block = a[start:end, :].astype(np.float32)
        b_layer = b[i, :, :].astype(np.float32)
        res = a_block @ b_layer
        results.append(res)
        prev = end
    return np.vstack(results).reshape(m * n)


def data_compare(problem_shape_list, a, b, c):
    golden = gen_golden_data(problem_shape_list, a, b).astype(np.half)
    rtol = 1.0 / 256
    bool_matrix = np.abs(golden - c) < rtol
    result = "success" if bool_matrix.all() else "failed"
    print("compare {}.".format(result))


if __name__ == "__main__":

    m = 1024
    n = 768
    k = 512
    problem_count = 8
    group_list = [128, 256, 512, 515, 568, 579, 678, 1024]

    # 创建结构体数组
    problem_shape_list = (GemmCoord * problem_count)()
    layout_a_list = (RowMajor * problem_count)()
    layout_b_list = (ColumnMajor * problem_count)()
    layout_c_list = (RowMajor * problem_count)()

    # 向结构体数组赋值
    prev = 0
    for i, group in enumerate(group_list):
        current_m = group - prev
        problem_shape_list[i] = GemmCoord(current_m, n, k)
        layout_a_list[i] = RowMajor(current_m, k)
        layout_b_list[i] = ColumnMajor(k, n)
        layout_c_list[i] = RowMajor(current_m, n)
        prev = group

    # 创建输入输出tensor
    a = np.random.randint(-5, 5, [m, k]).astype(np.half)
    b = np.random.randint(-5, 5, [problem_count, n, k]).astype(np.half)
    c = np.zeros([m * n]).astype(np.half)

    grouped_matmul(problem_count, problem_shape_list, a, layout_a_list, b, layout_b_list, c, layout_c_list)
    data_compare(problem_shape_list, a, b, c)
