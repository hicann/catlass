# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from torch_npu.testing.testcase import TestCase, run_tests
import torch_npu
import torch
import torch_act
torch.ops.load_library("../../../output/python_extension/libact_torch.so") # 手动指定so路径


class ActTest(TestCase):
    def test_basic_matmul_pybind(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        result = torch_act.basic_matmul(a, b, "float16")
        torch.npu.synchronize()
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)
        
    def test_basic_matmul_pybind_bf16(self):
        a = torch.ones((2, 3)).to(torch.bfloat16).npu()
        b = torch.ones((3, 4)).to(torch.bfloat16).npu()
        result = torch_act.basic_matmul(a, b, "bf16")
        torch.npu.synchronize()
        golden = torch.mm(a, b)
        self.assertRtolEqual(result.to(torch.float32), golden.to(torch.float32))

    def test_basic_matmul_torch_lib(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        result = torch.ops.ActTorch.basic_matmul(a, b, "float16")
        torch.npu.synchronize()
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)

    def test_grouped_matmul_slice_m_pybind(self):
        m_list = [16, 32, 64]
        k, n = 16, 16
        a_list = [torch.ones((m, k)).to(torch.float16).npu() for m in m_list]
        b_list = [torch.ones((k, n)).to(torch.float16).npu() for m in m_list]
        result = torch_act.grouped_matmul(a_list, b_list, "float16", False)
        torch.npu.synchronize()
        golden = torch_npu.npu_grouped_matmul(a_list, b_list)
        for i in range(len(m_list)):
            self.assertRtolEqual(result[i], golden[i])

    def test_grouped_matmul_slice_k_pybind(self):
        k_list = [16, 32, 64]
        m, n = 16, 16
        a_list = [torch.ones((m, k)).to(torch.float16).npu() for k in k_list]
        b_list = [torch.ones((k, n)).to(torch.float16).npu() for k in k_list]
        result = torch_act.grouped_matmul(a_list, b_list, "float16", True)
        torch.npu.synchronize()
        golden = torch_npu.npu_grouped_matmul(a_list, b_list)
        for i in range(len(k_list)):
            self.assertRtolEqual(result[i], golden[i])
        
    def test_optimized_matmul_pybind(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        result = torch_act.optimized_matmul(a, b, "float16")
        torch.npu.synchronize()
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)
        
if __name__ == "__main__":
    run_tests()
