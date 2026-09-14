# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import pytest
import torch
import torch_catlass

from common import only_on_3510
from mx_golden import prepare_fp8_e4m3_quant_matmul_inputs
from quant_matmul_test_utils import QUANT_MATMUL_SHAPES, check_quant_result, signed_scales, with_storage_offset


@only_on_3510
@pytest.mark.parametrize("shape", QUANT_MATMUL_SHAPES + [(17, 33, 65)])
@pytest.mark.parametrize("seed", range(10))
def test_ascend950_fp8_e4m3_quant_matmul(shape, seed, record_property):
    """Compare example 86 against the quantized CPU reference."""
    m, n, k = shape
    a, b, per_token, per_channel, expected = prepare_fp8_e4m3_quant_matmul_inputs(
        m, n, k, device="npu", seed=seed
    )
    if seed % 2:
        per_token, per_channel, expected = signed_scales(per_token, per_channel, expected)

    result = torch_catlass.ascend950_fp8_e4m3_quant_matmul(a, b, per_token, per_channel)

    check_quant_result(result, expected, record_property)


@only_on_3510
@pytest.mark.parametrize("input_index", range(4))
def test_fp8_quant_storage_offset(input_index, record_property):
    *inputs, expected = prepare_fp8_e4m3_quant_matmul_inputs(127, 130, 66)
    inputs[input_index] = with_storage_offset(inputs[input_index])
    result = torch_catlass.ascend950_fp8_e4m3_quant_matmul(*inputs)
    check_quant_result(result, expected, record_property)


@only_on_3510
@pytest.mark.parametrize("axis", range(3))
def test_fp8_quant_rejects_empty_shape(axis):
    shape = [16, 16, 16]
    shape[axis] = 0
    m, n, k = shape
    inputs = [torch.empty(s, dtype=torch.float8_e4m3fn, device="npu") for s in [(m, k), (k, n), (m,), (n,)]]
    with pytest.raises(RuntimeError, match="must be positive"):
        torch_catlass.ascend950_fp8_e4m3_quant_matmul(*inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
