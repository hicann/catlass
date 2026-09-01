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
import torch_npu
import torch_catlass

from common import only_on_3510


@only_on_3510
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("m,k", [(256, 256), (512, 128), (300, 200)])
def test_ascend950_basic_syrk(m, k, dtype):
    """Compare CATLASS Ascend950 Basic SYRK against torch.matmul(X, X.T)."""
    x = torch.randn(m, k, dtype=dtype, device="npu")

    result = torch_catlass.ascend950_basic_syrk(x)
    expected = torch.matmul(x.float(), x.float().T).to(dtype)

    assert result.shape == (m, m)
    assert result.dtype == dtype
    assert result.device.type == "npu"
    assert torch.allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
