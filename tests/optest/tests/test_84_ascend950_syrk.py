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


def _assert_allclose_with_diagnostics(
    actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float, expect_symmetric: bool
) -> None:
    if actual.dim() == 3:
        for batch_idx in range(actual.shape[0]):
            _assert_allclose_with_diagnostics(
                actual[batch_idx],
                expected[batch_idx],
                rtol=rtol,
                atol=atol,
                expect_symmetric=expect_symmetric,
            )
        return

    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    if torch.allclose(actual_fp32, expected_fp32, rtol=rtol, atol=atol):
        return

    abs_error = (actual_fp32 - expected_fp32).abs()
    close = torch.isclose(actual_fp32, expected_fp32, rtol=rtol, atol=atol)
    rows = torch.arange(actual.shape[0], device=actual.device)[:, None]
    cols = torch.arange(actual.shape[1], device=actual.device)[None, :]
    lower = rows > cols
    upper = rows < cols

    flat_index = abs_error.argmax().item()
    row, col = divmod(flat_index, actual.shape[1])
    details = [
        f"mismatches={(~close).sum().item()}/{actual.numel()}",
        f"lower_mismatches={((~close) & lower).sum().item()}",
        f"upper_mismatches={((~close) & upper).sum().item()}",
        f"diagonal_mismatches={(~close.diagonal()).sum().item()}",
        f"max_abs_error={abs_error[row, col].item()} at ({row}, {col})",
        f"actual={actual_fp32[row, col].item()}",
        f"expected={expected_fp32[row, col].item()}",
    ]
    if expect_symmetric:
        details.append(f"symmetry_max_abs={(actual_fp32 - actual_fp32.T).abs().max().item()}")
    raise AssertionError(", ".join(details))


@only_on_3510
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("b,m,k", [(1, 256, 256), (1, 512, 128), (1, 1024, 1024), (14, 1024, 1024)])
@pytest.mark.parametrize(
    "alpha,beta",
    [
        (1.0, 0.0),
        (0.5, 1.0),
        (-1.25, 0.75),
    ],
    ids=["a1b0", "a05b1", "aneg"],
)
def test_ascend950_syrk(b, m, k, dtype, alpha, beta):
    """Compare CATLASS Ascend950 SYRK against alpha*X@X.T + beta*Y."""
    torch.manual_seed(0)
    x = torch.randn((b, m, k), dtype=dtype, device="npu")
    # Asymmetric Y (not forced symmetric) to validate dual-path AXPBY.
    y = torch.randn((b, m, m), dtype=dtype, device="npu")

    result = torch_catlass.ascend950_syrk(x, y, alpha=alpha, beta=beta)
    expected = (alpha * torch.matmul(x.float(), x.float().transpose(-1, -2)) + beta * y.float()).to(dtype)

    assert result.shape == (b, m, m)
    assert result.dtype == dtype
    assert result.device.type == "npu"
    _assert_allclose_with_diagnostics(
        result, expected, rtol=1e-2, atol=1e-2, expect_symmetric=(beta == 0.0)
    )


@only_on_3510
def test_ascend950_syrk_mixed_y_d_dtype():
    """Y (input) and D (output) may use different fp16/bf16 dtypes."""
    torch.manual_seed(0)
    x = torch.randn((1, 256, 256), dtype=torch.bfloat16, device="npu")
    y = torch.randn((1, 256, 256), dtype=torch.bfloat16, device="npu")
    out_dtype = torch.float16

    result = torch_catlass.ascend950_syrk(x, y, alpha=1.0, beta=1.0, outDType=out_dtype)
    expected = (
        torch.matmul(x.float(), x.float().transpose(-1, -2)) + y.float()
    ).to(out_dtype)

    assert result.shape == (1, 256, 256)
    assert result.dtype == out_dtype
    _assert_allclose_with_diagnostics(result, expected, rtol=1e-2, atol=1e-2, expect_symmetric=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
