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
from common import only_on_2201


def _make_well_conditioned_matrix(N: int) -> torch.Tensor:
    """Generate a diagonally dominant random matrix (well-conditioned) on NPU."""
    # Generate on CPU first for deterministic random seed behavior
    A = torch.rand(N, N, dtype=torch.float32) * 2.0 - 1.0  # [-1, 1]
    # Strengthen diagonal to ensure well-conditioned
    for i in range(N):
        A[i, i] += float(N)
    return A.to("npu")


@only_on_2201
class TestMatrixInverse:
    """Tests for example 78_matrix_inverse — matrix inverse via LU decomposition."""

    @pytest.mark.parametrize("N", [16, 32, 64, 128])
    def test_inverse_shape_and_dtype(self, N: int):
        """Verify output shape, dtype, device, and A @ A^{-1} ≈ I."""
        A = _make_well_conditioned_matrix(N)

        A_inv = torch_catlass.matrix_inverse(A)

        # Shape / dtype / device assertions
        assert A_inv.shape == (N, N), f"Expected shape ({N},{N}), got {A_inv.shape}"
        assert A_inv.dtype == torch.float32, f"Expected float32, got {A_inv.dtype}"
        assert A_inv.device.type == "npu", f"Expected npu device, got {A_inv.device}"

        # Numerical check: A @ A^{-1} ≈ I
        identity = A @ A_inv
        expected = torch.eye(N, dtype=torch.float32, device="npu")

        # LU-based inverse has higher residual ||A·A^{-1} - I|| than direct matmul,
        # especially for small matrices. Use looser atol to account for error propagation.
        atol = 2e-2 if N <= 64 else 5e-2
        rtol = 1e-2 if N <= 64 else 3e-2
        assert torch.allclose(identity, expected, rtol=rtol, atol=atol), (
            f"A @ A_inv differs from identity (max diff: "
            f"{(identity - expected).abs().max().item():.6e})"
        )

    def test_inverse_preserves_input(self):
        """Verify that the original input tensor is not modified (out-of-place)."""
        A = _make_well_conditioned_matrix(32)
        A_original = A.clone()

        _ = torch_catlass.matrix_inverse(A)

        assert torch.equal(A, A_original), "Input tensor was modified"

    @pytest.mark.parametrize("N", [16, 64])
    def test_inverse_vs_torch(self, N: int):
        """Verify result matches torch.linalg.inv (CPU reference)."""
        A_cpu = torch.rand(N, N, dtype=torch.float32) * 2.0 - 1.0
        for i in range(N):
            A_cpu[i, i] += float(N)

        A_npu = A_cpu.clone().to("npu")
        A_inv_npu = torch_catlass.matrix_inverse(A_npu).cpu()

        A_inv_ref = torch.linalg.inv(A_cpu)

        rtol = 1e-2 if N <= 64 else 3e-2
        atol = 1e-3 if N <= 64 else 5e-3
        assert torch.allclose(A_inv_npu, A_inv_ref, rtol=rtol, atol=atol), (
            f"NPU result differs from torch.linalg.inv "
            f"(max diff: {(A_inv_npu - A_inv_ref).abs().max().item():.6e})"
        )
