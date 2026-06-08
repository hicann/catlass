import pytest
import torch
import torch_npu
import torch_catlass

pytestmark = pytest.mark.skipif(
    torch_npu.npu.device_count() <= 0,
    reason="torch-catlass integration tests require an available Ascend NPU",
)


def test_matmul_silu():
    m, n, k = 256, 128, 64
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(k, n, dtype=torch.float16, device="npu")

    result = torch_catlass.matmul_silu(a, b, "float16", False, False, False, False)
    expected = torch.nn.functional.silu(torch.matmul(a, b))

    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
