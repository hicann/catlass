import pytest
import torch
import torch_catlass
import torch_npu  # noqa: F401  required for NPU tensor dispatch
from common import only_on_2201


@only_on_2201
def test_planar_complex_matmul():
    m, n, k = 128, 256, 256
    torch.manual_seed(42)

    a_real = (torch.rand((m, k), dtype=torch.float32, device="npu") * 10 - 5).to(torch.float16)
    a_imag = (torch.rand((m, k), dtype=torch.float32, device="npu") * 10 - 5).to(torch.float16)
    b_real = (torch.rand((k, n), dtype=torch.float32, device="npu") * 10 - 5).to(torch.float16)
    b_imag = (torch.rand((k, n), dtype=torch.float32, device="npu") * 10 - 5).to(torch.float16)

    result_real, result_imag = torch_catlass.planar_complex_matmul(a_real, a_imag, b_real, b_imag)
    expected_real = torch.matmul(a_real.float(), b_real.float()) - torch.matmul(
        a_imag.float(), b_imag.float()
    )
    expected_imag = torch.matmul(a_real.float(), b_imag.float()) + torch.matmul(
        a_imag.float(), b_real.float()
    )

    assert result_real.shape == (m, n)
    assert result_imag.shape == (m, n)
    assert result_real.dtype == torch.float32
    assert result_imag.dtype == torch.float32
    assert result_real.device.type == "npu"
    assert result_imag.device.type == "npu"

    rtol = 5e-2
    atol = 5e-2
    assert torch.allclose(result_real, expected_real, rtol=rtol, atol=atol), (
        f"real result max diff = {(result_real - expected_real).abs().max().item()}"
    )
    assert torch.allclose(result_imag, expected_imag, rtol=rtol, atol=atol), (
        f"imag result max diff = {(result_imag - expected_imag).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
