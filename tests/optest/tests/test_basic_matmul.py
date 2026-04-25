import pytest
import torch
import torch_npu

import torch_catlass


def test_basic_matmul():
    m, n, k = 128, 128, 128
    
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(k, n, dtype=torch.float16, device="npu")
    
    print(f"\nInput A shape: {a.shape}, dtype: {a.dtype}")
    print(f"Input B shape: {b.shape}, dtype: {b.dtype}")
    print(f"Input A sample: {a[0, :5]}")
    print(f"Input B sample: {b[0, :5]}")
    
    result = torch_catlass.basic_matmul(a, b, "float16", False, False, False, False)
    expected = torch.matmul(a, b)
    
    print(f"\nResult shape: {result.shape}, dtype: {result.dtype}")
    print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
    print(f"Result sample: {result[0, :5]}")
    print(f"Expected sample: {expected[0, :5]}")
    print(f"Max diff: {(result - expected).abs().max().item()}")
    print(f"Mean diff: {(result - expected).abs().mean().item()}")
    
    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    
    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(result, expected, rtol=rtol, atol=atol), \
        f"Results not close: max diff = {(result - expected).abs().max().item()}"


def test_npu_arch_detection():
    arch = torch_catlass.get_npu_arch()
    assert arch in [2201, 3510]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
