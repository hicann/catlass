import pytest
import torch
import torch_npu

import torch_catlass


def test_basic_matmul():
    m, n, k = 1024, 1024, 1024
    
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(k, n, dtype=torch.float16, device="npu")
    
    result = torch_catlass.ops.basic_matmul(a, b)
    
    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"


def test_basic_matmul_with_output():
    m, n, k = 512, 512, 512
    
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(k, n, dtype=torch.float16, device="npu")
    out = torch.empty(m, n, dtype=torch.float16, device="npu")
    
    result = torch_catlass.ops.basic_matmul(a, b, out=out)
    
    assert result is out
    assert result.shape == (m, n)


def test_npu_arch_detection():
    arch = torch_catlass.get_npu_arch()
    assert arch in [2201, 3510]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
