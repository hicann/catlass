import pytest
import torch
import torch_npu
import torch_catlass

pytestmark = pytest.mark.skipif(
    torch_npu.npu.device_count() <= 0,
    reason="requires Ascend NPU",
)

def test_padding_splitk_matmul():
    for m, n, k in [(256, 512, 1024), (128, 128, 128), (13, 14, 15)]:
        a = torch.randn(m, k, dtype=torch.float16, device="npu")
        b = torch.randn(k, n, dtype=torch.float16, device="npu")
        r = torch_catlass.padding_splitk_matmul(a, b)
        e = torch.matmul(a, b)
        assert r.shape == (m, n)
        assert torch.allclose(r, e, rtol=1e-2, atol=1e-2), f"({m},{n},{k}) failed"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
