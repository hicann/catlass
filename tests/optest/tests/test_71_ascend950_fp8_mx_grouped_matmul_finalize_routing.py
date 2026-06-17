import re

import pytest
import torch
import torch_npu

from mx_golden import prepare_fp8_mx_grouped_matmul_finalize_routing_inputs

def _is_ascend950() -> bool:
    if torch_npu.npu.device_count() <= 0:
        return False
    name = torch_npu.npu.get_device_name()
    return bool(re.search(r"Ascend950(PR|DT)", name, re.I))

pytestmark = pytest.mark.skipif(
    not _is_ascend950(),
    reason="example 71_ascend950_fp8_mx_grouped_matmul_finalize_routing requires Ascend 950 NPU",
)

def _run_and_check(
    m: int, n: int, k: int, problem_count: int, batch: int,
    enable_bias: bool, enable_shared_input: bool,
    shared_input_weight: float = 0.5, shared_input_offset: int = 0,
    data_parallel_size: int = 1, group_list_type: int = 0,
    trans_b: bool = False, quant_type: torch.dtype = torch.float8_e4m3fn,
) -> None:
    import torch_catlass

    (a_fp8, b_fp8, a_scale, b_scale,
     group_list, logit, row_index,
     bias, shared_input, expected) = prepare_fp8_mx_grouped_matmul_finalize_routing_inputs(
        m=m, n=n, k=k, problem_count=problem_count, batch=batch,
        data_parallel_size=data_parallel_size,
        enable_bias=enable_bias,
        enable_shared_input=enable_shared_input,
        shared_input_weight=shared_input_weight,
        shared_input_offset=shared_input_offset,
        group_list_type=group_list_type,
        trans_b=trans_b,
        quant_type=quant_type,
        device="npu",
    )
    print("expected:", expected)

    result = torch_catlass.ascend950_fp8_mx_grouped_matmul_finalize_routing(
        a_fp8, b_fp8, a_scale, b_scale,
        group_list, logit, row_index, bias, shared_input,
        batch=batch,
        data_parallel_size=data_parallel_size,
        shared_input_weight=shared_input_weight,
        shared_input_offset=shared_input_offset,
        group_list_type=group_list_type,
        transA=False, transB=trans_b,
    )
    print("result:", result)

    assert result.shape == (batch, n), f"shape mismatch: {result.shape} vs {(batch, n)}"
    assert result.dtype == torch.float32, f"dtype mismatch: {result.dtype}"
    assert result.device.type == "npu"

    rtol, atol = 1e-1, 1e-1
    assert torch.allclose(result.cpu().float(), expected.float(), rtol=rtol, atol=atol), (
        f"max diff = {(result.cpu().float() - expected.float()).abs().max().item()}"
    )

def test_basic_with_bias_and_shared_input():
    """Grouped MX FP8 matmul + finalize routing with bias and shared input."""
    _run_and_check(
        m=512, n=256, k=1024, problem_count=2, batch=4,
        enable_bias=True, enable_shared_input=True,
    )

def test_no_bias():
    """Grouped MX FP8 matmul + finalize routing without bias."""
    _run_and_check(
        m=512, n=256, k=1024, problem_count=2, batch=4,
        enable_bias=False, enable_shared_input=True,
    )

def test_no_shared_input():
    """Grouped MX FP8 matmul + finalize routing without shared input."""
    _run_and_check(
        m=512, n=256, k=1024, problem_count=2, batch=4,
        enable_bias=True, enable_shared_input=False,
    )

def test_no_bias_no_shared_input():
    """Grouped MX FP8 matmul + finalize routing, pure scatter-add."""
    _run_and_check(
        m=512, n=256, k=1024, problem_count=2, batch=4,
        enable_bias=False, enable_shared_input=False,
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
