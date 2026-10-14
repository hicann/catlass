# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import re
from dataclasses import dataclass

import pytest
import torch
import torch_npu

import torch_catlass

try:
    import acl

    _device_name = acl.get_soc_name()
    _is_ascend950 = re.search(r"Ascend950(PR|DT)", _device_name, re.IGNORECASE) is not None
except Exception:  # noqa: BLE001 - allow collection without an NPU runtime
    _is_ascend950 = False

only_on_3510 = pytest.mark.skipif(not _is_ascend950, reason="This case only runs on 3510")

FP16_ATOL = 2.0**-9
FP16_RTOL = 2.0**-9
MIN_MATCHED_RATIO = 0.99
MAX_ABS_ERROR_LIMIT = 0.1


@dataclass(frozen=True)
class TaskCase:
    idx: int
    batch: int
    m: int
    k: int
    n: int

    @property
    def test_id(self):
        return f"{self.idx:03d}-b{self.batch}-m{self.m}-k{self.k}-n{self.n}"


# Keep CI coverage small while exercising regular and tail dimensions.
TEST_CASES = (
    TaskCase(idx=1, batch=2, m=128, k=128, n=128),
    TaskCase(idx=2, batch=2, m=130, k=130, n=130),
    TaskCase(idx=3, batch=2, m=129, k=128, n=128),
    TaskCase(idx=4, batch=2, m=128, k=130, n=1000),
)


def random_tensor(shape):
    return torch.randn(shape, dtype=torch.float16, device="npu")


def make_column_major_b(batch, k, n):
    storage = random_tensor((batch, n, k)).contiguous()
    return torch.as_strided(storage, (batch, k, n), (k * n, 1, k))


def run_and_check(case):
    torch.manual_seed(case.idx)
    torch_npu.npu.manual_seed_all(case.idx)
    a = random_tensor((case.batch, case.m, case.k)).contiguous()
    b = make_column_major_b(case.batch, case.k, case.n)

    result = torch_catlass.ascend950_batched_matmul_softmax(a, b, "float16")
    expected = torch.softmax(torch.bmm(a, b), dim=-1)

    assert result.shape == (case.batch, case.m, case.n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.isfinite(result).all().item()
    abs_error = (result - expected).abs()
    matched = abs_error <= FP16_ATOL + FP16_RTOL * expected.abs()
    assert matched.float().mean().item() >= MIN_MATCHED_RATIO
    assert abs_error.max().item() <= MAX_ABS_ERROR_LIMIT


@only_on_3510
@pytest.mark.parametrize("case", TEST_CASES, ids=lambda case: case.test_id)
def test_ascend950_batched_matmul_softmax(case):
    run_and_check(case)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
