import torch
from torch import Tensor


def ascend950_basic_syrk(
    matX: Tensor,
    outDType: str | torch.dtype | None = None,
) -> Tensor:
    """Run CATLASS Ascend950 Basic SYRK: ``Y = X @ X.T``.

    Source: example 82_ascend950_basic_syrk.

    Args:
        matX: Input matrix ``X`` with shape ``(M, K)`` on NPU.
        outDType: Output dtype. Defaults to ``matX.dtype``. Accepted strings
            include ``float16`` / ``fp16`` and ``bfloat16`` / ``bf16``.

    Returns:
        Output tensor ``Y`` with shape ``(M, M)`` on the active NPU device.
    """
    if outDType is None:
        outDType = matX.dtype
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        if dtype_lower in ("bf16", "bfloat16"):
            outDType = torch.bfloat16
        elif dtype_lower in ("fp16", "float16"):
            outDType = torch.float16
        else:
            outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.ascend950_basic_syrk(matX, outDType)
