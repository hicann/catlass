import torch
from torch import Tensor

def gemv_aic(
    matA: Tensor, vecX: Tensor, vecY: Tensor,
    outDType: str | torch.dtype = torch.float32,
    alpha: float = 1.0, beta: float = 0.0,
) -> Tensor:
    """Run CATLASS GEMV (AIC) with alpha/beta scaling on NPU tensors.

    Computes ``out = alpha * A @ X + beta * Y``.

    Args:
        matA:  Input matrix ``(M, N)``.
        vecX:  Input vector ``(N,)``.
        vecY:  Bias vector ``(M,)`` scaled by ``beta``.
        outDType: Output dtype.
        alpha:   Scaling factor for the matrix-vector product.
        beta:    Scaling factor for the bias vector ``vecY``.

    Returns:
        Output tensor ``(M,)`` on the active NPU device.
    """
    if isinstance(outDType, str):
        dt = outDType.lower()
        outDType = getattr(torch, dt, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.gemv_aic(
        matA, vecX, vecY, outDType, alpha, beta, False, False, False, False)
