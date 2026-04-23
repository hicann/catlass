from typing import Optional
from torch import Tensor
import torch


def basic_matmul(a: Tensor, b: Tensor, out: Optional[Tensor] = None, use_nz_b: bool = False) -> Tensor:
    return torch.ops.catlass.basic_matmul(a, b, out, use_nz_b)
