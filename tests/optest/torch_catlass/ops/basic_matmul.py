from typing import Optional
from torch import Tensor
import torch


def basic_matmul(
    mat1: Tensor, 
    mat2: Tensor, 
    outDType: str = "float16",
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False
) -> Tensor:
    """
    Basic matrix multiplication.
    
    Args:
        mat1: Input matrix A
        mat2: Input matrix B
        outDType: Output data type ("float16", "float32", "bf16")
        transA: Whether A is transposed
        transB: Whether B is transposed
        formatA: Whether A uses NZ format
        formatB: Whether B uses NZ format
    
    Returns:
        Output tensor with shape (m, n)
    """
    return torch.ops.catlass.basic_matmul(
        mat1, mat2, outDType, transA, transB, formatA, formatB
    )
