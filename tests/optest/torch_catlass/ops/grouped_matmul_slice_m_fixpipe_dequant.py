import torch
from torch import Tensor

def grouped_matmul_slice_m_fixpipe_dequant(
    mat1: Tensor,
    mat2: Tensor,
    group_list: Tensor,
    per_tensor_scale: Tensor,
    per_channel_scale: Tensor,
    per_channel_mode: bool = True,
) -> Tensor:
    """Run CATLASS Ascend950 grouped matmul slice-M fixpipe dequant on NPU tensors [dev-mode].

    Source: example 70_ascend950_grouped_matmul_slice_m_fixpipe_dequant.

    A/B matrices are int8, output is float16. Combines grouped matmul (slice-M)
    with fixpipe dequantization. The quant mode (PER_CHANNEL / PER_TENSOR) is
    selected at JIT compile time.

    Args:
        mat1: Left input matrix, shape ``(M, K)``, dtype int8.
        mat2: Right input matrix, shape ``(K, N)``, dtype int8.
        group_list: Group size list, shape ``(G,)``, dtype int64.
        per_tensor_scale: Per-tensor dequant scale, shape ``(1,)``, dtype float32.
        per_channel_scale: Per-channel dequant scales,
            shape ``(N,)``, dtype uint64 (packed float32 bit pattern).
        per_channel_mode: ``True`` for PER_CHANNEL, ``False`` for PER_TENSOR.

    Returns:
        Output tensor with shape ``(M, N)``, dtype float16.
    """
    return torch.ops.catlass.grouped_matmul_slice_m_fixpipe_dequant(
        mat1, mat2, group_list, per_tensor_scale, per_channel_scale, per_channel_mode
    )
