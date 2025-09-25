import random
from typing import Dict, List, Literal, Union

import torch


def generate_sequence_split(n, sum_num):
    split_points = sorted([random.randint(0, sum_num) for _ in range(n - 1)])
    split_points = [0] + split_points + [sum_num]
    sequence = [split_points[i + 1] - split_points[i] for i in range(n)]
    sequence[-1] = sum_num - sum(sequence[:-1])
    return sequence


def calculate_prefix_sum(sequence):
    prefix_sum = []
    current_sum = 0
    for num in sequence:
        current_sum += num
        prefix_sum.append(current_sum)
    return prefix_sum


def grouped_matmul_split_m_golden(
    a: torch.Tensor,
    b: torch.Tensor,
    group_list: Union[torch.Tensor, List[int]],
    dtype: torch.dtype,
    transpose_a: bool = False,
    transpose_b: bool = False,
):
    assert not transpose_a
    if isinstance(group_list, list):
        a_list = torch.split(a, group_list)
        result_list = []
        for i in range(len(group_list)):
            a_g = a_list[i] if not transpose_a else a_list[i].T
            b_g = b[i] if not transpose_b else b[i].T
            result_list.append(a_g @ b_g)
        return torch.cat(result_list).to(dtype)


def grouped_matmul_split_k_golden(
    a: torch.Tensor,
    b: torch.Tensor,
    group_list: Union[torch.Tensor, List[int]],
    dtype: torch.dtype,
    transpose_a: bool = True,
    transpose_b: bool = False,
):
    assert transpose_a and not transpose_b
    if isinstance(group_list, list):
        a_list = torch.split(a, group_list)
        b_list = torch.split(b, group_list)
        result_list = []
        for i in range(len(group_list)):
            a_g = a_list[i].T
            b_g = b_list[i]
            result_list.append(a_g @ b_g)
        return torch.stack(result_list).to(dtype)


def grouped_matmul_golden(
    a: torch.Tensor,
    b: torch.Tensor,
    group_list: Union[torch.Tensor, List[int]],
    out_dtype: torch.dtype,
    transpose_a: bool,
    transpose_b: bool,
    slice_axis: Literal["m", "k"],
):
    if slice_axis == "m":
        return grouped_matmul_split_m_golden(
            a, b, group_list, out_dtype, transpose_a, transpose_b
        )
    elif slice_axis == "k":
        return grouped_matmul_split_k_golden(
            a, b, group_list, out_dtype, transpose_a, transpose_b
        )


def grouped_matmul_gen_case(
    m: int,
    n: int,
    k: int,
    g: int,
    in_dtype: torch.dtype,
    transpose_a: bool,
    transpose_b: bool,
    slice_axis: Literal["m", "k"],
) -> List[torch.Tensor]:
    if slice_axis == "m":
        group_list = generate_sequence_split(g, m)
        group_list_prefix_sum = calculate_prefix_sum(group_list)
        m_sum = group_list_prefix_sum[-1]
        a = torch.randn((m_sum, k), device="npu").to(in_dtype)
        if transpose_a:
            a = a.permute([0, 1])
        b = torch.randn((g, k, n), device="npu").to(in_dtype)
        if transpose_b:
            b = b.permute([0, 2, 1])
        group_list_prefix_sum_tensor = torch.tensor(
            group_list_prefix_sum, device="npu"
        ).to(torch.int64)
        return [a, b, group_list_prefix_sum_tensor]
    elif slice_axis == "k":
        group_list = generate_sequence_split(g, k)
        group_list_prefix_sum = calculate_prefix_sum(group_list)
        k_sum = group_list_prefix_sum[-1]
        a = torch.randn((k_sum, m), device="npu").to(in_dtype)
        if transpose_a:
            a = a.permute([0, 1])
        b = torch.randn((k_sum, n), device="npu").to(in_dtype)
        if transpose_b:
            b = b.permute([0, 1])
        group_list_prefix_sum_tensor = torch.tensor(
            group_list_prefix_sum, device="npu"
        ).to(torch.int64)
        return [a, b, group_list_prefix_sum_tensor]
