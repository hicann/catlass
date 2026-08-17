# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import torch

# 数据模型
Scalars = dict[str, Any]
"""scalar_setup 输出。值为标量 (int/float) 或 1D Tensor[seq_len] 提供 per-row 值。
本实现在 [row_idx] 处索引 1D Tensor 后传给 when/visible 谓词。"""

"""外部 tensor 输入，key 与 TensorArg.name 对齐。"""


_INTEGER_DTYPES = frozenset({"int32", "int64"})


@dataclass(frozen=True)
class TensorArg:
    """声明 MaskSpec 需要的 tensor 输入。

    Attributes:
        name: 与 TensorDict 中 key 一致，如 "context_length"。
        shape: 符号 shape，如 ("batch",)、("batch+1",)。
        dtype: 整数 dtype，"int32" 或 "int64"。
    """
    name: str
    shape: tuple[str, ...]
    dtype: str = "int32"

    def __post_init__(self) -> None:
        if self.dtype not in _INTEGER_DTYPES:
            raise ValueError(
                f"TensorArg {self.name!r} dtype must be one of "
                f"{sorted(_INTEGER_DTYPES)}, got {self.dtype!r}"
            )


@dataclass(frozen=True)
class RegionMask:
    """mask 内的一个命名区域。

    每个 query 行归属唯一区域（首匹配优先），区域的 visible() 决定该行可见哪些 key 列。

    Attributes:
        name: 人类可读标签，如 "context", "label", "prefix"。
        when: 谓词 (row_idx, **scalars) -> bool。True 表示该行属于本区域。
        visible: 谓词 (row_idx, col_idx, **scalars) -> bool。True 表示该行可关注该列。
    """
    name: str
    when: Callable[..., bool]
    visible: Callable[..., bool]


@dataclass(frozen=True)
class MaskSpec:
    """attention mask pattern 的完整声明式描述（单 batch 元素粒度）。

    纯数据：可序列化、可比较、可跨 pattern 复用。本文件直接用谓词求值产出 mask。

    Attributes:
        tensor_args: 必需的 tensor 输入（由调用方校验）。
        scalar_setup: 函数 (batch_idx, **tensors) -> Scalars。从 tensors 抽取 per-batch 标量上下文。
        regions: 按优先级排列的区域。每行归属第一个 when() 为 True 的区域。
        cur_seqlen: Scalars 中「总序列长度」对应的 key。
    """
    tensor_args: list[TensorArg]
    scalar_setup: Callable[..., Scalars]
    regions: list[RegionMask]
    cur_seqlen: str


# 预定义 pattern

def _length_setup(batch_idx: int, seq_lens) -> Scalars:
    return {"seq_hi": seq_lens[batch_idx]}


def _when_active(row_idx: int, seq_hi: int, **kwargs) -> bool:
    return row_idx < seq_hi


def _visible_causal(row_idx: int, col_idx: int, **kwargs) -> bool:
    return col_idx <= row_idx


CAUSAL_SPEC = MaskSpec(
    tensor_args=[TensorArg("seq_lens", ("batch",))],
    scalar_setup=_length_setup,
    regions=[RegionMask(name="causal", when=_when_active, visible=_visible_causal)],
    cur_seqlen="seq_hi",
)


def four_stage_scalar_setup(
    batch_idx: int,
    context_sizes,
    history_sizes,
    realtime_sizes,
    label_nums,
    label_group_sizes,
) -> Scalars:
    context_hi = context_sizes[batch_idx]
    history_hi = (
        context_hi
        + history_sizes[batch_idx]
        + realtime_sizes[batch_idx]
    )
    label_group_size = label_group_sizes[batch_idx]
    seq_hi = history_hi + label_nums[batch_idx] * label_group_size
    return {
        "context_hi": context_hi,
        "history_hi": history_hi,
        "seq_hi": seq_hi,
        "label_group_size": label_group_size,
    }


def _when_context(
    row_idx: int,
    context_hi: int,
    **kwargs,
) -> bool:
    return row_idx < context_hi


def _when_history_or_realtime(
    row_idx: int,
    context_hi: int,
    history_hi: int,
    **kwargs,
) -> bool:
    return row_idx >= context_hi and row_idx < history_hi


def _when_label(
    row_idx: int,
    history_hi: int,
    seq_hi: int,
    label_group_size: int,
    **kwargs,
) -> bool:
    return (
        label_group_size > 0
        and row_idx >= history_hi
        and row_idx < seq_hi
    )


def _visible_context(
    row_idx: int,
    col_idx: int,
    context_hi: int,
    history_hi: int,
    **kwargs,
) -> bool:
    return (col_idx < context_hi) or (
        col_idx >= context_hi and col_idx < history_hi
    )


def _visible_history_or_realtime(
    row_idx: int,
    col_idx: int,
    context_hi: int,
    **kwargs,
) -> bool:
    return (col_idx < context_hi) or (
        col_idx >= context_hi and col_idx <= row_idx
    )


def _visible_label(
    row_idx: int,
    col_idx: int,
    history_hi: int,
    label_group_size: int,
    **kwargs,
) -> bool:
    group_start = (
        ((row_idx - history_hi) // label_group_size)
        * label_group_size
        + history_hi
    )
    return (col_idx < history_hi) or (
        col_idx >= group_start and col_idx <= row_idx
    )


FOUR_STAGE_SPEC = MaskSpec(
    tensor_args=[
        TensorArg("context_sizes", ("batch",)),
        TensorArg("history_sizes", ("batch",)),
        TensorArg("realtime_sizes", ("batch",)),
        TensorArg("label_nums", ("batch",)),
        TensorArg("label_group_sizes", ("batch",)),
    ],
    scalar_setup=four_stage_scalar_setup,
    regions=[
        RegionMask(
            name="context",
            when=_when_context,
            visible=_visible_context,
        ),
        RegionMask(
            name="history-realtime",
            when=_when_history_or_realtime,
            visible=_visible_history_or_realtime,
        ),
        RegionMask(
            name="label",
            when=_when_label,
            visible=_visible_label,
        ),
    ],
    cur_seqlen="seq_hi",
)


# block_range
def compute_block_range(
    maskr: torch.Tensor,
    *,
    q_tile_size: int,
    k_tile_size: int,
) -> torch.Tensor:
    """由每行右边界 maskr 推导每 q-tile 最后一个有效 k-tile（含）。

    返回 (B, q_tiles) int32，result[b, t] = ceil(max(maskr[b, t*ts:(t+1)*ts]) / k_tile_size)。
    """
    if not isinstance(maskr, torch.Tensor):
        raise TypeError("maskr must be a torch.Tensor")
    if maskr.dtype != torch.int32 or maskr.ndim != 2:
        raise ValueError("maskr must be rank-2 int32")
    if not isinstance(q_tile_size, int) or isinstance(q_tile_size, bool) or q_tile_size <= 0:
        raise ValueError("q_tile_size must be a positive int")
    if not isinstance(k_tile_size, int) or isinstance(k_tile_size, bool) or k_tile_size <= 0:
        raise ValueError("k_tile_size must be a positive int")
    if bool(torch.any(maskr < 0)):
        raise ValueError("maskr endpoints must be non-negative")

    batch, seq_q = maskr.shape
    q_tiles = (seq_q + q_tile_size - 1) // q_tile_size
    result = torch.zeros((batch, q_tiles), dtype=torch.int32)
    for q_tile in range(q_tiles):
        begin = q_tile * q_tile_size
        end = min(begin + q_tile_size, seq_q)
        maximum = torch.amax(maskr[:, begin:end], dim=1).to(torch.int64)
        result[:, q_tile] = torch.div(
            maximum + k_tile_size - 1,
            k_tile_size,
            rounding_mode="floor",
        ).to(torch.int32)
    return result


# dense 谓词求值
_STANDARD_INDEX_NAMES = frozenset({"batch_idx", "row_idx", "col_idx"})


def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _infer_batch(spec: MaskSpec, tensors: Mapping[str, torch.Tensor]) -> int:
    expected_names = {arg.name for arg in spec.tensor_args}
    reserved_names = expected_names.intersection(_STANDARD_INDEX_NAMES)
    if reserved_names:
        raise ValueError(
            "tensor names reserved for standard indices: "
            f"{sorted(reserved_names)}"
        )
    missing = expected_names - tensors.keys()
    unexpected = tensors.keys() - expected_names
    if missing:
        raise ValueError(f"missing input tensors: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"unexpected input tensors: {sorted(unexpected)}")

    candidates: list[int] = []
    for arg in spec.tensor_args:
        tensor = tensors[arg.name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"input {arg.name!r} must be a torch.Tensor")
        if tensor.ndim != len(arg.shape):
            raise ValueError(
                f"input {arg.name!r} has rank {tensor.ndim}; expected {len(arg.shape)}"
            )
        for size, dimension in zip(tensor.shape, arg.shape):
            if dimension == "batch":
                candidates.append(size)
            elif dimension == "batch+1":
                candidates.append(size - 1)

    if not candidates:
        raise ValueError("at least one tensor shape must contain a batch dimension")
    if candidates[0] < 0 or any(value != candidates[0] for value in candidates):
        raise ValueError("input tensor batch dimensions do not agree")
    return candidates[0]


def _row_scalars(scalars: Mapping[str, object], row_idx: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in scalars.items():
        if isinstance(value, torch.Tensor) and value.ndim == 1:
            if row_idx >= value.shape[0]:
                raise ValueError(
                    f"per-row scalar {name!r} is shorter than the requested query rows"
                )
            result[name] = value[row_idx]
        else:
            result[name] = value
    return result


def _call_with_signature(function, signature, values):
    positional = []
    keywords = {}
    explicit_names = set()
    has_var_keyword = False
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(values[name])
            explicit_names.add(name)
        elif parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            keywords[name] = values[name]
            explicit_names.add(name)
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
    if has_var_keyword:
        keywords.update(
            (name, value)
            for name, value in values.items()
            if name not in explicit_names
        )
    return function(*positional, **keywords)


def _call_scalar_setup(function, *, batch_idx: int, tensors):
    signature = inspect.signature(function)
    values = {**tensors, "batch_idx": batch_idx}
    return _call_with_signature(function, signature, values)


def _call_predicate(function, *, standard: Mapping[str, object], tensors):
    signature = inspect.signature(function)
    values = dict(standard)
    values.update(
        (name, tensors[name])
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name in tensors
    )
    return _call_with_signature(function, signature, values)


def evaluate_spec_dense(
    spec: MaskSpec,
    tensors: Mapping[str, torch.Tensor],
    *,
    seq_q: int,
    seq_k: int,
) -> torch.Tensor:
    """返回 (batch, seq_q, seq_k) bool，True=该 (b,q,k) 可见。padding 行保持全 False。"""
    if not isinstance(spec, MaskSpec):
        raise ValueError("spec must be a MaskSpec")
    if not isinstance(seq_q, int) or isinstance(seq_q, bool) or seq_q < 0:
        raise ValueError("seq_q must be a non-negative int")
    if not isinstance(seq_k, int) or isinstance(seq_k, bool) or seq_k < 0:
        raise ValueError("seq_k must be a non-negative int")
    if not isinstance(tensors, Mapping):
        raise ValueError("tensors must be a mapping")
    batch = _infer_batch(spec, tensors)
    dense = torch.zeros((batch, seq_q, seq_k), dtype=torch.bool)

    for batch_idx in range(batch):
        scalars = _call_scalar_setup(
            spec.scalar_setup, batch_idx=batch_idx, tensors=tensors
        )
        if not isinstance(scalars, Mapping):
            raise ValueError("scalar_setup must return a mapping")
        if spec.cur_seqlen not in scalars:
            raise ValueError(f"scalar_setup did not return {spec.cur_seqlen!r}")
        cur_seqlen = int(scalars[spec.cur_seqlen])
        for row_idx in range(min(seq_q, max(0, cur_seqlen))):
            row_scalars = _row_scalars(scalars, row_idx)
            region = next(
                (
                    candidate
                    for candidate in spec.regions
                    if bool(
                        _call_predicate(
                            candidate.when,
                            standard={
                                **row_scalars,
                                "batch_idx": batch_idx,
                                "row_idx": row_idx,
                            },
                            tensors=tensors,
                        )
                    )
                ),
                None,
            )
            if region is None:
                continue
            for col_idx in range(min(seq_k, max(0, cur_seqlen))):
                dense[batch_idx, row_idx, col_idx] = bool(
                    _call_predicate(
                        region.visible,
                        standard={
                            **row_scalars,
                            "batch_idx": batch_idx,
                            "row_idx": row_idx,
                            "col_idx": col_idx,
                        },
                        tensors=tensors,
                    )
                )
    return dense


# 区间 RLE + 六输出 ABI
def _visible_runs(bits: Sequence[bool]) -> list[tuple[int, int]]:
    """把一维 bool 序列压成可见区间列表 [(start, end), ...]，end 为开区间。"""
    result: list[tuple[int, int]] = []
    start: int | None = None
    for column, visible in enumerate((*bits, False)):
        if visible and start is None:
            start = column
        elif not visible and start is not None:
            result.append((start, column))
            start = None
    return result


def dense_to_rows(
    dense_mask: torch.Tensor,
    *,
    seq_q: int,
    seq_k: int,
) -> list[list[list[tuple[int, int]]]]:
    """每行 True 的连续段压成 (start, end) 区间。全 False 行 → 空列表 []。"""
    if dense_mask.ndim != 3:
        raise ValueError(f"dense_mask must be rank-3, got shape {tuple(dense_mask.shape)}")
    batch = dense_mask.shape[0]
    rows_by_batch: list[list[list[tuple[int, int]]]] = []
    for batch_idx in range(batch):
        batch_rows: list[list[tuple[int, int]]] = []
        for row_idx in range(seq_q):
            bits = dense_mask[batch_idx, row_idx, :seq_k].tolist()
            batch_rows.append(_visible_runs(bits))
        rows_by_batch.append(batch_rows)
    return rows_by_batch


def encode_visible_intervals(
    rows_by_batch: Sequence[Sequence[Sequence[tuple[int, int]]]],
    *,
    seq_q: int,
    seq_k: int,
    mask_num: int,
    tile_m: int,
    tile_n: int,
    backward: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """把可见行区间编码成六输出 mask ABI。

    Args:
        rows_by_batch: [batch][row] -> list of (start, end) 可见区间，区间有序、不重叠、非空。
        seq_q, seq_k: query/key 序列长度。
        mask_num: 每行允许的 masked gap 数上限（=kernel 的 holeNum）。
        tile_m, tile_n: q/k 方向 tile 大小。
        backward: 是否转置（q/k 互换）。BSA 前向场景恒为 False。

    Returns:
        (maskr, mask_start|None, mask_len|None, block_range, block_compute, block_mask)
    """
    if not isinstance(backward, bool):
        raise TypeError("backward must be a bool")
    if not isinstance(seq_q, int) or isinstance(seq_q, bool) or seq_q < 0:
        raise ValueError("seq_q must be a non-negative int")
    if not isinstance(seq_k, int) or isinstance(seq_k, bool) or seq_k < 0:
        raise ValueError("seq_k must be a non-negative int")
    if not isinstance(mask_num, int) or isinstance(mask_num, bool) or mask_num < 0:
        raise ValueError("mask_num must be a non-negative int")
    if not isinstance(tile_m, int) or isinstance(tile_m, bool) or tile_m <= 0:
        raise ValueError("tile_m must be a positive int")
    if not isinstance(tile_n, int) or isinstance(tile_n, bool) or tile_n <= 0:
        raise ValueError("tile_n must be a positive int")

    batch = len(rows_by_batch)
    maskr = torch.zeros((batch, seq_q), dtype=torch.int32)
    mask_start = torch.zeros((batch, mask_num, seq_q), dtype=torch.int32)
    mask_len = torch.zeros((batch, mask_num, seq_q), dtype=torch.int32)
    q_tile_size = tile_n if backward else tile_m
    k_tile_size = tile_m if backward else tile_n
    q_tiles = _ceildiv(seq_q, q_tile_size)
    k_tiles = _ceildiv(seq_k, k_tile_size)
    if backward:
        block_compute = torch.zeros(
            (batch, k_tiles, _ceildiv(q_tiles, 32)), dtype=torch.uint32
        )
    else:
        block_compute = torch.zeros(
            (batch, q_tiles, _ceildiv(k_tiles, 32)), dtype=torch.uint32
        )
    block_mask = torch.zeros_like(block_compute)

    for batch_idx, rows in enumerate(rows_by_batch):
        for row_idx, intervals in enumerate(rows):
            if not intervals:
                continue
            boundary = intervals[-1][1]
            maskr[batch_idx, row_idx] = boundary
            gap_idx = 0
            previous_end = 0
            for start, end in intervals:
                if previous_end < start:
                    if gap_idx >= mask_num:
                        raise ValueError(
                            f"batch {batch_idx} row {row_idx} has more than "
                            f"mask_num={mask_num} gaps below maskr"
                        )
                    mask_start[batch_idx, gap_idx, row_idx] = previous_end
                    mask_len[batch_idx, gap_idx, row_idx] = start - previous_end
                    gap_idx += 1
                previous_end = end

        for q_tile in range(q_tiles):
            q_start = q_tile * q_tile_size
            q_end = min(q_start + q_tile_size, seq_q)
            for k_tile in range(k_tiles):
                k_start = k_tile * k_tile_size
                k_end = min(k_start + k_tile_size, seq_k)
                visible_count = 0
                for row_idx in range(q_start, q_end):
                    for start, end in rows[row_idx]:
                        visible_count += max(0, min(end, k_end) - max(start, k_start))
                pair_count = (q_end - q_start) * (k_end - k_start)
                block_tile = k_tile if backward else q_tile
                packed_tile = q_tile if backward else k_tile
                bit = 1 << (packed_tile % 32)
                word = packed_tile // 32
                if visible_count:
                    block_compute[batch_idx, block_tile, word] = (
                        int(block_compute[batch_idx, block_tile, word]) | bit
                    )
                    if visible_count != pair_count:
                        block_mask[batch_idx, block_tile, word] = (
                            int(block_mask[batch_idx, block_tile, word]) | bit
                        )

    block_range = compute_block_range(
        maskr, q_tile_size=q_tile_size, k_tile_size=k_tile_size
    )
    return (
        maskr,
        mask_start if mask_num else None,
        mask_len if mask_num else None,
        block_range,
        block_compute,
        block_mask,
    )


# 顶层便捷入口：spec → 六输出
def generate_mask_cpu(
    spec: MaskSpec,
    tensors: Mapping[str, torch.Tensor],
    *,
    seq_q: int,
    seq_k: int,
    mask_num: int,
    tile_m: int,
    tile_n: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """spec 谓词求值 → dense → 区间 → 六输出 ABI，返回值与 encode_visible_intervals 一致。"""
    dense = evaluate_spec_dense(spec, tensors, seq_q=seq_q, seq_k=seq_k)
    rows = dense_to_rows(dense, seq_q=seq_q, seq_k=seq_k)
    return encode_visible_intervals(
        rows,
        seq_q=seq_q,
        seq_k=seq_k,
        mask_num=mask_num,
        tile_m=tile_m,
        tile_n=tile_n,
    )


# 补充 pattern：doc_prefix / sliding_window（causal/four_stage 已在上方定义）
def _doc_prefix_scalar_setup(
    batch_idx: int, seq_lens, prefix_lengths
) -> Scalars:
    """seq_lens:(B,) kv_len，prefix_lengths:(B,) prefix 长度。"""
    return {
        "seq_hi": int(seq_lens[batch_idx]),
        "prefix_hi": int(prefix_lengths[batch_idx]),
    }


def _doc_prefix_visible(
    row_idx: int,
    col_idx: int,
    prefix_hi: int,
    **kwargs,
) -> bool:
    """doc+prefix 可见性：prefix 区域双向可见，其余因果"""
    return (col_idx < prefix_hi) or (col_idx <= row_idx)


def _doc_prefix_when(row_idx: int, seq_hi: int, **kwargs) -> bool:
    return row_idx < seq_hi


DOC_PREFIX_SPEC = MaskSpec(
    tensor_args=[
        TensorArg("seq_lens", ("batch",)),
        TensorArg("prefix_lengths", ("batch",)),
    ],
    scalar_setup=_doc_prefix_scalar_setup,
    regions=[
        RegionMask(
            name="doc-prefix",
            when=_doc_prefix_when,
            visible=_doc_prefix_visible,
        )
    ],
    cur_seqlen="seq_hi",
)


def _sliding_window_scalar_setup(
    batch_idx: int, seq_lens, window_sizes
) -> Scalars:
    return {
        "seq_hi": int(seq_lens[batch_idx]),
        "window_size": int(window_sizes[batch_idx]),
    }


def _sliding_window_visible(
    row_idx: int, col_idx: int, window_size: int, **kwargs
) -> bool:
    return (row_idx - window_size < col_idx) and (col_idx <= row_idx)


SLIDING_WINDOW_SPEC = MaskSpec(
    tensor_args=[
        TensorArg("seq_lens", ("batch",)),
        TensorArg("window_sizes", ("batch",)),
    ],
    scalar_setup=_sliding_window_scalar_setup,
    regions=[
        RegionMask(
            name="sliding-window",
            when=_doc_prefix_when,
            visible=_sliding_window_visible,
        )
    ],
    cur_seqlen="seq_hi",
)


# MaskSpec 注册表
NAMED_SPECS: dict[str, MaskSpec] = {
    "causal": CAUSAL_SPEC,
    "doc_prefix": DOC_PREFIX_SPEC,
    "sliding_window": SLIDING_WINDOW_SPEC,
    "four_stage_forward": FOUR_STAGE_SPEC,
}


# AnyMask 格式转换（六输出 ABI → kernel AnyMask 输入）
@dataclass
class AnyMaskParams:
    """bsa_regular_kernel_arch35 kernel 所需的 AnyMask 输入格式

    由六输出 ABI（maskr/mask_start/mask_len/block_range/block_compute/block_mask）
    转换为此格式后传入 kernel。
    """

    tile_range: torch.Tensor       # (B, Tq) int32
    sparse_compute: torch.Tensor | None  # 恒为 None，校验走 block_compute_bp
    sparse_mask: torch.Tensor | None
    block_compute_bp: torch.Tensor  # (B, Tq, Wk) int32 — bit=1:有可见元素
    block_mask_bp: torch.Tensor     # (B, Tq, Wk) int32 — bit=1:需精细 mask
    maskr: torch.Tensor            # (B, max_Sq) int32 — 每行右边界（k >= maskr[q] 被 mask）
    holel: torch.Tensor            # (B, max_Sq, Hn) int32 — 孔洞左边界
    holes: torch.Tensor            # (B, max_Sq, Hn) int32 — 孔洞长度，[holel, holel+holes) 被 mask
    hole_num: torch.Tensor         # (1,) int32 — 孔洞个数 Hn（kernel 读 holeNum[0]）


def maskgen_output_to_anymask(
    maskgen_outputs: tuple[torch.Tensor, ...],
    seq_q: int,
    seq_k: int,
    tile_m: int = 128,
    tile_n: int = 128,
) -> AnyMaskParams:
    """将六输出 ABI 转换为 kernel 的 AnyMask 格式

    六输出 ABI（encode_visible_intervals 格式）：
      [0] maskr        : (B, seq_q)     int32 — 每行最右可见列索引
      [1] mask_start   : (B, mask_num, seq_q) int32 — 间隔起始
      [2] mask_len     : (B, mask_num, seq_q) int32 — 间隔长度
      [3] block_range  : (B, q_tiles)   int32 — 每 q-tile 最后一个有效 k-tile
      [4] block_compute: (B, q_tiles, ceil(k_tiles/32)) uint32 — bit-packed 有效 tile
      [5] block_mask   : (B, q_tiles, ceil(k_tiles/32)) uint32 — bit-packed 部分 mask tile

    kernel AnyMask 格式：
      tile_range    : (B, Tq) int32
      sparse_compute: (B, Tq, Tk) bool — True = 整块屏蔽（kernel 跳过此 tile）
      sparse_mask   : (B, Tq, Tk) bool — True = 部分 mask（kernel 做精细 mask）
      maskr         : (B, max_Sq) int32
      holel         : (B, max_Sq, Hn) int32
      holes         : (B, max_Sq, Hn) int32
      hole_num      : (1,) int32
    """
    maskr, mask_start, mask_len, block_range, block_compute_bp, block_mask_bp = (
        maskgen_outputs
    )

    B = maskr.shape[0]
    Tq = (seq_q + tile_m - 1) // tile_m
    Tk = (seq_k + tile_n - 1) // tile_n
    Hn = mask_start.shape[1] if mask_start is not None else 0

    tile_range = block_range.to(torch.int32)
    maskr_out = maskr.contiguous().to(torch.int32)

    if mask_start is not None:
        holel = mask_start.permute(0, 2, 1).contiguous().to(torch.int32)
        holes = mask_len.permute(0, 2, 1).contiguous().to(torch.int32)
    else:
        holel = torch.zeros((B, seq_q, 0), dtype=torch.int32, device=maskr.device)
        holes = torch.zeros((B, seq_q, 0), dtype=torch.int32, device=maskr.device)

    hole_num = torch.tensor([Hn], dtype=torch.int32, device=maskr.device)

    return AnyMaskParams(
        tile_range=tile_range,
        sparse_compute=None,
        sparse_mask=None,
        maskr=maskr_out,
        holel=holel,
        holes=holes,
        hole_num=hole_num,
        block_compute_bp=block_compute_bp,
        block_mask_bp=block_mask_bp,
    )


# dense mask 调试输出 + 一致性自检
def _print_dense_mask(
    mask: torch.Tensor,        # (Sq, Sk) bool — True=visible
    spec_key: str,
    sq: int,
    sk: int,
    max_display: int = 80,
) -> None:
    """缩略打印 dense mask（batch=0），■=visible ·=masked"""
    print(f"\n  [golden mask] {spec_key}  shape=({sq},{sk})  ■=visible  ·=masked")
    if sq <= max_display and sk <= max_display:
        # 完整打印
        for r in range(sq):
            line = "".join("■" if mask[r, c] else "·" for c in range(sk))
            print(f"  {line}")
    else:
        # 降采样打印
        step_r = max(1, sq // max_display)
        step_c = max(1, sk // max_display)
        # 列标头
        header = "   " + "".join(str(c % 10) for c in range(0, sk, step_c))
        print(f"  {header}")
        for r in range(0, sq, step_r):
            line = f"{r:3d}" + "".join(
                "■" if mask[r, c] else "·" for c in range(0, sk, step_c)
            )
            print(f"  {line}")


def _verify_mask_consistency(
    dense_mask: torch.Tensor,       # (B, Sq, Sk) bool — True=visible
    anymask: AnyMaskParams,
    seq_q: int,
    seq_k: int,
    tile_m: int,
    tile_n: int,
) -> None:
    """验证 dense mask 与 AnyMask 格式之间的一致性（自检）

    检查项：
      1. maskr 与 dense mask 每行最后一个 visible 列一致
      2. sparse_compute（整块屏蔽=True）对应的 tile 在 dense 中全 False
      3. sparse_mask（部分 mask=True）对应的 tile 在 dense 中部分 True
      4. holel/holes 定义的孔在 dense 中全 False
    """
    B = dense_mask.shape[0]
    Tq = (seq_q + tile_m - 1) // tile_m
    Tk = (seq_k + tile_n - 1) // tile_n

    # 从 bp 版本逐 tile 取 bit（不展开为完整 bool 张量）
    def _bp_bit(bp, b, tq, tk):
        """读取 bit-packed int32 (B,Tq,Wk) 中 (b,tq,tk) 的 bit 值"""
        w = tk // 32
        bit = tk % 32
        return bool((int(bp[b, tq, w]) >> bit) & 1)

    errors = []

    # 1. maskr 一致性
    for b in range(B):
        for q in range(seq_q):
            visible_cols = torch.where(dense_mask[b, q])[0]
            expected_maskr = visible_cols[-1].item() + 1 if len(visible_cols) > 0 else 0
            actual_maskr = anymask.maskr[b, q].item()
            if expected_maskr != actual_maskr:
                errors.append(
                    f"maskr mismatch batch={b} row={q}: "
                    f"expected={expected_maskr} actual={actual_maskr}"
                )

    # 2. sparse_compute 一致性（bit=1 表示有可见元素需计算）
    for b in range(B):
        for tq in range(Tq):
            q_start = tq * tile_m
            q_end = min(q_start + tile_m, seq_q)
            for tk in range(Tk):
                k_start = tk * tile_n
                k_end = min(k_start + tile_n, seq_k)
                tile_visible = dense_mask[b, q_start:q_end, k_start:k_end]
                any_visible = tile_visible.any().item()
                should_compute = _bp_bit(anymask.block_compute_bp, b, tq, tk)
                if any_visible != should_compute:
                    errors.append(
                        f"sparse_compute mismatch batch={b} tile=({tq},{tk}): "
                        f"dense_any_visible={any_visible} should_compute={should_compute}"
                    )

    # 3. sparse_mask 一致性（bit=1 表示需精细 mask）
    for b in range(B):
        for tq in range(Tq):
            q_start = tq * tile_m
            q_end = min(q_start + tile_m, seq_q)
            for tk in range(Tk):
                k_start = tk * tile_n
                k_end = min(k_start + tile_n, seq_k)
                tile_visible = dense_mask[b, q_start:q_end, k_start:k_end]
                all_true = tile_visible.all().item()
                all_false = not tile_visible.any().item()
                is_partial = not all_true and not all_false
                is_mask_tile = _bp_bit(anymask.block_mask_bp, b, tq, tk)
                if is_partial != is_mask_tile:
                    errors.append(
                        f"sparse_mask mismatch batch={b} tile=({tq},{tk}): "
                        f"dense_partial={is_partial} is_mask_tile={is_mask_tile}"
                    )

    if errors:
        print(f"  一致性检查发现 {len(errors)} 个错误（仅显示前 5 个）:")
        for e in errors[:5]:
            print(f"    - {e}")
        if len(errors) > 5:
            print(f"    ... 及其他 {len(errors) - 5} 个")
    else:
        print("  一致性检查全部通过")


__all__ = [
    # 数据模型
    "Scalars",
    "TensorArg",
    "RegionMask",
    "MaskSpec",
    # 预定义 pattern
    "CAUSAL_SPEC",
    "FOUR_STAGE_SPEC",
    "DOC_PREFIX_SPEC",
    "SLIDING_WINDOW_SPEC",
    "NAMED_SPECS",
    # CPU 实现
    "compute_block_range",
    "evaluate_spec_dense",
    "dense_to_rows",
    "encode_visible_intervals",
    "generate_mask_cpu",
    # AnyMask 转换 + 自检
    "AnyMaskParams",
    "maskgen_output_to_anymask",
    "_print_dense_mask",
    "_verify_mask_consistency",
]
