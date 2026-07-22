"""E2E: GM tensor scalar indexing (``tensor[i]`` / ``tensor[r,c]``).

Layout notes (Phase-1 ``scalar_load`` / ``scalar_store``):
- ``RowMajor`` / ``ColumnMajor``: 2D GM tensors; address is ``i*stride0 + j*stride1``.
- ``RowMajor`` (rank-1): contiguous 1D GM vectors (``from_dlpack(...contiguous())``).

Control-flow patterns in this example:
- Static 1D/2D scalar read + store (no loop)
- Python scalar literal store (``out[i] = 1.1125``)
- ``tla.range`` loop copy
- Dynamic ``if`` selecting read index (index merge, load after branch)
- Dynamic ``if`` selecting scalar *values* (Numeric carried through ``scf.if``)
- ``tla.const_expr`` compile-time forward vs reversed indexing
- Scalar read/store inside ``tla.vector`` / ``tla.vec.func`` (VF)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass as tla

LENGTH = 8
ROW = 2
COLS = 64
SCALAR_COL = 5
MARKER_LEN = 2


@tla.kernel
def scalar_index_static_kernel(
    meta: tla.Tensor,
    meta_row: tla.Tensor,
    markers: tla.Tensor,
) -> None:
    """Static 1D ``RowMajor`` (rank-1) + 2D ``RowMajor`` scalar read/store."""
    elem_1d = meta_row[0]
    elem_2d = meta[ROW, SCALAR_COL]
    markers[0, 0] = elem_1d
    markers[0, 1] = elem_2d


@tla.kernel
def scalar_index_literal_store_kernel(out: tla.Tensor) -> None:
    """Python float/int literals stored via scalar indexing."""
    out[0] = 1.1125
    out[1] = 42


@tla.kernel
def scalar_index_loop_kernel(meta: tla.Tensor, out: tla.Tensor) -> None:
    for i in tla.range(0, LENGTH, 1):
        out[i] = meta[i]


@tla.kernel
def scalar_index_dynamic_if_kernel(meta: tla.Tensor, out: tla.Tensor) -> None:
    for i in tla.range(0, LENGTH, 1):
        read_idx = 0
        if i == 0:
            read_idx = i
        else:
            read_idx = 0
        out[i] = meta[read_idx]


@tla.kernel
def scalar_index_value_through_dynamic_if_kernel(
    out: tla.Tensor,
    meta: tla.Tensor,
    selector: int,
) -> None:
    """Dynamic ``if`` selects among scalar *values* (load on both sides), then store."""
    value = meta[0]
    if selector == 0:
        value = meta[1]
    else:
        value = meta[2]
    out[0] = value


@tla.kernel
def scalar_index_constexpr_if_kernel(
    meta: tla.Tensor,
    out: tla.Tensor,
    reverse: tla.Constexpr[bool],
) -> None:
    if tla.const_expr(reverse):
        for i in tla.range(0, LENGTH, 1):
            out[i] = meta[LENGTH - 1 - i]
    else:
        for i in tla.range(0, LENGTH, 1):
            out[i] = meta[i]


@tla.kernel
def scalar_index_vec_func_kernel(meta: tla.Tensor, out: tla.Tensor) -> None:
    """GM scalar load/store nested in ``tla.vector`` / ``tla.vec.func`` (VF).

    Scalar-only VF bodies are inlined by ``tla-vector-region`` (no ``tla.store``
    to outline a helper); the frontend form still exercises the VF nesting.
    """
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for i in tla.range(0, LENGTH, 1):
                out[i] = meta[i]


def _require_torch_npu(device: int):
    import torch

    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("torch_npu is required for this example") from exc
    torch.npu.set_device(device)
    return torch


def _gm_vector_contiguous(meta_1d) -> tla.Tensor:
    return tla.from_dlpack(meta_1d.contiguous(), layout_tag=tla.arch.RowMajor)


def _run_static_1d_2d(args: argparse.Namespace, torch, device: str) -> int:
    rows, cols = 4, COLS
    base = torch.arange(rows * cols, dtype=torch.float32, device=device).reshape(rows, cols)
    expected_markers = torch.tensor(
        [[base[ROW, 0].item(), base[ROW, SCALAR_COL].item()]],
        dtype=torch.float32,
        device=device,
    )
    markers = torch.full((1, MARKER_LEN), -1.0, dtype=torch.float32, device=device)

    meta = tla.from_dlpack(base, layout_tag=tla.arch.RowMajor)
    meta_row = tla.from_dlpack(base[ROW].contiguous(), layout_tag=tla.arch.RowMajor)
    markers_t = tla.from_dlpack(markers, layout_tag=tla.arch.RowMajor)

    artifact = tla.compile(
        scalar_index_static_kernel,
        meta,
        meta_row,
        markers_t,
        cache_dir=args.cache_dir,
        force_recompile=args.force_recompile,
    )
    artifact(meta, meta_row, markers_t, block=args.block)
    torch.npu.synchronize()
    if not torch.allclose(markers, expected_markers, rtol=0.0, atol=1e-4):
        print(f"static_1d_2d_failed expected={expected_markers.tolist()} actual={markers.tolist()}")
        return 1
    print("static_1d_2d_ok=True")
    return 0


def _run_literal_store(args: argparse.Namespace, torch, device: str) -> int:
    out = torch.full((2,), -1.0, dtype=torch.float32, device=device)
    out_t = _gm_vector_contiguous(out)
    artifact = tla.compile(
        scalar_index_literal_store_kernel,
        out_t,
        cache_dir=args.cache_dir,
        force_recompile=args.force_recompile,
    )
    artifact(out_t, block=args.block)
    torch.npu.synchronize()
    expected = torch.tensor([1.1125, 42.0], dtype=torch.float32, device=device)
    if not torch.allclose(out, expected, rtol=0.0, atol=1e-4):
        print(f"literal_store_failed expected={expected.tolist()} actual={out.tolist()}")
        return 1
    print("literal_store_ok=True")
    return 0


def _run_loop_copy(
    args: argparse.Namespace,
    torch,
    meta_t: tla.Tensor,
    out_t: tla.Tensor,
    out: "torch.Tensor",
) -> int:
    expected = torch.arange(LENGTH, dtype=torch.float32, device=out.device)
    artifact = tla.compile(
        scalar_index_loop_kernel,
        meta_t,
        out_t,
        cache_dir=args.cache_dir,
        force_recompile=args.force_recompile,
    )
    artifact(meta_t, out_t, block=args.block)
    torch.npu.synchronize()
    if not torch.allclose(out, expected, rtol=0.0, atol=1e-4):
        print(f"loop_copy_failed expected={expected.tolist()} actual={out.tolist()}")
        return 1
    print("loop_copy_ok=True")
    return 0


def _run_dynamic_if(
    args: argparse.Namespace,
    torch,
    meta_t: tla.Tensor,
    out_t: tla.Tensor,
    out: "torch.Tensor",
) -> int:
    base = torch.arange(LENGTH, dtype=torch.float32, device=out.device)
    expected = torch.full((LENGTH,), base[0].item(), dtype=torch.float32, device=out.device)
    expected[0] = base[0]
    artifact = tla.compile(
        scalar_index_dynamic_if_kernel,
        meta_t,
        out_t,
        cache_dir=args.cache_dir,
        force_recompile=args.force_recompile,
    )
    artifact(meta_t, out_t, block=args.block)
    torch.npu.synchronize()
    if not torch.allclose(out, expected, rtol=0.0, atol=1e-4):
        print(f"dynamic_if_failed expected={expected.tolist()} actual={out.tolist()}")
        return 1
    print("dynamic_if_ok=True")
    return 0


def _run_value_through_dynamic_if(
    args: argparse.Namespace,
    torch,
    device: str,
) -> int:
    """Numeric value selected by a dynamic host/runtime ``selector`` int."""
    meta = torch.arange(LENGTH, dtype=torch.float32, device=device)
    out = torch.full((1,), -1.0, dtype=torch.float32, device=device)
    meta_t = _gm_vector_contiguous(meta)
    out_t = _gm_vector_contiguous(out)

    for selector, expected_val in ((0, float(meta[1].item())), (1, float(meta[2].item()))):
        out.fill_(-1.0)
        artifact = tla.compile(
            scalar_index_value_through_dynamic_if_kernel,
            type_args=(out_t, meta_t, selector),
            cache_dir=args.cache_dir,
            force_recompile=args.force_recompile,
        )
        artifact(out_t, meta_t, selector, block=args.block)
        torch.npu.synchronize()
        actual = float(out[0].item())
        if abs(actual - expected_val) > 1e-4:
            print(
                f"value_through_dynamic_if_failed selector={selector} "
                f"expected={expected_val} actual={actual}"
            )
            return 1
        print(f"value_through_dynamic_if_ok=True selector={selector} value={actual}")
    return 0


def _run_constexpr_if(
    args: argparse.Namespace,
    torch,
    meta_t: tla.Tensor,
    out_t: tla.Tensor,
    out: "torch.Tensor",
) -> int:
    base = torch.arange(LENGTH, dtype=torch.float32, device=out.device)
    for reverse in (False, True):
        expected = base.flip(0) if reverse else base
        artifact = tla.compile(
            scalar_index_constexpr_if_kernel,
            type_args=(meta_t, out_t, reverse),
            cache_dir=args.cache_dir,
            force_recompile=args.force_recompile,
        )
        artifact(meta_t, out_t, block=args.block)
        torch.npu.synchronize()
        if not torch.allclose(out, expected, rtol=0.0, atol=1e-4):
            print(
                f"constexpr_if_failed reverse={reverse} "
                f"expected={expected.tolist()} actual={out.tolist()}"
            )
            return 1
        print(f"constexpr_if_ok=True reverse={reverse}")
    return 0


def _run_vec_func(
    args: argparse.Namespace,
    torch,
    meta_t: tla.Tensor,
    out_t: tla.Tensor,
    out: "torch.Tensor",
) -> int:
    expected = torch.arange(LENGTH, dtype=torch.float32, device=out.device)
    artifact = tla.compile(
        scalar_index_vec_func_kernel,
        meta_t,
        out_t,
        cache_dir=args.cache_dir,
        force_recompile=args.force_recompile,
    )
    artifact(meta_t, out_t, block=args.block)
    torch.npu.synchronize()
    if not torch.allclose(out, expected, rtol=0.0, atol=1e-4):
        print(f"vec_func_failed expected={expected.tolist()} actual={out.tolist()}")
        return 1
    print("vec_func_ok=True")
    return 0


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    torch = _require_torch_npu(args.device)
    try:
        device = f"npu:{args.device}"
        meta = torch.arange(LENGTH, dtype=torch.float32, device=device)
        out = torch.full((LENGTH,), -1.0, dtype=torch.float32, device=device)
        meta_t = _gm_vector_contiguous(meta)
        out_t = _gm_vector_contiguous(out)

        runners = (
            lambda: _run_static_1d_2d(args, torch, device),
            lambda: _run_literal_store(args, torch, device),
            lambda: _run_loop_copy(args, torch, meta_t, out_t, out),
            lambda: _run_dynamic_if(args, torch, meta_t, out_t, out),
            lambda: _run_value_through_dynamic_if(args, torch, device),
            lambda: _run_constexpr_if(args, torch, meta_t, out_t, out),
            lambda: _run_vec_func(args, torch, meta_t, out_t, out),
        )
        for runner in runners:
            rc = runner()
            if rc != 0:
                return rc
            out.fill_(-1.0)
        print("verification_ok=True")
        return 0
    finally:
        tla.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GM tensor scalar indexing E2E (1D/2D layouts, control flow, vec.func)."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).resolve().parent / "artifacts" / "runtime-cache"),
    )
    parser.add_argument("--force-recompile", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
