"""Verify a stdlib ``@dataclass`` instance passed as a kernel argument.

The dataclass fields lower to runtime scalar kernel args (one block arg per
field); the kernel copies each field into an output tensor and the host checks
the stored values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import catlass.tla as tla


STATIC_TILE_M = 128
STATIC_TILE_N = 256
INT_VALUE = 1234
FLOAT_VALUE = 2.5


@dataclass(frozen=True)
class TilingData:
    static_tile_m: tla.Constexpr[int] # compile-time int
    static_tile_n: tla.Constexpr[int]
    tiling_gm_out: tla.Tensor         # tensor
    tiling_int16: tla.Int16           # Int16
    tiling_float: tla.Float32         # Float32


@dataclass(frozen=True)
class Info:
    tile_m: int
    tile_n: int

@tla.jit
def print_info(info: Info):
    tla.print("batch={} size={}", info.tile_m, info.tile_n)

@tla.kernel
def dataclass_arg(
    tiling: TilingData,
    out_int: tla.Tensor,
    out_float: tla.Tensor,
) -> None:
    """Copy the dataclass scalar fields into the output tensors."""
    info = Info(128, 256) # 可以直接在kernel内创建
    with tla.vector():
        out_int[0] = tiling.tiling_int16
        out_float[0] = tiling.tiling_float
        tiling.tiling_gm_out[0] = tiling.tiling_float * tiling.tiling_float
        print_info(info)


def _require_torch_npu(device: int):
    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("torch_npu is required for this example") from exc
    torch.npu.set_device(device)
    return torch


def run(args: argparse.Namespace) -> int:
    torch = _require_torch_npu(args.device)
    device = f"npu:{args.device}"
    out_int = torch.zeros(1, dtype=torch.int16, device=device)
    out_float = torch.zeros(1, dtype=torch.float32, device=device)
    out_tiling_float = torch.zeros(1, dtype=torch.float32, device=device)
    out_int_tensor = tla.from_dlpack(
        out_int.contiguous(), layout_tag=tla.arch.RowMajor
    )
    out_float_tensor = tla.from_dlpack(
        out_float.contiguous(), layout_tag=tla.arch.RowMajor
    )
    tiling_out_float_tensor = tla.from_dlpack(
        out_tiling_float.contiguous(), layout_tag=tla.arch.RowMajor
    )
    tiling = TilingData(
        static_tile_m=STATIC_TILE_M,
        static_tile_n=STATIC_TILE_N,
        tiling_gm_out=tiling_out_float_tensor,
        tiling_int16=tla.Int16(INT_VALUE), # Int16 必须显示用tla.Int16构造
        tiling_float=FLOAT_VALUE,
    )

    artifact = tla.compile(
        dataclass_arg,
        tiling,
        out_int_tensor,
        out_float_tensor,
        options="--npu-arch 3510",
        force_recompile=args.force_recompile,
    )
    artifact(tiling, out_int_tensor, out_float_tensor, block_dim=1)
    torch.npu.synchronize()

    actual_int = int(out_int.cpu().item())
    actual_float = float(out_float.cpu().item())
    actual_float_squre = float(out_tiling_float.cpu().item())
    if actual_int != INT_VALUE or actual_float != FLOAT_VALUE or actual_float_squre != FLOAT_VALUE**2:
        print(
            "dataclass_arg_ok=False "
            f"expected=({INT_VALUE}, {FLOAT_VALUE}, {FLOAT_VALUE**2}) "
            f"actual=({actual_int}, {actual_float}, {actual_float_squre})"
        )
        return 1
    print(
        "dataclass_arg_ok=True "
        f"tiling_int16={actual_int} tiling_float={actual_float}, tiling_float**2={actual_float_squre}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal dataclass-argument kernel."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force-recompile", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
