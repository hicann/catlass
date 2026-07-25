"""Verify a tensor, scalar, tensor kernel launch on Ascend."""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass as tla


SCALAR_VALUE = -1234
TRAILING_VALUE = 2468
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "artifacts" / "runtime-cache"


@tla.kernel
def scalar_arg_alignment(
    output_tensor: tla.Tensor,
    scalar: tla.Int16,
    trailing_tensor: tla.Tensor,
) -> None:
    """Store the scalar and a value loaded through the trailing pointer."""
    with tla.vector():
        output_tensor[0] = scalar
        output_tensor[1] = trailing_tensor[0]


def _require_torch_npu(device: int):
    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("torch_npu is required for this example") from exc
    torch.npu.set_device(device)
    return torch


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        torch = _require_torch_npu(args.device)
        device = f"npu:{args.device}"
        output = torch.zeros(2, dtype=torch.int16, device=device)
        trailing = torch.tensor(
            [TRAILING_VALUE], dtype=torch.int16, device=device
        )
        output_tensor = tla.from_dlpack(
            output.contiguous(), layout_tag=tla.arch.RowMajor
        )
        trailing_tensor = tla.from_dlpack(
            trailing.contiguous(), layout_tag=tla.arch.RowMajor
        )
        scalar = tla.Int16(SCALAR_VALUE)

        artifact = tla.compile(
            scalar_arg_alignment,
            output_tensor,
            scalar,
            trailing_tensor,
            arch_scope="aiv.c310",
            target_arch="c310",
            core_type="aiv",
            kernel_mode="aiv",
            cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
            force_recompile=args.force_recompile,
        )
        artifact(output_tensor, scalar, trailing_tensor, block=1)
        torch.npu.synchronize()

        actual = [int(value) for value in output.cpu().tolist()]
        expected = [SCALAR_VALUE, TRAILING_VALUE]
        if actual != expected:
            print(f"scalar_arg_alignment_ok=False expected={expected} actual={actual}")
            return 1
        print(
            "scalar_arg_alignment_ok=True "
            f"scalar={actual[0]} trailing={actual[1]}"
        )
        return 0
    finally:
        tla.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal tensor-scalar-tensor kernel."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
