"""Runtime probe for lazy DSL ``and`` / ``or`` conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass as tla


VALUES = (0.0, 1.0, 2.0, 4.0, -1.0)
EXPECTED = (0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "artifacts" / "runtime-cache"


@tla.kernel
def lazy_conditions_kernel(values: tla.Tensor, out: tla.Tensor) -> None:
    """The division must only execute in the region selected by the prefix."""
    for i in tla.range(0, len(VALUES), 1):
        x = values[i]
        if x != 0.0 and 10.0 / x > 5.0:
            out[i] = 1.0
        else:
            out[i] = 0.0

        if x == 0.0 or 10.0 / x > 5.0:
            out[len(VALUES) + i] = 1.0
        else:
            out[len(VALUES) + i] = 0.0

        j = 0
        while j < 1 and x != 0.0 and 10.0 / x > 5.0:
            j += 1
        out[2 * len(VALUES) + i] = j.to(tla.Float32)

        j = 0
        while j < 1 and (x == 0.0 or 10.0 / x > 5.0):
            j += 1
        out[3 * len(VALUES) + i] = j.to(tla.Float32)


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
        values = torch.tensor(VALUES, dtype=torch.float32, device=device)
        out = torch.full((4 * len(VALUES),), -1.0, dtype=torch.float32, device=device)
        values_t = tla.from_dlpack(values, layout_tag=tla.arch.RowMajor)
        out_t = tla.from_dlpack(out, layout_tag=tla.arch.RowMajor)

        artifact = tla.compile(
            lazy_conditions_kernel,
            values_t,
            out_t,
            arch_scope="aiv.c310",
            target_arch="c310",
            core_type="aiv",
            kernel_mode="aiv",
            cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
            force_recompile=args.force_recompile,
        )
        artifact(values_t, out_t, block=1)
        torch.npu.synchronize()

        actual = [int(value) for value in out.cpu().tolist()]
        expected = list(EXPECTED)
        if actual != expected:
            print(f"lazy_conditions_ok=False expected={expected} actual={actual}")
            return 1
        print(f"lazy_conditions_ok=True output={actual}")
        return 0
    finally:
        tla.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
