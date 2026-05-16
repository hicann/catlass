from __future__ import annotations

import argparse
from pathlib import Path

from catlass.base_dsl import BaseDSL
from catlass.compiler_bridge import lower_tlair_module_to_mlir

from basic_mmad import TYPE_ARGS, basic_mmad

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dump raw TLA MLIR (tla dialect), final lowered MLIR, and the full before/after-all "
            "intermediate IR trace."
        )
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory where MLIR artifacts will be written.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tlair_out = out_dir / "0_basic_mmad.tlair.mlir"
    lowered_out = out_dir / "1_basic_mmad.lowered.mlir"
    trace_out = out_dir / "3_basic_mmad.intermediate_trace.txt"

    lowered = BaseDSL()._lower(
        basic_mmad.fn,
        kind=basic_mmad.kind,
        options=dict(basic_mmad.options),
        type_args=TYPE_ARGS,
        location=basic_mmad.decorator_location,
    )
    tlair_out.write_text(lowered.asm())
    bridge_result = lower_tlair_module_to_mlir(
        lowered.module,
        mlir_print_ir_before_all=True,
        mlir_print_ir_after_all=True,
    )
    lowered_out.write_text(bridge_result.lowered_mlir)
    trace_out.write_text(bridge_result.pass_ir_dump)

    print(f"Wrote DSL TLA MLIR: {tlair_out}")
    print(f"Wrote lowered MLIR: {lowered_out}")
    print(f"Wrote intermediate IR trace: {trace_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
