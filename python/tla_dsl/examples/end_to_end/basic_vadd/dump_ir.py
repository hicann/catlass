from __future__ import annotations

import argparse
from pathlib import Path

from catlass.base_dsl import BaseDSL
from catlass.compiler_bridge import lower_tlair_module_to_mlir

from basic_vadd import _compile_only_type_args, basic_vadd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "artifacts"


def _extract_trace_section(trace: str, header: str) -> str:
    lines = trace.splitlines()
    start = None
    for index, line in enumerate(lines):
        if header in line:
            start = index + 1
            break
    if start is None:
        raise RuntimeError(f"missing IR dump section: {header}")

    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("// -----// IR Dump "):
            end = index
            break
    return "\n".join(lines[start:end]).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump basic_vadd MLIR artifacts.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tlair_out = out_dir / "0_basic_vadd.tlair.mlir"
    before_ascend_out = out_dir / "1a_basic_vadd.before_ascendnpu_ir_passes.mlir"
    lowered_out = out_dir / "1_basic_vadd.lowered.mlir"
    trace_out = out_dir / "3_basic_vadd.intermediate_trace.txt"

    lowered = BaseDSL()._lower(
        basic_vadd.fn,
        kind=basic_vadd.kind,
        options=dict(basic_vadd.options),
        type_args=_compile_only_type_args(),
        location=basic_vadd.decorator_location,
    )
    tlair_out.write_text(lowered.asm())
    bridge_result = lower_tlair_module_to_mlir(
        lowered.module,
        mlir_print_ir_before_all=True,
        mlir_print_ir_after_all=True,
    )
    lowered_out.write_text(bridge_result.lowered_mlir)
    trace_out.write_text(bridge_result.pass_ir_dump)
    before_ascend_out.write_text(
        _extract_trace_section(
            bridge_result.pass_ir_dump,
            "IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation)",
        )
    )

    print(f"Wrote DSL TLA MLIR: {tlair_out}")
    print(f"Wrote MLIR before AscendNPU-IR passes: {before_ascend_out}")
    print(f"Wrote lowered MLIR: {lowered_out}")
    print(f"Wrote intermediate IR trace: {trace_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
