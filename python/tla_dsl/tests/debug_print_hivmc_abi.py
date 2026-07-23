#!/usr/bin/env python3
"""Prove the static HIVMC ABI of CATLASS DSL's debug-print workspace.

This compiler-object test starts with project-generated non-print AIV MLIR and
creates matched control/print variants.  The print-side ABI addition is only a
final attributed ``i64`` workspace.  HIVMC ELF evidence is compared through
``__CCE_KernelArgSize``, ``.ParamInfo_*`` symbols, and the entry's
``.ascend.meta.<entry>`` section.

Produce the required non-print AIV input from ``python/tla_dsl`` with
``python examples/end_to_end/basic_vadd/dump_ir.py --out-dir <dir>``; this
creates ``<dir>/1_basic_vadd.lowered.mlir``.  Do not use a print kernel as the
input control for this test.

The metadata header and parameter record sizes below are the fixed layout
observed by the static PoC.  They are intentionally not a general ELF parser:
if the toolchain stops exposing this layout, the test must fail rather than
guessing.  This test does not prove device allocation, launch, copyback, or
TLV decode; Task 5 owns that runtime proof.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


METADATA_HEADER_BYTES = 16
PARAMETER_RECORD_BYTES = 40
ENTRY_RE = re.compile(r"func\.func @([^\s(]+)\(([^)]*)\)([^\{]*)attributes \{([^}]*)\} \{")


@dataclass(frozen=True)
class Case:
    name: str
    extra_types: tuple[str, ...]
    print_must_compile: bool = True


CASES = (
    Case("base_3ptr", ()),
    Case("plus_1_i64", ("i64",)),
    Case("plus_4_mixed_scalars", ("i1", "i32", "i64", "i32")),
    Case("plus_8_gm_ptrs", ("memref<1xi8, #hivm.address_space<gm>>",) * 8),
    Case(
        "plus_32_mixed",
        ("memref<1xi8, #hivm.address_space<gm>>", "i1", "i32", "i64") * 8,
    ),
    Case("at_1536_bytes_plus_188_i64", ("i64",) * 188),
    # This is evidence only: current 171 accepts 1544 bytes.  It must never
    # encode a CATLASS parameter-capacity policy or decide pass/fail.
    Case("observed_1544_bytes_plus_189_i64", ("i64",) * 189, False),
)


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )


def entry_match(text: str) -> re.Match[str]:
    entries = [match for match in ENTRY_RE.finditer(text) if "hacc.entry" in match.group(4)]
    if len(entries) != 1:
        raise RuntimeError(f"expected exactly one hacc.entry function, found {len(entries)}")
    return entries[0]


def add_printf_declaration(text: str) -> str:
    declaration = (
        "  func.func private @_mlir_ciface_tla_printf_x_i32(i32, i64) "
        "attributes {hacc.always_inline, "
        "hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>}\n"
    )
    anchor = text.find("} {\n  func.func")
    if anchor < 0:
        raise RuntimeError("cannot locate module body for printf declaration")
    return text[: anchor + 3] + "\n" + declaration + text[anchor + 3 :]


def make_variant(source: str, extra_types: tuple[str, ...], with_print: bool) -> tuple[str, str, int]:
    match = entry_match(source)
    original_args = match.group(2)
    numbers = [int(value) for value in re.findall(r"%arg(\d+)", original_args)]
    next_number = max(numbers, default=-1) + 1
    additions = [f"%arg{next_number + index}: {arg_type}" for index, arg_type in enumerate(extra_types)]
    workspace_number = next_number + len(additions)
    if with_print:
        additions.append(
            f"%arg{workspace_number}: i64 "
            "{hacc.arg_type = #hacc.arg_type<workspace>, tla.debug_print.workspace}"
        )
    args = ", ".join(part for part in (original_args, *additions) if part.strip())
    header = match.group(0).replace(f"({original_args})", f"({args})", 1)
    injection = ""
    if with_print:
        injection = (
            "\n    %debug_print_value = arith.constant 424242 : i32"
            f"\n    call @_mlir_ciface_tla_printf_x_i32(%debug_print_value, %arg{workspace_number}) "
            ": (i32, i64) -> ()"
        )
    variant = source[: match.start()] + header + injection + source[match.end() :]
    if with_print:
        variant = add_printf_declaration(variant)
    return variant, match.group(1), len(numbers)


def kernel_arg_size(path: Path) -> int:
    result = run(["readelf", "-x", "__CCE_KernelArgSize", str(path)])
    if result.returncode != 0:
        raise RuntimeError(
            f"readelf could not read __CCE_KernelArgSize from {path} "
            f"(exit {result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    match = re.search(r"0x[0-9a-fA-F]+\s+([0-9a-fA-F]{8})", result.stdout)
    if match is None:
        raise RuntimeError(f"cannot read __CCE_KernelArgSize from {path}:\n{result.stdout}{result.stderr}")
    return int.from_bytes(bytes.fromhex(match.group(1)), "little")


def parameter_count(path: Path) -> int:
    result = run(["readelf", "-Ws", str(path)])
    if result.returncode != 0:
        raise RuntimeError(
            f"readelf could not list symbols from {path} "
            f"(exit {result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return len(set(re.findall(r"\.ParamInfo_[^\s]+_([0-9]+)", result.stdout)))


def section_bytes(path: Path, section: str) -> bytes:
    result = run(["readelf", "-x", section, str(path)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    words: list[str] = []
    for line in result.stdout.splitlines():
        match = re.match(r"\s*0x[0-9a-fA-F]+\s+((?:[0-9a-fA-F]{8}\s*){1,4})", line)
        if match is not None:
            words.extend(re.findall(r"[0-9a-fA-F]{8}", match.group(1)))
    if not words:
        raise RuntimeError(f"section {section} was empty or unreadable in {path}")
    return bytes.fromhex("".join(words))


def compile_variant(hivmc: Path, source: Path, output: Path, bitcode: Path) -> subprocess.CompletedProcess[str]:
    return run([
        str(hivmc), str(source), "--target=Ascend950PR_9589", "--disable-ffts",
        "--enable-hivm-compile=False", f"--link-aicore-bitcode={bitcode}", "-o", str(output),
    ])


def inspect_case(source: str, case: Case, args: argparse.Namespace) -> dict[str, object]:
    case_dir = args.output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    control_mlir, entry, original_count = make_variant(source, case.extra_types, False)
    print_mlir, _, _ = make_variant(source, case.extra_types, True)
    control_source, print_source = case_dir / "control.mlir", case_dir / "printf.mlir"
    control_object, print_object = case_dir / "control.o", case_dir / "printf.o"
    control_source.write_text(control_mlir)
    print_source.write_text(print_mlir)
    control = compile_variant(args.hivmc, control_source, control_object, args.printf_bitcode)
    candidate = compile_variant(args.hivmc, print_source, print_object, args.printf_bitcode)
    (case_dir / "control.stderr.txt").write_text(control.stderr)
    (case_dir / "printf.stderr.txt").write_text(candidate.stderr)
    record: dict[str, object] = {
        "case": case.name,
        "control_compile_exit_code": control.returncode,
        "printf_compile_exit_code": candidate.returncode,
    }

    if not case.print_must_compile:
        record["observed_toolchain_outcome"] = {
            "control_succeeded": control.returncode == 0,
            "printf_succeeded": candidate.returncode == 0,
            "note": "Informational only; no CATLASS capacity policy is asserted above 1536 bytes.",
        }
        record["passed"] = True
        return record

    if control.returncode != 0 or candidate.returncode != 0:
        record.update({"checks": {"both_compiles_succeeded": False}, "passed": False})
        return record

    control_size, print_size = kernel_arg_size(control_object), kernel_arg_size(print_object)
    control_count, print_count = parameter_count(control_object), parameter_count(print_object)
    control_meta = section_bytes(control_object, f".ascend.meta.{entry}")
    print_meta = section_bytes(print_object, f".ascend.meta.{entry}")
    prefix_end = METADATA_HEADER_BYTES + control_count * PARAMETER_RECORD_BYTES
    checks = {
        "workspace_adds_exactly_8_bytes": print_size == control_size + 8,
        "workspace_adds_exactly_1_parameter_record": print_count == control_count + 1,
        "preceding_parameter_records_unchanged": control_meta[METADATA_HEADER_BYTES:prefix_end]
        == print_meta[METADATA_HEADER_BYTES:prefix_end],
        "control_record_count_matches_signature": control_count == original_count + len(case.extra_types),
        "workspace_is_final_attributed_i64_in_input": re.search(
            r"%arg\d+: i64 \{hacc.arg_type = #hacc.arg_type<workspace>, tla.debug_print.workspace\}",
            print_mlir,
        ) is not None,
    }
    if case.name == "at_1536_bytes_plus_188_i64":
        checks["print_signature_reaches_exactly_1536_bytes"] = print_size == 1536
        checks["parameter_records_reach_192"] = print_count == 192
    record.update({
        "control_argument_bytes": control_size,
        "printf_argument_bytes": print_size,
        "control_parameter_records": control_count,
        "printf_parameter_records": print_count,
        "checks": checks,
        "passed": all(checks.values()),
    })
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mlir", required=True, type=Path, help="project-generated non-print AIV lowered MLIR")
    parser.add_argument("--hivmc", required=True, type=Path)
    parser.add_argument("--printf-bitcode", required=True, type=Path, help="project-built AIV meta_op bitcode containing printf")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.input_mlir, args.hivmc, args.printf_bitcode):
        if not path.is_file():
            raise SystemExit(f"required input is not a file: {path}")
    source = args.input_mlir.read_text()
    if "tla.debug_print.workspace" in source:
        raise SystemExit("--input-mlir must be the project-generated non-print control")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [inspect_case(source, case, args) for case in CASES]
    report = {
        "classification": "abi-matrix-proven" if all(item["passed"] for item in records) else "abi-matrix-incomplete",
        "cases": records,
        "runtime_abi_proven": False,
        "runtime_abi_note": "Task 5 owns device allocation, launch, copyback, and TLV decode proof.",
    }
    report_path = args.output_dir / "abi-matrix-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["classification"] == "abi-matrix-proven" else 1)


if __name__ == "__main__":
    main()
