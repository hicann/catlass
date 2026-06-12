#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
cd "$repo_root"
export PYTHONDONTWRITEBYTECODE=1

out_dir="${BASIC_VADD_ARTIFACT_DIR:-$script_dir/artifacts}"
python -B examples/end_to_end/basic_vadd/dump_ir.py --out-dir "$out_dir"

tla_compile="${TLA_COMPILE:-}"
if [[ -z "$tla_compile" ]]; then
  for candidate in \
    csrc/mlir/build/tools/tla-compile/TlaCompile \
    csrc/mlir/build/TlaCompile; do
    if [[ -x "$candidate" ]]; then
      tla_compile="$candidate"
      break
    fi
  done
fi
if [[ -z "$tla_compile" ]]; then
  echo "error: TlaCompile not found; build tla-compiler first" >&2
  exit 1
fi

tlair_out="$out_dir/0_basic_vadd.tlair.mlir"
lowered_out="$out_dir/1_basic_vadd.lowered.mlir"
trace_out="$out_dir/3_basic_vadd.intermediate_trace.txt"
tmp_tlair="$(mktemp "$out_dir/0_basic_vadd.tlair.XXXXXX.mlir")"
"$tla_compile" --emit=tlair "$tlair_out" -o "$tmp_tlair"
mv "$tmp_tlair" "$tlair_out"
"$tla_compile" \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all \
  "$tlair_out" \
  -o "$lowered_out" \
  2> "$trace_out"
