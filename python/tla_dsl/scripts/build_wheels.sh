#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${repo_root}/.artifacts/cache}"
export HATCH_DATA_DIR="${HATCH_DATA_DIR:-${repo_root}/.artifacts/hatch/data}"
export HATCH_CONFIG="${HATCH_CONFIG:-${repo_root}/.artifacts/hatch/config.toml}"
mkdir -p "${XDG_CACHE_HOME}" "${HATCH_DATA_DIR}" "$(dirname "${HATCH_CONFIG}")"
touch "${HATCH_CONFIG}"

shopt -s nullglob
type_extensions=(csrc/mlir/build/python/catlass/_tla_type_bridge_native*.so)
if (( ${#type_extensions[@]} == 0 )); then
  cat <<'EOF'
error: missing Python compiler bridge extensions

Build the MLIR compiler first (recommended via ./build.sh in this directory), then rerun:
  ./scripts/build_wheels.sh
EOF
  exit 1
fi

hatch build -t wheel
