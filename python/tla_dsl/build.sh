#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${repo_root}/.artifacts/cache}"
export HATCH_DATA_DIR="${HATCH_DATA_DIR:-${repo_root}/.artifacts/hatch/data}"
export HATCH_CONFIG="${HATCH_CONFIG:-${repo_root}/.artifacts/hatch/config.toml}"
strict_packaging="${TLA_DSL_STRICT_PACKAGING:-0}"
mkdir -p "${XDG_CACHE_HOME}" "${HATCH_DATA_DIR}" "$(dirname "${HATCH_CONFIG}")"
touch "${HATCH_CONFIG}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  cat <<'EOF'
error: ASCEND_HOME_PATH is not set.
Export it to your CANN / ascend-toolkit root (see README §2.3), e.g.:
  export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
EOF
  exit 1
fi

ensure_pybind11_headers() {
  local pydeps_dir="${repo_root}/.artifacts/pydeps"
  local existing_pythonpath="${PYTHONPATH:-}"

  export PYTHONPATH="${pydeps_dir}${existing_pythonpath:+:${existing_pythonpath}}"
  if python -c 'import pybind11' >/dev/null 2>&1; then
    return 0
  fi

  echo "==> Installing pybind11 build headers into ${pydeps_dir}"
  python -m pip install --target "${pydeps_dir}" pybind11
}

python_site_packages_is_writable() {
  python - <<'PY'
import os
import site
import sys

paths = []
try:
    paths.extend(site.getsitepackages())
except Exception:
    pass

user_site = site.getusersitepackages()
if user_site:
    paths.append(user_site)

seen = set()
for path in paths:
    if not path or path in seen:
        continue
    seen.add(path)
    if os.path.isdir(path):
        sys.exit(0 if os.access(path, os.W_OK) else 1)

sys.exit(1)
PY
}

find_mlir_include_dir() {
  if [[ -n "${MLIR_TBLGEN_INCLUDE_DIR:-}" && -f "${MLIR_TBLGEN_INCLUDE_DIR}/mlir/IR/OpBase.td" ]]; then
    printf '%s\n' "${MLIR_TBLGEN_INCLUDE_DIR}"
    return 0
  fi

  if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/include/mlir/IR/OpBase.td" ]]; then
    printf '%s\n' "${CONDA_PREFIX}/include"
    return 0
  fi

  if command -v llvm-config >/dev/null 2>&1; then
    local llvm_include_dir
    llvm_include_dir="$(llvm-config --includedir)"
    if [[ -f "${llvm_include_dir}/mlir/IR/OpBase.td" ]]; then
      printf '%s\n' "${llvm_include_dir}"
      return 0
    fi
  fi

  return 1
}

echo "==> Configuring MLIR build (csrc/mlir)"
ensure_pybind11_headers
mlir_include_dir="$(find_mlir_include_dir)" || {
  cat <<'EOF'
error: unable to locate MLIR includes (mlir/IR/OpBase.td).
Set MLIR_TBLGEN_INCLUDE_DIR or CONDA_PREFIX, or ensure llvm-config is on PATH.
EOF
  exit 1
}

cmake_args=(
  -G Ninja
  -S csrc/mlir
  -B csrc/mlir/build
  -DCMAKE_C_COMPILER=$(which clang)
  -DCMAKE_CXX_COMPILER=$(which clang++)
  -DCMAKE_SYSROOT=/
  -DCMAKE_SUPPRESS_REGENERATION=ON
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
  -DMLIR_TBLGEN_INCLUDE_DIR="${mlir_include_dir}"
  -DBISHENGIR_BUILD_TEMPLATE="${BISHENGIR_BUILD_TEMPLATE:-ON}"
)
if [[ -n "${MLIR_DIR:-}" ]]; then
  cmake_args+=(-DMLIR_DIR="${MLIR_DIR}")
elif [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib/cmake/mlir" ]]; then
  cmake_args+=(-DMLIR_DIR="${CONDA_PREFIX}/lib/cmake/mlir")
fi

if [[ -z "${BISHENG_COMPILER_PATH:-}" && -n "${BISHENG_INSTALL_PATH:-}" ]]; then
  export BISHENG_COMPILER_PATH="${BISHENG_INSTALL_PATH}"
fi

if [[ -z "${BISHENG_COMPILER_PATH:-}" ]]; then
  bisheng_path="$(command -v bisheng || true)"
  llvm_link_path="$(command -v llvm-link || true)"
  if [[ -n "${bisheng_path}" && -n "${llvm_link_path}" ]]; then
    bisheng_dir="$(dirname "${bisheng_path}")"
    llvm_link_dir="$(dirname "${llvm_link_path}")"
    if [[ "${bisheng_dir}" == "${llvm_link_dir}" ]]; then
      export BISHENG_COMPILER_PATH="${bisheng_dir}"
    fi
  fi
fi

if [[ "${BISHENGIR_BUILD_TEMPLATE:-ON}" == "ON" ]]; then
  if [[ -z "${BISHENG_COMPILER_PATH:-}" ]]; then
    cat <<'EOF'
error: BISHENG_COMPILER_PATH is required when BISHENGIR_BUILD_TEMPLATE=ON.
Set BISHENG_COMPILER_PATH to the Bisheng toolchain directory containing bisheng and llvm-link.
EOF
    exit 1
  fi
  cmake_args+=(-DBISHENG_COMPILER_PATH="${BISHENG_COMPILER_PATH}")
elif [[ -n "${BISHENG_COMPILER_PATH:-}" ]]; then
  cmake_args+=(-DBISHENG_COMPILER_PATH="${BISHENG_COMPILER_PATH}")
fi

cmake "${cmake_args[@]}"

echo "==> Building Tla compiler targets (ninja tla-compiler)"
ninja -C csrc/mlir/build tla-compiler

shopt -s nullglob
type_extensions=(csrc/mlir/build/python/catlass/_tla_type_bridge_native*.so)
if (( ${#type_extensions[@]} > 0 )); then
  echo "==> Installing catlass (TLA DSL) editable (after Python bridge extensions are available)"
  if python_site_packages_is_writable; then
    if ! python -m pip install -e .; then
      if [[ "${strict_packaging}" == "1" ]]; then
        echo "error: editable install failed and TLA_DSL_STRICT_PACKAGING=1" >&2
        exit 1
      fi
      echo "warning: editable install failed; continuing because strict packaging env is not 1" >&2
    fi
  else
    echo "==> Skipping editable install (site-packages not writable for current user)"
  fi
else
  echo "==> Skipping editable install (missing Python bridge extensions)"
fi

if (( ${#type_extensions[@]} > 0 )); then
  echo "==> Building wheel(s)"
  if ! ./scripts/build_wheels.sh; then
    if [[ "${strict_packaging}" == "1" ]]; then
      echo "error: wheel build failed and TLA_DSL_STRICT_PACKAGING=1" >&2
      exit 1
    fi
    echo "warning: wheel build failed; continuing because strict packaging env is not 1" >&2
  fi
else
  echo "==> Skipping wheel build (missing Python bridge extensions)"
fi

echo "Build complete."
echo "Wheels (if hatch succeeded):"
echo "  ${repo_root}/dist/"
