#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

usage() {
  cat <<'EOF'
Usage: build.sh [--release] [--clean]

Options:
  --release    CMAKE_BUILD_TYPE=Release
  --clean      Remove all build artifacts and rebuild from scratch
  (default)    CMAKE_BUILD_TYPE=Debug and editable installation

Environment variables:
  ASCEND_HOME_PATH                      CANN / ascend-toolkit root (REQUIRED)
  CATLASS_DSL_PREBUILT_ASCENDNPU_IR     Shared AscendNPU-IR source/build root
EOF
}

mode="debug"
do_clean=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      mode="release"
      ;;
    --clean)
      do_clean=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

# ============================================================================
# 0. Clean build artifacts
# ============================================================================

if [[ $do_clean -eq 1 ]]; then
  echo "==> Cleaning build artifacts..."
  rm -rf build/ csrc/mlir/build/ dist/ *.egg-info .pytest_cache/
  rm -rf catlass/_mlir
  echo "Clean complete."
fi

# ============================================================================
# 1. Validate required environment variables
# ============================================================================

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "ERROR: ASCEND_HOME_PATH is not set." >&2
  echo "  Source set_env.sh from your CANN installation, e.g.:" >&2
  echo '    source /usr/local/Ascend/cann/set_env.sh' >&2
  exit 1
fi

# ============================================================================
# 2. Resolve AscendNPU-IR source and install trees
# ============================================================================

submodule_root="${repo_root}/3rdparty/AscendNPU-IR"

npu_ir_source=""
source_candidates=(
  "${CATLASS_DSL_ASCENDNPU_IR_ROOT:-}"
  "${submodule_root}"
  "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR:-}"
)
for candidate in "${source_candidates[@]}"; do
  if [[ -n "${candidate}" && -d "${candidate}/third-party/llvm-project/mlir" ]]; then
    npu_ir_source="$(cd "${candidate}" && pwd -P)"
    break
  fi
done
if [[ -z "${npu_ir_source}" ]]; then
  echo "ERROR: Unable to locate an AscendNPU-IR source tree." >&2
  exit 1
fi

npu_ir_install=""
install_candidates=(
  "${CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR:-}"
  "${submodule_root}/build/install"
)
if [[ -n "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR:-}" ]]; then
  install_candidates+=("${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install")
fi
for candidate in "${install_candidates[@]}"; do
  if [[ -f "${candidate}/lib/cmake/mlir/MLIRConfig.cmake" &&
        -f "${candidate}/lib/cmake/llvm/LLVMConfig.cmake" ]]; then
    npu_ir_install="$(cd "${candidate}" && pwd -P)"
    break
  fi
done

if [[ -z "${npu_ir_install}" ]]; then
  echo "ERROR: Unable to locate an AscendNPU-IR install tree." >&2
  exit 1
fi

export CATLASS_DSL_ASCENDNPU_IR_ROOT="${npu_ir_source}"
export CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR="${npu_ir_install}"

echo "==> ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "==> AscendNPU-IR source=${CATLASS_DSL_ASCENDNPU_IR_ROOT}"
echo "==> AscendNPU-IR install=${CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR}"

# ============================================================================
# 3. Check whether C++ sources are stale (dev mode only; triggers cmake reconfiguration)
# ============================================================================

if [[ "$mode" == "debug" ]]; then
  tla_compile_bin="csrc/mlir/build/tools/tla-compile/TlaCompile"
  if [[ -f "$tla_compile_bin" ]]; then
    stale=$(find csrc/mlir/ -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.td' -o -name '*.inc' \) \
      -newer "$tla_compile_bin" -print -quit)
    if [[ -n "$stale" ]]; then
      echo "==> Source file newer than tla-compile: ${stale#csrc/mlir/}"
      echo "==> Removing cmake build stamp to force reconfigure."
      rm -f csrc/mlir/build/CMakeCache.txt
    fi
  fi
fi

# ============================================================================
# 4. Build

if [[ "$mode" == "release" ]]; then
  export CMAKE_BUILD_TYPE="Release"
  rm -rf build/cmake build/lib.*
  python -m pip wheel --no-deps . -w dist/
  echo "Release build complete."
  echo "Wheels: ${repo_root}/dist/"
else
  export CMAKE_BUILD_TYPE="Debug"
  export CMAKE_BUILD_DIR="csrc/mlir/build"
  python setup.py build_ext --inplace
  (
    cd "$repo_root/.."
    python -I -m pip install --no-build-isolation -e "$repo_root" --no-deps
  )
  echo "Debug build and install complete."
fi
