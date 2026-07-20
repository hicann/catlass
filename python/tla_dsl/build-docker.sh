#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"

usage() {
  cat <<'EOF'
Usage: build-docker.sh <base_image> [options]

Arguments:
  <base_image>       Base Docker image, e.g. cann:9.1.0-beta.3-950-ubuntu22.04-py3.12
                     The output image will be named ascend-catlass-dsl:<tag>
                     (where <tag> is the part after ':' in the base image)

Mirror options (default: all empty = use upstream):
  --default-mirror           Set all mirrors to mirror.nju.edu.cn (overridable)
  --apt-mirror <url>         APT mirror for archive.ubuntu.com (x86_64, full URL)
  --apt-ports-mirror <url>   APT mirror for ports.ubuntu.com (non-x86, full URL)
  --apt-llvm-mirror <url>    APT mirror for LLVM (default: apt.llvm.org)
  --pip-mirror <url>         PyPI mirror for pip index
  --torch-wheel-mirror <url> Mirror for PyTorch wheel (default: download.pytorch.org)

Build options:
  --llvm-version <ver>  LLVM version to install (default: 19)
  --build-jobs <num>    Parallel build jobs (default: 192)
  -h, --help            Show this help
EOF
}

# Parse positional argument and options
BASE_IMAGE=""
LLVM_VERSION="19"
BUILD_JOBS="192"
APT_MIRROR=""
APT_PORTS_MIRROR=""
APT_LLVM_MIRROR=""
PIP_MIRROR=""
TORCH_WHEEL_MIRROR=""

if [[ $# -eq 0 ]]; then
  echo "Error: missing <base_image> argument" >&2
  usage >&2
  exit 1
fi

# First positional arg is the base image
BASE_IMAGE="$1"
shift

# Extract tag from base image (part after ':')
IMAGE_TAG_SUFFIX="${BASE_IMAGE#*:}"
if [[ "$IMAGE_TAG_SUFFIX" == "$BASE_IMAGE" ]]; then
  echo "Error: base image must include a tag (e.g. cann:9.1.0-beta.3-950-ubuntu22.04-py3.12)" >&2
  exit 1
fi
OUTPUT_IMAGE="ascend-catlass-dsl:${IMAGE_TAG_SUFFIX}"

# Extract Python version from tag suffix, e.g. "9.1.0-beta.3-950-ubuntu22.04-py3.12" -> "py3.12" -> "cp312"
PYTHON_TAG="${IMAGE_TAG_SUFFIX##*-}"
PYTHON_TAG_CP="cp${PYTHON_TAG#py}"
PYTHON_TAG_CP="${PYTHON_TAG_CP/./}"

# Detect target architecture
TARGET_ARCH="$(uname -m)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --default-mirror)
      APT_MIRROR="https://mirror.nju.edu.cn/ubuntu"
      APT_PORTS_MIRROR="https://mirror.nju.edu.cn/ubuntu-ports"
      APT_LLVM_MIRROR="mirror.nju.edu.cn"
      PIP_MIRROR="mirror.nju.edu.cn"
      TORCH_WHEEL_MIRROR="mirror.nju.edu.cn"
      shift
      ;;
    --apt-mirror)
      APT_MIRROR="${2:-}"
      shift 2
      ;;
    --apt-ports-mirror)
      APT_PORTS_MIRROR="${2:-}"
      shift 2
      ;;
    --apt-llvm-mirror)
      APT_LLVM_MIRROR="${2:-}"
      shift 2
      ;;
    --pip-mirror)
      PIP_MIRROR="${2:-}"
      shift 2
      ;;
    --torch-wheel-mirror)
      TORCH_WHEEL_MIRROR="${2:-}"
      shift 2
      ;;
    --llvm-version)
      LLVM_VERSION="${2:-}"
      shift 2
      ;;
    --build-jobs)
      BUILD_JOBS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo "==> Base image:         $BASE_IMAGE"
echo "==> Output image:       $OUTPUT_IMAGE"
echo "==> Python tag:         $PYTHON_TAG"
echo "==> Python cp:          $PYTHON_TAG_CP"
echo "==> Target arch:        $TARGET_ARCH"
echo "==> APT mirror:         ${APT_MIRROR:-<upstream>}"
echo "==> APT ports mirror:   ${APT_PORTS_MIRROR:-<upstream>}"
echo "==> APT LLVM mirror:    ${APT_LLVM_MIRROR:-<upstream>}"
echo "==> PIP mirror:         ${PIP_MIRROR:-<upstream>}"
echo "==> Torch wheel mirror: ${TORCH_WHEEL_MIRROR:-<upstream>}"
echo "==> LLVM version:       $LLVM_VERSION"
echo "==> Build jobs:         $BUILD_JOBS"
echo ""

# Build args array
build_args=()
build_args+=(--build-arg "BASE_IMAGE=$BASE_IMAGE")
build_args+=(--build-arg "LLVM_VERSION=$LLVM_VERSION")
build_args+=(--build-arg "BUILD_JOBS=$BUILD_JOBS")
build_args+=(--build-arg "PYTHON_TAG_CP=$PYTHON_TAG_CP")
build_args+=(--build-arg "TARGET_ARCH=$TARGET_ARCH")
[[ -n "$APT_MIRROR" ]]         && build_args+=(--build-arg "APT_MIRROR=$APT_MIRROR")
[[ -n "$APT_PORTS_MIRROR" ]]   && build_args+=(--build-arg "APT_PORTS_MIRROR=$APT_PORTS_MIRROR")
[[ -n "$APT_LLVM_MIRROR" ]]    && build_args+=(--build-arg "APT_LLVM_MIRROR=$APT_LLVM_MIRROR")
[[ -n "$PIP_MIRROR" ]]         && build_args+=(--build-arg "PIP_MIRROR=$PIP_MIRROR")
[[ -n "$TORCH_WHEEL_MIRROR" ]] && build_args+=(--build-arg "TORCH_WHEEL_MIRROR=$TORCH_WHEEL_MIRROR")

docker build \
  "${build_args[@]}" \
  -f "$DOCKERFILE" \
  -t "$OUTPUT_IMAGE" \
  "$SCRIPT_DIR"

echo ""
echo "==> Build complete: $OUTPUT_IMAGE"