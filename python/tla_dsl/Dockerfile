ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG LLVM_VERSION=19
ARG BUILD_JOBS=128
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_TAG_CP=cp312
ARG TARGET_ARCH=aarch64
ARG APT_MIRROR
ARG APT_PORTS_MIRROR
ARG APT_LLVM_MIRROR
ARG PIP_MIRROR
ARG TORCH_WHEEL_MIRROR

# Layer 1: apt mirror replacement (conditional on arch, mirror is the full URL)
#   x86_64       -> archive.ubuntu.com/ubuntu  -> ${APT_MIRROR}
#   non-x86_64   -> ports.ubuntu.com/ubuntu-ports -> ${APT_PORTS_MIRROR}
RUN if [ "${TARGET_ARCH}" = "x86_64" ] && [ -n "${APT_MIRROR}" ]; then \
      sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|'"${APT_MIRROR}"'|g' \
        -e 's|https://archive.ubuntu.com/ubuntu|'"${APT_MIRROR}"'|g' \
        /etc/apt/sources.list; \
    elif [ "${TARGET_ARCH}" != "x86_64" ] && [ -n "${APT_PORTS_MIRROR}" ]; then \
      sed -i \
        -e 's|http://ports.ubuntu.com/ubuntu-ports|'"${APT_PORTS_MIRROR}"'|g' \
        -e 's|https://ports.ubuntu.com/ubuntu-ports|'"${APT_PORTS_MIRROR}"'|g' \
        /etc/apt/sources.list; \
    fi

# Layer 2: install base packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      gnupg && \
    install -d -m 0755 /etc/apt/keyrings

# Layer 3: LLVM repo + clang (APT_LLVM_MIRROR overrides apt.llvm.org)
RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | \
      gpg --batch --yes --dearmor \
      -o /etc/apt/keyrings/llvm-snapshot.gpg && \
    printf '%s\n' \
      "deb [signed-by=/etc/apt/keyrings/llvm-snapshot.gpg] https://${APT_LLVM_MIRROR:-apt.llvm.org}/llvm-apt/jammy/ llvm-toolchain-jammy-${LLVM_VERSION} main" \
      > /etc/apt/sources.list.d/llvm-apt.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      "clang-${LLVM_VERSION}" \
      "clang-tools-${LLVM_VERSION}" \
      "lld-${LLVM_VERSION}"

# Layer 4: clean apt cache
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Layer 5: pip index config (only if PIP_MIRROR is set)
RUN if [ -n "${PIP_MIRROR}" ]; then \
      pip3 config set global.index-url https://${PIP_MIRROR}/pypi/web/simple; \
    fi

# Layer 6: install Python packages
RUN pip3 install --no-cache-dir \
      "https://${TORCH_WHEEL_MIRROR:-download.pytorch.org}/pytorch/whl/cpu/torch-2.9.0%2Bcpu-${PYTHON_TAG_CP}-${PYTHON_TAG_CP}-manylinux_2_28_${TARGET_ARCH}.whl" \
      "https://gitcode.com/Ascend/pytorch/releases/download/v26.0.1-pytorch2.9.0/torch_npu-2.9.0.post4-${PYTHON_TAG_CP}-${PYTHON_TAG_CP}-manylinux_2_28_${TARGET_ARCH}.whl" \
      'pybind11==2.13.6' \
      'cmake>=3.28,>4' \
      'ninja>=1.12,<2' \
      pytest hatch hatchling hatch-vcs ruff

RUN git clone --branch feature/regbase https://gitcode.com/Ascend/AscendNPU-IR.git /opt/AscendNPU-IR && \
  cd /opt/AscendNPU-IR && \
  git reset --hard a07821269ede7a5e683ac02c8a2d291608083741 && \
  git submodule update --init

RUN cd /opt/AscendNPU-IR && \
  ./build-tools/build.sh \
  --c-compiler $(which clang-${LLVM_VERSION}) \
  --cxx-compiler $(which clang++-${LLVM_VERSION}) \
  '--add-cmake-options=-DCMAKE_SYSROOT=/' \
  '--add-cmake-options=-DLLVM_ENABLE_ZSTD=OFF' \
  '--add-cmake-options=-DLLVM_ENABLE_RTTI=ON' \
  --build-type Release \
  -j ${BUILD_JOBS} \
  --enable-assertion \
  --disable-werror \
  --disable-mlir-werror \
  --disable-bishengir-werror \
  --build-triton \
  --enable-lld \
  --build ./build \
  --apply-patches \
  --python-binding

RUN find /opt/AscendNPU-IR \
  -name .git \
  -prune \
  -exec rm -rf '{}' +

ENV TLA_DSL_PREBUILT_ASCENDNPU_IR=/opt/AscendNPU-IR
ENV MLIR_TBLGEN_INCLUDE_DIR="$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/install/include"
ENV PYTHONPATH="$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/install/python_packages/mlir_core:${PYTHONPATH:-}"
ENV BISHENG_COMPILER_PATH=${ASCEND_HOME_PATH}/bin

WORKDIR /opt