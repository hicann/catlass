ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG LLVM_VERSION=19
ARG BUILD_JOBS=128
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_TAG_CP=cp312
ARG TARGET_ARCH=aarch64
ARG TORCH_VERSION=2.9.0
ARG APT_MIRROR
ARG APT_PORTS_MIRROR
ARG APT_LLVM_MIRROR
ARG PIP_MIRROR
ARG TORCH_WHEEL_MIRROR

# APT mirror: ${APT_MIRROR} for x86_64, ${APT_PORTS_MIRROR} otherwise
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

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      gnupg \
      patch && \
    install -d -m 0755 /etc/apt/keyrings

# clang/LLVM (APT_LLVM_MIRROR overrides apt.llvm.org)
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
      "llvm-${LLVM_VERSION}-tools" \
      "lld-${LLVM_VERSION}"

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN if [ -n "${PIP_MIRROR}" ]; then \
      pip3 config set global.index-url "${PIP_MIRROR}"; \
    fi

# lit is only published up to 18.1.8 on PyPI, so install it from LLVM source in the build layer
COPY requirements.txt /tmp/requirements.txt
RUN grep -viE '^lit([=<>!~ ]|$)' /tmp/requirements.txt > /tmp/requirements-nolit.txt && \
    pip3 install --no-cache-dir -r /tmp/requirements-nolit.txt

# torch CPU wheel, pinned by TORCH_VERSION
RUN pip3 install --no-cache-dir \
      "https://${TORCH_WHEEL_MIRROR:-download.pytorch.org}/pytorch/whl/cpu/torch-${TORCH_VERSION}%2Bcpu-${PYTHON_TAG_CP}-${PYTHON_TAG_CP}-manylinux_2_28_${TARGET_ARCH}.whl"

# match TORCH_VERSION's base version, newest postN; bare torch_npu would pull a CUDA torch
RUN pip3 install --no-cache-dir "torch_npu==${TORCH_VERSION}.*"

# clone + patch + lit + build + prune in ONE layer (a separate prune layer frees nothing).
# Keeps only build/install and third-party/llvm-project/mlir, needed by csrc/mlir/python.
RUN --mount=type=bind,source=tools/patches,target=/opt/catlass-patches,ro \
  git clone --branch stable https://gitcode.com/Ascend/AscendNPU-IR.git /opt/AscendNPU-IR && \
  cd /opt/AscendNPU-IR && \
  git reset --hard 693abf31f3d88d13d30d9578fcf6513b90c0ee33 && \
  git submodule update --init && \
  CATLASS_DSL_ASCENDNPU_IR_ROOT=/opt/AscendNPU-IR bash /opt/catlass-patches/apply_patch.sh && \
  pip3 install --no-cache-dir /opt/AscendNPU-IR/third-party/llvm-project/llvm/utils/lit && \
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
    --bishengir-publish OFF \
    --build ./build && \
  find /opt/AscendNPU-IR -name .git -prune -exec rm -rf '{}' + && \
  find /opt/AscendNPU-IR/build -mindepth 1 -maxdepth 1 ! -name install -exec rm -rf {} + && \
  find /opt/AscendNPU-IR/third-party/llvm-project -mindepth 1 -maxdepth 1 \
    -type d ! -name mlir -exec rm -rf {} + && \
  rm -rf /opt/AscendNPU-IR/third-party/torch-mlir \
         /opt/AscendNPU-IR/third-party/shmem \
         /opt/AscendNPU-IR/bishengir \
         /opt/AscendNPU-IR/build-tools \
         /opt/AscendNPU-IR/docs

ENV CATLASS_DSL_PREBUILT_ASCENDNPU_IR=/opt/AscendNPU-IR
