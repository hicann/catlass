# AscendNPU-IR

TLA DSL 基于 AscendNPU-IR 的 Dialect 构建 TLA Dialect，并在降级到最终表示时依赖 AscendNPU-IR 的头文件和静态库。CANN 内置组件不能满足这些构建依赖，因此需要构建仓库中记录的 AscendNPU-IR 子模块。

## 依赖与版本

- 构建依赖 Clang/Clang++ `>=10`、CMake 和 Ninja；如环境允许，建议使用 Clang 19。
- 当前构建选项适用于 `feature/regbase` 分支的 `a07821269ede7a5e683ac02c8a2d291608083741` 提交。
- 使用其他分支或提交时，可能需要调整构建选项；即使 AscendNPU-IR 能够成功构建，也不代表能与当前版本的 TLA DSL 兼容。

## 获取源码

在 CATLASS 仓库根目录执行。AscendNPU-IR 依赖 LLVM 子模块，因此需要递归初始化：

```bash
git submodule sync --recursive
git submodule update --init --recursive python/tla_dsl/3rdparty/AscendNPU-IR
```

## 构建

```bash
cd "${CATLASS_ROOT}/python/tla_dsl/3rdparty/AscendNPU-IR"
./build-tools/build.sh \
  --c-compiler /path/to/clang \
  --cxx-compiler /path/to/clang++ \
  '--add-cmake-options=-DCMAKE_SYSROOT=/' \
  '--add-cmake-options=-DLLVM_ENABLE_ZSTD=OFF' \
  '--add-cmake-options=-DLLVM_ENABLE_RTTI=ON' \
  --build-type Release \
  -j 128 \
  --enable-assertion \
  --disable-werror \
  --disable-mlir-werror \
  --disable-bishengir-werror \
  --build-triton \
  --enable-lld \
  --build ./build \
  --apply-patches \
  --python-binding
```

- 根据 Clang 的部署情况，修改 `--c-compiler` 和 `--cxx-compiler` 参数。编译器版本需不低于 10；如环境中同时提供多个版本，优先选择 Clang 19。
- `-j 128` 仅为示例。请按可用 CPU、内存和磁盘 I/O 资源调整并发数。线程数过小会降低构建速度；线程数过大可能占满内存或磁盘 I/O，导致构建失败。
  - 在一台 `AMD EPYC 9575F` 服务器上，使用 128 线程构建的时间约为 150s。
  - 在一台 `鲲鹏910` 服务器上，使用 192 线程构建的时间约为 700s。

## 构建产物

在 AscendNPU-IR 源码根目录执行：

```bash
test -f bishengir/include/bishengir/Dialect/HIVM/IR/HIVM.h
test -f build/tools/bishengir/include/bishengir/Interfaces/BiShengIREnums.h.inc
test -f build/install/lib/cmake/mlir/MLIRConfig.cmake
test -d build/install/python_packages/mlir_core
```
