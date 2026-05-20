# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import ctypes
import os
import re

import torch
import torch_npu

# Version info — auto-derived from catlass git repo
from ._version import get_version as _get_version

_vers = _get_version()
__version__ = _vers["full"]
__catlass_version__ = _vers["full"]
__catlass_tag__ = _vers["tag"]
__catlass_commit__ = _vers["commit"]
# Expose to JIT compiler at runtime
os.environ["TORCH_CATLASS_VERSION"] = _vers["full"]

__all__ = ["ops", "basic_matmul", "__version__", "__catlass_version__"]

_catlass_loaded: bool = False


def enable_mssanitizer():
    """Enable Ascend memory sanitizer for subsequent JIT compilations."""
    os.environ["MS_SANITIZE_MEMORY"] = "1"


def get_npu_arch():
    """Return the CATLASS architecture id for the active torch-npu device."""
    device_count = torch_npu.npu.device_count()
    if device_count <= 0:
        raise RuntimeError("No Ascend NPU device is available for torch-catlass")

    device_name = torch_npu.npu.get_device_name()
    if re.match(r"Ascend910B.+", device_name, re.I) or re.search(
        r"Ascend910_93", device_name, re.I
    ):
        return 2201
    elif re.search("Ascend950(PR|DT)", device_name, re.I):
        return 3510
    else:
        raise RuntimeError(f"Unsupported device name: {device_name}")


def _load_so_files(lib_dir):
    """Load every shared object in ``lib_dir`` with global symbol visibility."""
    for lib_file in os.listdir(lib_dir):
        if lib_file.endswith(".so"):
            lib_path = os.path.join(lib_dir, lib_file)
            _dl_mode = getattr(os, "RTLD_NOW", 0x2) | getattr(
                os, "RTLD_GLOBAL", 0x100
            )
            ctypes.CDLL(lib_path, mode=_dl_mode)


def _load_kernel_libs():
    """Load JIT and architecture-specific kernel libraries once per process."""
    global _catlass_loaded
    if not _catlass_loaded:
        base = os.path.dirname(__file__)

        # 设置包根路径环境变量（供 JIT 编译器运行时使用）
        os.environ["TORCH_CATLASS_PKG_DIR"] = base

        # 1. JIT 编译器 & JIT kernel 入口 — 先加载编译器，再加载统一入口库
        jit_dir = os.path.join(base, "lib", "jit")
        if os.path.exists(jit_dir):
            mode = getattr(os, "RTLD_NOW", 0x2) | getattr(os, "RTLD_GLOBAL", 0x100)
            compiler = os.path.join(jit_dir, "libcatlass_kernel_jit_compiler.so")
            if os.path.exists(compiler):
                ctypes.CDLL(compiler, mode=mode)
            jit_lib = os.path.join(jit_dir, "libcatlass_kernel_jit.so")
            if os.path.exists(jit_lib):
                ctypes.CDLL(jit_lib, mode=mode)

        # 2. 架构相关内核
        arch = get_npu_arch()
        arch_dir = os.path.join(base, "lib", str(arch))
        if os.path.exists(arch_dir):
            _load_so_files(arch_dir)

        if not os.path.exists(jit_dir) and not os.path.exists(arch_dir):
            raise RuntimeError(
                f"No library directory found: checked {jit_dir} and {arch_dir}"
            )

        _catlass_loaded = True


def _load_main_lib():
    """Load the PyTorch extension that registers ``torch.ops.catlass`` ops."""
    torch.ops.load_library(
        os.path.join(os.path.dirname(__file__), "lib", "libcatlass_torch.so")
    )


_load_kernel_libs()
_load_main_lib()

from . import ops
from .ops import *
