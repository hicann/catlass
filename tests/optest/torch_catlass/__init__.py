import os
import re
import ctypes

import torch
import torch_npu

__all__ = ["ops", "basic_matmul"]

_catlass_loaded: bool = False


def get_npu_arch():
    device_name = torch_npu.npu.get_device_name()
    if re.match(r"Ascend910B.+", device_name, re.I) or re.search(
        r"Ascend910_93", device_name, re.I
    ):
        return 2201
    elif re.search("Ascend950(PR|DT)", device_name, re.I):
        return 3510
    else:
        raise RuntimeError(f"Unsupported device name: {device_name}")


def _load_kernel_libs():
    global _catlass_loaded
    if not _catlass_loaded:
        arch = get_npu_arch()
        lib_dir = os.path.join(os.path.dirname(__file__), "lib", str(arch))

        if not os.path.exists(lib_dir):
            raise RuntimeError(f"Library directory not found: {lib_dir}")

        for lib_file in os.listdir(lib_dir):
            if lib_file.endswith(".so"):
                lib_path = os.path.join(lib_dir, lib_file)
                _dl_mode = getattr(os, "RTLD_NOW", 0x2) | getattr(
                    os, "RTLD_GLOBAL", 0x100
                )
                ctypes.CDLL(lib_path, mode=_dl_mode)

        _catlass_loaded = True


def _load_main_lib():
    torch.ops.load_library(
        os.path.join(os.path.dirname(__file__), "lib", "libcatlass_torch.so")
    )


_load_kernel_libs()
_load_main_lib()

from . import ops
from .ops import *
