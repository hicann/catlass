import os
import sysconfig
import torch
import torch_npu

__all__ = []

def _load_depend_libs():
    PYTHON_PKG_PATH=sysconfig.get_paths()['purelib']
    TORCH_LIB_PATH=os.path.join(PYTHON_PKG_PATH,"torch/lib")
    TORCH_NPU_LIB_PATH=os.path.join(PYTHON_PKG_PATH,"torch_npu/lib")
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ['LD_LIBRARY_PATH']}:{TORCH_LIB_PATH}:{TORCH_NPU_LIB_PATH}"
    
_load_depend_libs()

from torch_catlass._C import *
