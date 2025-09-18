# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Dict, Literal, Tuple
import ctypes
import torch
import torch_npu


def get_current_stream_ptr() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.npu.current_stream().npu_stream)


BishengDType = Literal[
    "half",
    "bfloat16_t",
    "float",
    "int8_t",
    "uint8_t",
    "int32_t",
    "uint32_t",
    "int64_t",
    "uint64_t",
]


def torch_dtype_to_bisheng_dtype(torch_dtype: torch.dtype) -> BishengDType:
    bisheng_dtype_mapper: Dict[torch.dtype, BishengDType] = {
        torch.float16: "half",
        torch.float: "float",
        torch.bfloat16: "bfloat16_t",
        torch.int8: "int8_t",
        torch.uint8: "uint8_t",
        torch.int32: "int32_t",
        torch.uint32: "uint32_t",
        torch.int64: "int64_t",
        torch.uint64: "uint64_t",
    }
    bisheng_dtype = bisheng_dtype_mapper.get(torch_dtype, None)
    if bisheng_dtype is None:
        raise RuntimeError("no dtype mapped")
    return bisheng_dtype


def swap(*args):
    assert len(args) >= 2
    return (*args[:-2], args[-1], args[-2])


def is_transposed(mat: torch.Tensor) -> bool:
    assert len(mat.shape) >= 2
    return mat.stride(-2) == 1 and mat.stride(-1) == mat.shape[0]
