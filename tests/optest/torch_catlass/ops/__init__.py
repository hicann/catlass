# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from .basic_matmul import basic_matmul  # example 00
from .padding_matmul import padding_matmul  # example 04
from .optimized_matmul import optimized_matmul  # example 06
from .basic_matmul_tla import basic_matmul_tla  # example 13
from .optimized_matmul_tla import optimized_matmul_tla  # example 14
from .basic_matmul_preload_zN import basic_matmul_preload_zN  # example 21
from .matmul_full_loadA import matmul_full_loadA  # example 25
from .small_matmul import small_matmul  # example 31
from .single_core_splitk_matmul import single_core_splitk_matmul  # example 34
from .streamk_matmul import streamk_matmul  # example 37
from .big_matmul_tla import big_matmul_tla  # example 39
from .quant_optimized_matmul_tla import quant_optimized_matmul_tla  # example 42
from .ascend950_mx_matmul import ascend950_fp8_mx_matmul_aswt, ascend950_fp4_mx_matmul_aswt  # example 53, 54

__all__ = [
    "basic_matmul",                       # example 00
    "padding_matmul",                     # example 04
    "optimized_matmul",                   # example 06
    "basic_matmul_tla",                   # example 13
    "optimized_matmul_tla",              # example 14
    "basic_matmul_preload_zN",            # example 21
    "matmul_full_loadA",                  # example 25
    "small_matmul",                       # example 31
    "single_core_splitk_matmul",          # example 34
    "streamk_matmul",                     # example 37
    "big_matmul_tla",                     # example 39
    "quant_optimized_matmul_tla",         # example 42
    "ascend950_fp8_mx_matmul_aswt",       # example 53
    "ascend950_fp4_mx_matmul_aswt",       # example 54
]
