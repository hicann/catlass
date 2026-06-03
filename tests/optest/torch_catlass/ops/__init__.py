# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from .basic_matmul import basic_matmul  # example 00
from .matmul_add import matmul_add  # example 03
from .padding_matmul import padding_matmul  # example 04
from .optimized_matmul import optimized_matmul  # example 06
from .basic_matmul_tla import basic_matmul_tla  # example 13
from .optimized_matmul_tla import optimized_matmul_tla  # example 14
from .matmul_bias import matmul_bias  # example 20
from .basic_matmul_preload_zN import basic_matmul_preload_zN  # example 21
from .flash_attention_infer import flash_attention_infer  # example 23
from .matmul_full_loadA import matmul_full_loadA  # example 25
from .matmul_relu import matmul_relu  # example 26
from .matmul_gelu import matmul_gelu  # example 27
from .matmul_silu import matmul_silu  # example 28
from .a2_fp8_e4m3_matmul import a2_fp8_e4m3_matmul  # example 29
from .w8a16_matmul import w8a16_matmul  # example 30
from .w4a8_matmul import w4a8_matmul  # example 32
from .small_matmul import small_matmul  # example 31
from .single_core_splitk_matmul import single_core_splitk_matmul  # example 34
from .streamk_matmul import streamk_matmul  # example 37
from .big_matmul_tla import big_matmul_tla  # example 39
from .sparse_matmul_tla import sparse_matmul_tla  # example 41
from .quant_optimized_matmul_tla import quant_optimized_matmul_tla  # example 42
from .ascend950_basic_matmul import ascend950_basic_matmul  # example 43
from .quant_matmul_full_loadA_tla import quant_matmul_full_loadA_tla  # example 44
from .strided_batched_matmul_tla import strided_batched_matmul_tla  # example 45
from .quant_multi_core_splitk_matmul_tla import quant_multi_core_splitk_matmul_tla  # example 52
from .ascend950_mx_matmul import ascend950_fp8_mx_matmul_aswt, ascend950_fp4_mx_matmul_aswt  # example 53, 54

__all__ = [
    "basic_matmul",                       # example 00
    "matmul_add",                         # example 03
    "padding_matmul",                     # example 04
    "optimized_matmul",                   # example 06
    "basic_matmul_tla",                   # example 13
    "optimized_matmul_tla",               # example 14
    "matmul_bias",                        # example 20
    "basic_matmul_preload_zN",            # example 21
    "flash_attention_infer",              # example 23
    "matmul_full_loadA",                  # example 25
    "matmul_relu",                        # example 26
    "matmul_gelu",                        # example 27
    "matmul_silu",                        # example 28
    "a2_fp8_e4m3_matmul",                # example 29
    "w8a16_matmul",                       # example 30
    "w4a8_matmul",                        # example 32
    "small_matmul",                       # example 31
    "single_core_splitk_matmul",          # example 34
    "streamk_matmul",                     # example 37
    "big_matmul_tla",                     # example 39
    "sparse_matmul_tla",                  # example 41
    "quant_optimized_matmul_tla",         # example 42
    "ascend950_basic_matmul",             # example 43
    "quant_matmul_full_loadA_tla",        # example 44
    "strided_batched_matmul_tla",         # example 45
    "quant_multi_core_splitk_matmul_tla", # example 52
    "ascend950_fp8_mx_matmul_aswt",       # example 53
    "ascend950_fp4_mx_matmul_aswt",       # example 54
]
