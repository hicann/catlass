# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from .basic_matmul import basic_matmul  # example 00
from .batched_matmul import batched_matmul  # example 01
from .grouped_matmul_slice_m import grouped_matmul_slice_m  # example 02
from .matmul_add import matmul_add  # example 03
from .padding_matmul import padding_matmul  # example 04
from .grouped_matmul_slice_k import grouped_matmul_slice_k  # example 05
from .optimized_matmul import optimized_matmul  # example 06
from .grouped_matmul_slice_m_per_token_dequant import (
    grouped_matmul_slice_m_per_token_dequant,               # example 07
    grouped_matmul_slice_m_per_token_dequant_multistage,    # example 10
    grouped_matmul_slice_k_per_token_dequant,               # example 11
)
from .grouped_matmul import grouped_matmul  # example 08
from .splitk_matmul import splitk_matmul  # example 09
from .quant_matmul import quant_matmul  # example 12
from .basic_matmul_tla import basic_matmul_tla  # example 13
from .optimized_matmul_tla import optimized_matmul_tla  # example 14
from .mla import mla  # example 19
from .matmul_bias import matmul_bias  # example 20
from .basic_matmul_preload_zN import basic_matmul_preload_zN  # example 21
from .padding_splitk_matmul import padding_splitk_matmul  # example 22
from .flash_attention_infer import flash_attention_infer  # example 23
from .flash_attention_infer_tla import flash_attention_infer_tla  # example 40
from .matmul_full_loadA import matmul_full_loadA  # example 25
from .matmul_relu import matmul_relu  # example 26
from .matmul_gelu import matmul_gelu  # example 27
from .matmul_silu import matmul_silu  # example 28
from .a2_fp8_e4m3_matmul import a2_fp8_e4m3_matmul  # example 29
from .w8a16_matmul import w8a16_matmul  # example 30
from .small_matmul import small_matmul  # example 31
from .w4a8_matmul import w4a8_matmul  # example 32
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
from .broadcast_matmul_perblock_quant import broadcast_matmul_perblock_quant  # example 62
from .ascend950_matmul_evg import EvgPostprocessMode, ascend950_matmul_evg  # example 64

__all__ = [
    "basic_matmul",                       # example 00
    "batched_matmul",                     # example 01
    "grouped_matmul_slice_m",             # example 02
    "matmul_add",                         # example 03
    "padding_matmul",                     # example 04
    "grouped_matmul_slice_k",             # example 05
    "optimized_matmul",                   # example 06
    "grouped_matmul_slice_m_per_token_dequant",               # example 07
    "grouped_matmul",                     # example 08
    "splitk_matmul",                      # example 09
    "grouped_matmul_slice_m_per_token_dequant_multistage",    # example 10
    "grouped_matmul_slice_k_per_token_dequant",               # example 11
    "quant_matmul",                       # example 12
    "basic_matmul_tla",                   # example 13
    "optimized_matmul_tla",               # example 14
    "mla",                                # example 19
    "matmul_bias",                        # example 20
    "basic_matmul_preload_zN",            # example 21
    "padding_splitk_matmul",              # example 22
    "flash_attention_infer",              # example 23
    "flash_attention_infer_tla",          # example 40
    "matmul_full_loadA",                  # example 25
    "matmul_relu",                        # example 26
    "matmul_gelu",                        # example 27
    "matmul_silu",                        # example 28
    "a2_fp8_e4m3_matmul",                # example 29
    "w8a16_matmul",                       # example 30
    "small_matmul",                       # example 31
    "w4a8_matmul",                        # example 32
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
    "broadcast_matmul_perblock_quant",    # example 62
    "ascend950_matmul_evg",               # example 64
    "EvgPostprocessMode",                 # example 64
]
