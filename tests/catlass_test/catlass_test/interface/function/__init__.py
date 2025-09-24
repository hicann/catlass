# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
from typing import Iterable, List, Optional, Union

import torch
from catlass_test import CATLASS_TEST_KERNEL_EXAMPLES_PATH
from catlass_test.adapter import (
    BatchedMatmulAdapter,
    GroupedMatmulAdapter,
    MatmulAdapter,
)
from catlass_test.common import OpType


def basic_matmul(
    input: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Test function for `examples/00_basic_matmul`.
    This function is equal to `torch.mm`.
    """
    output_tensors = {"C": out} if out is not None else {}
    adapter = MatmulAdapter(
        os.path.join(
            CATLASS_TEST_KERNEL_EXAMPLES_PATH, "00_basic_matmul", "basic_matmul.hpp"
        ),
        {"A": input, "B": mat2},
        output_tensors,
        {"out_dtype": out_dtype},
        OpType.AIC_ONLY,
    )
    adapter.run()
    return adapter.get_tensor("C")


def batched_matmul(
    input: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Test function for `examples/01_batched_matmul`.
    This function is equal to `torch.bmm`.
    """
    output_tensors = {"C": out} if out is not None else {}

    adapter = BatchedMatmulAdapter(
        os.path.join(
            CATLASS_TEST_KERNEL_EXAMPLES_PATH, "01_batched_matmul", "batched_matmul.hpp"
        ),
        {"A": input, "B": mat2},
        output_tensors,
        {"out_dtype": out_dtype},
        OpType.AIC_ONLY,
    )
    adapter.run()
    return adapter.get_tensor("C")


def grouped_matmul_slice_m(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: Union[torch.Tensor, Iterable[int]],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Test function for `examples/02_grouped_matmul_slice_m`.
    This function does not have equivalent torch function.
    """
    if isinstance(group_list, Iterable):
        group_list_tensor = torch.tensor(group_list, dtype=torch.int64).npu()
    else:
        group_list_tensor = group_list
    output_tensors = {"C": out} if out is not None else {}

    adapter = GroupedMatmulAdapter(
        os.path.join(
            CATLASS_TEST_KERNEL_EXAMPLES_PATH,
            "02_grouped_matmul_slice_m",
            "grouped_matmul_slice_m.hpp",
        ),
        {"A": x, "B": weight, "GroupList": group_list_tensor},
        output_tensors,
        {"out_dtype": out_dtype},
        OpType.AIC_ONLY,
        "m",
        True,
    )
    adapter.run()
    return adapter.get_tensor("C")


def padding_matmul(
    input: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Test function for `examples/04_padding_matmul`.
    This function is equal to `torch.mm`.
    """
    output_tensors = {"C": out} if out is not None else {}
    adapter = MatmulAdapter(
        os.path.join(
            CATLASS_TEST_KERNEL_EXAMPLES_PATH, "04_padding_matmul", "padding_matmul.hpp"
        ),
        {"A": input, "B": mat2},
        output_tensors,
        {"out_dtype": out_dtype},
        OpType.AIC_ONLY,
    )
    adapter.run()
    return adapter.get_tensor("C")


def grouped_matmul_slice_k(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: Union[torch.Tensor, Iterable[int]],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Test function for `examples/05_grouped_matmul_slice_k`.
    This function does not have equivalent torch function.
    """
    if isinstance(group_list, Iterable):
        group_list_tensor = torch.tensor(group_list, dtype=torch.int64).npu()
    else:
        group_list_tensor = group_list
    output_tensors = {"C": out} if out is not None else {}

    adapter = GroupedMatmulAdapter(
        os.path.join(
            CATLASS_TEST_KERNEL_EXAMPLES_PATH,
            "05_grouped_matmul_slice_k",
            "grouped_matmul_slice_k.hpp",
        ),
        {"A": x, "B": weight, "GroupList": group_list_tensor},
        output_tensors,
        {"out_dtype": out_dtype},
        OpType.AIC_ONLY,
        "k",
        True,
    )
    adapter.run()
    return adapter.get_tensor("C")


def splitk_matmul(
    input: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Test function for `examples/09_splitk_matmul`.
    This function is equal to `torch.mm`.
    """
    output_tensors = {"C": out} if out is not None else {}
    adapter = MatmulAdapter(
        os.path.join(
            CATLASS_TEST_KERNEL_EXAMPLES_PATH, "09_splitk_matmul", "splitk_matmul.hpp"
        ),
        {"A": input, "B": mat2},
        output_tensors,
        {"out_dtype": out_dtype},
        OpType.MIX_AIC_1_2,
    )
    adapter.run()
    return adapter.get_tensor("C")


def conv_bias(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: List[int],
    padding: List[int],
    dilations: List[int],
    groups: int,
) -> torch.Tensor:
    pass
