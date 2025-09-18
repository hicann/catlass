# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
from typing import Optional

import torch

from catlass_test import CATLASS_TEST_KERNEL_EXAMPLES_PATH
from catlass_test.adapter import MatmulAdapter


def basic_matmul(
    input: torch.Tensor, mat2: torch.Tensor, out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Test function for `examples/00_basic_matmul`.
    This function is equal to `torch.mm`.
    """
    adapter = MatmulAdapter(
        os.path.join(
            CATLASS_TEST_KERNEL_EXAMPLES_PATH, "00_basic_matmul", "basic_matmul.hpp"
        ),
        {"A": input, "B": mat2},
        attrs={"out_dtype": out_dtype},
    )
    adapter.run()
    return adapter.get_tensor("C")
