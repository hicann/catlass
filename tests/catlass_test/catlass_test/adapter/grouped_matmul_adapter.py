# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Dict, List, Literal, Tuple, Union

import torch

from catlass_test.adapter import MatmulAdapter
from catlass_test.catlass.gemm_coord import GemmCoord
from catlass_test.catlass_test.common import swap


class GroupedMatmulCase(MatmulAdapter):
    def __init__(
        self,
        kernel_name: str,
        kernel_src_file: str,
        input_tensors: Dict[str, torch.Tensor],
        output_tensors: Dict[str, torch.Tensor] = {},
        slice_axis: Literal["m", "k", "n"] = "m",
        group_list_prefix_sum: bool = False,
        attrs: Dict[str, Any] = {},
    ) -> None:
        self.slice_axis = slice_axis
        self.group_list_prefix_sum = group_list_prefix_sum
        super().__init__(kernel_src_file, input_tensors, output_tensors, attrs)

    @property
    def GroupList(self):
        return self.get_tensor("GroupList")

    def get_output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        if self.slice_axis == "m":
            # [M, k] @ [g, k, n] -> [M, n]
            assert len(self.A.shape) == 2
            assert len(self.B.shape) == 3
            Ma, ka = (swap(*self.A.shape) if self.get_transpose("A") else self.A.shape)[
                -2:
            ]
            g, kb, nb = (
                swap(*self.B.shape) if self.get_transpose("B") else self.B.shape
            )[-3:]
            assert ka == kb
            return {"C": (Ma, nb)}
        elif self.slice_axis == "k":
            # [m, K] @ [K, n] -> [g, m, n]
            assert len(self.A.shape) == 2
            assert len(self.B.shape) == 2
            ma, Ka = (swap(*self.A.shape) if self.get_transpose("A") else self.A.shape)[
                -2:
            ]
            Kb, nb = (swap(*self.B.shape) if self.get_transpose("B") else self.B.shape)[
                -2:
            ]
            assert Ka == Kb
            return {"C": (self.get_problem_count(), ma, nb)}
        elif self.slice_axis == "n":
            # [g, m, k] @ [k, N] -> [m, N]
            assert len(self.A.shape) == 3
            assert len(self.B.shape) == 2
            g, ma, ka = (
                swap(*self.A.shape) if self.get_transpose("A") else self.A.shape
            )[-3:]
            kb, Nb = (swap(*self.B.shape) if self.get_transpose("B") else self.B.shape)[
                -2:
            ]
            assert g == self.get_problem_count()
            assert ka == kb

            return {"C": (ma, Nb)}
        return super().get_output_shapes()

    def get_problem_shape(self) -> GemmCoord:
        assert len(self.input_tensors) == 3
        assert len(self.output_tensors) == 1
        group_list_len = len(self.GroupList.shape)
        assert group_list_len == 1
        if self.group_list_prefix_sum:
            group_list_sum = self.GroupList[-1].item()
        else:
            group_list_sum = self.GroupList.sum().item()
        if self.slice_axis == "m":
            # [M, k] @ [g, k, n] -> [M, n]
            assert len(self.A.shape) == 2
            assert len(self.B.shape) == 3
            assert len(self.C.shape) == 2
            Ma, ka = (swap(*self.A.shape) if self.get_transpose("A") else self.A.shape)[
                -2:
            ]
            g, kb, nb = (
                swap(*self.B.shape) if self.get_transpose("B") else self.B.shape
            )[-3:]
            Mc, nc = self.C.shape
            assert g == self.get_problem_count()
            assert Ma == Mc
            assert ka == kb
            assert nb == nc
            assert group_list_sum == Ma
            return GemmCoord(Ma, nb, ka)
        elif self.slice_axis == "k":
            # [m, K] @ [K, n] -> [g, m, n]
            assert len(self.A.shape) == 2
            assert len(self.B.shape) == 2
            assert len(self.C.shape) == 3
            ma, Ka = (swap(*self.A.shape) if self.get_transpose("A") else self.A.shape)[
                -2:
            ]
            Kb, nb = (swap(*self.B.shape) if self.get_transpose("B") else self.B.shape)[
                -2:
            ]
            g, mc, nc = self.C.shape
            assert g == self.get_problem_count()
            assert ma == mc
            assert Ka == Kb
            assert mc == nc
            assert group_list_sum == Ka

            return GemmCoord(ma, nb, Ka)
        elif self.slice_axis == "n":
            # [g, m, k] @ [k, N] -> [m, N]
            assert len(self.A.shape) == 3
            assert len(self.B.shape) == 2
            assert len(self.C.shape) == 2
            g, ma, ka = (
                swap(*self.A.shape) if self.get_transpose("A") else self.A.shape
            )[-3:]
            kb, Nb = (swap(*self.B.shape) if self.get_transpose("B") else self.B.shape)[
                -2:
            ]
            mc, Nc = self.C.shape
            assert g == self.get_problem_count()
            assert ma == mc
            assert ka == kb
            assert Nb == Nc
            assert group_list_sum == Nb

            return GemmCoord(ma, Nb, ka)
        return GemmCoord(0, 0, 0)

    def get_problem_count(self):
        return self.GroupList.shape[0]
