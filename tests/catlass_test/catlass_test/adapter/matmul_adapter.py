# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Dict, Tuple

from catlass_test.adapter import AdapterBase
from catlass_test.catlass.gemm_coord import GemmCoord


class MatmulAdapter(AdapterBase):
    def get_output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        ma, ka = self.A.shape[-2:]
        kb, nb = self.B.shape[-2:]
        assert ka == kb
        return {"C": (ma, nb)}

    def get_output_dtypes(self):
        if self.attrs.get("output_dtype") is None:
            assert self.A.dtype == self.B.dtype
            self.attrs["output_dtype"] = self.A.dtype
        return {"C": self.attrs["output_dtype"]}

    def get_problem_shape(self) -> GemmCoord:
        assert "A" in self.input_tensors
        assert "B" in self.input_tensors
        assert "C" in self.output_tensors
        # ma, ka = (swap(*self.A.shape) if self.get_transpose("A") else self.A.shape)[-2:]
        # kb, nb = (swap(*self.B.shape) if self.get_transpose("B") else self.B.shape)[-2:]
        ma, ka = self.A.shape[-2:]
        kb, nb = self.B.shape[-2:]
        mc, nc = self.C.shape[-2:]
        assert ma == mc
        assert ka == kb
        assert nb == nc

        return GemmCoord(ma, nb, ka)

    def get_problem_count(self) -> int:
        return 1

    @property
    def A(self):
        return self.get_tensor("A")

    @property
    def B(self):
        return self.get_tensor("B")

    @property
    def C(self):
        return self.get_tensor("C")

    def get_transpose(self, tensor_name: str) -> bool:
        return bool(self.attrs.get(f"Trans{tensor_name}", False))
