# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ctypes import Structure, c_uint32, c_int32, c_int64


class GemmCoord(Structure):
    _fields_ = [("m", c_uint32),
                ("n", c_uint32),
                ("k", c_uint32)]

    def __init__(self, m, n, k):
        super().__init__()
        self.m = (c_uint32)(m)
        self.n = (c_uint32)(n)
        self.k = (c_uint32)(k)

    @property
    def m(self):
        return self.m

    @property
    def n(self):
        return self.n

    @property
    def k(self):
        return self.k

    @staticmethod
    def get_namespace():
        return "Act::"


class MatrixCoord(Structure):
    _fields_ = [("idx", c_uint32 * 2)]

    def __init__(self, i : int, j : int):
        super().__init__()
        self.idx = (c_uint32 * 2)(i, j)

    def at(self, index):
        if index >= 0 and index < 2:
            return self.idx[index]
        else:
            raise IndexError("Index out of range")

    def row(self):
        return self.at(0)

    def column(self):
        return self.at(1)

    @staticmethod
    def get_namespace():
        return "Act::"


class RowMajor(Structure):
    _fields_ = [("shape", c_int32 * 2),
                ("stride", c_int64 * 2)]

    def __init__(self, rows : int = 0, cols : int = 0, ldm : int = None):
        super().__init__()
        self.shape = (c_int32 * 2)(rows, cols)
        if ldm is None:
            self.stride = (c_int64 * 2)(cols, 1)
        else:
            self.stride = (c_int64 * 2)((c_int64)(ldm), 1)

    @staticmethod
    def get_namespace():
        return "Act::layout::"


class ColumnMajor(Structure):
    _fields_ = [("shape", c_int32 * 2),
                ("stride", c_int64 * 2)]

    def __init__(self, rows : int = 0, cols : int = 0, ldm : int = None):
        super().__init__()
        self.shape = (c_int32 * 2)(rows, cols)
        if ldm is None:
            self.stride = (c_int64 * 2)(1, rows)
        else:
            self.stride = (c_int64 * 2)(1, ldm)

    @staticmethod
    def get_namespace():
        return "Act::layout::"