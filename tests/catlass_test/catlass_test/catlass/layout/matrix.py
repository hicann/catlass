from ctypes import Structure, c_uint, c_int64


class RowMajor(Structure):
    _fieids_ = [("shape_", c_uint * 2), ("stride_", c_int64 * 2)]


class ColumnMajor(Structure):
    _fieids_ = [("shape_", c_uint * 2), ("stride_", c_int64 * 2)]
