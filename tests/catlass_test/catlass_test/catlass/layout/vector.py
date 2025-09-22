from ctypes import Structure, c_uint, c_int64


class VectorLayout(Structure):
    _fieids_ = [("shape_", c_uint * 1), ("stride_", c_int64 * 1)]
