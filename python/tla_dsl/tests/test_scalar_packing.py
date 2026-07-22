import struct

import catlass as tla


def test_float16_packing_basic() -> None:
    assert tla.Float16(0.0).__c_pointers__()[0] == 0x0000
    assert tla.Float16(1.0).__c_pointers__()[0] == 0x3C00
    assert tla.Float16(-1.0).__c_pointers__()[0] == 0xBC00
    assert tla.Float16(0.5).__c_pointers__()[0] == 0x3800


def test_bfloat16_packing_basic() -> None:
    assert tla.BFloat16(0.0).__c_pointers__()[0] == 0x0000
    assert tla.BFloat16(1.0).__c_pointers__()[0] == 0x3F80
    assert tla.BFloat16(-1.0).__c_pointers__()[0] == 0xBF80


def test_float16_nan_inf() -> None:
    inf = tla.Float16(float("inf")).__c_pointers__()[0]
    ninf = tla.Float16(float("-inf")).__c_pointers__()[0]
    nan = tla.Float16(float("nan")).__c_pointers__()[0]
    assert inf == 0x7C00
    assert ninf == 0xFC00
    assert (nan & 0x7C00) == 0x7C00
    assert (nan & 0x03FF) != 0


def test_bfloat16_nan_inf() -> None:
    inf = tla.BFloat16(float("inf")).__c_pointers__()[0]
    ninf = tla.BFloat16(float("-inf")).__c_pointers__()[0]
    nan = tla.BFloat16(float("nan")).__c_pointers__()[0]
    assert inf == 0x7F80
    assert ninf == 0xFF80
    assert (nan & 0x7F80) == 0x7F80
    assert (nan & 0x007F) != 0


def test_float16_rounding_halfway() -> None:
    # Value halfway between 1.0 and next representable half (1.0009765625)
    halfway = 1.00048828125
    assert tla.Float16(halfway).__c_pointers__()[0] == 0x3C00


def test_bfloat16_rounding_halfway_even() -> None:
    # Halfway: lower 16 bits are exactly 0x8000, upper even → stay (RNE).
    f32_bits = 0x3F808000
    value = struct.unpack("f", struct.pack("I", f32_bits))[0]
    assert tla.BFloat16(value).__c_pointers__()[0] == 0x3F80


def test_bfloat16_rounding_halfway_odd_upper() -> None:
    # f32 = 1.0 + 2^-7 + 2^-8 = 1.01171875, upper=0x3F81 (odd), lower=0x8000
    # RNE rounds up to 0x3F82 (bf16 = 1.015625); truncate would keep 0x3F81.
    f32_bits = 0x3F818000
    value = struct.unpack("f", struct.pack("I", f32_bits))[0]
    assert tla.BFloat16(value).__c_pointers__()[0] == 0x3F82
