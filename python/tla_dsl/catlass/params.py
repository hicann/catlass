from __future__ import annotations

import enum
from .types import TlaTensor
from dataclasses import dataclass


class QuantMode(enum.IntEnum):
    NO_QUANT = 0
    PER_TENSOR = 1  # need quant_scale
    PER_CHANNEL = 2 # need quant_tensor

    def __str__(self):
        return self.name  # "NO_QUANT"/..

class L0C2UBMode(enum.IntEnum):
    NO_SPLIT_VEC_0 = 0
    NO_SPLIT_VEC_1 = 1
    SPLIT_M = 2
    SPLIT_N = 3

    def __str__(self):
        return self.name  # "NO_SPLIT_VEC_0"/..

class CopyParams:
    """Marker annotation for base copy params. """
    pass


@dataclass
class CopyL0C2DstParams(CopyParams):
    unit_flag: int = 0
    relu_enable: bool = False
    quant_mode: QuantMode = QuantMode.NO_QUANT
    quant_scale: float | None = None
    quant_tensor: TlaTensor | None = None
    l0c2ub_mode: L0C2UBMode = L0C2UBMode.NO_SPLIT_VEC_0

    def _validate(self):
        from .core_api import _category
        from .execution_lowering import TlaLoweringError
        VALID_UNIT_FLAG = (0b00, 0b11)
        cls_name = type(self).__name__
        if not isinstance(self.unit_flag, int):
            raise TlaLoweringError(f"{cls_name}.unit_flag must be a compile-time int")
        if self.unit_flag not in VALID_UNIT_FLAG:
            raise ValueError(f"{cls_name}.unig_flag must be one of {VALID_UNIT_FLAG}")
        if not isinstance(self.relu_enable, bool):
            raise TlaLoweringError(f"{cls_name}.relu_enable must be a compile-time bool")
        if not isinstance(self.quant_mode, QuantMode):
            raise TlaLoweringError(
                f"{cls_name}.quant_mode must be a {QuantMode}, got {type(self.quant_mode).__name__}"
            )
        if not isinstance(self.l0c2ub_mode, L0C2UBMode):
            raise TlaLoweringError(
                f"{cls_name}.l0c2ub_mode must be a {L0C2UBMode}, got {type(self.l0c2ub_mode).__name__}"
            )
        if self.quant_mode == QuantMode.PER_TENSOR:
            if not isinstance(self.quant_scale, float):
                raise TlaLoweringError(
                    f"{cls_name}.quant_scale must be a `float` temporarily, got {type(self.quant_scale).__name__}"
                )
        if self.quant_mode == QuantMode.PER_CHANNEL:
            if _category(self.quant_tensor) != "tensor":
                raise TlaLoweringError(
                    f"{cls_name}.quant_tensor must be a tensor, got {type(self.quant_tensor).__name__}"
                )


# ---------------------------------------------------------------------------
# CastParams: the four knobs of a tla.cast (VectorSSA.to). The enum member
# *values* are the MLIR-asm keywords; the integer codes match the
# I32EnumAttrCase values in Tla.td and the order in the tla.cast trait array.
# ---------------------------------------------------------------------------


class RegSlot(enum.Enum):
    """Packed-register position the narrow cast result lands in.

    For a 2x-width cast (e.g. f32->f16, i32<->i16) this maps to the AVE even/odd
    ``part`` (only ZERO=part_even / ONE=part_odd are valid there). For a 4x-width
    cast (i32<->i8) it maps to the AVE pack pattern ``pp`` and all four values
    select the pack quarter: ZERO=pp0, ONE=pp1, TWO=pp2, THREE=pp3.
    """

    ZERO = "zero"    # part_even / pp0
    ONE = "one"      # part_odd  / pp1
    TWO = "two"      # pp2 (4x casts only)
    THREE = "three"  # pp3 (4x casts only)

    def __str__(self) -> str:
        return self.value


class SatMode(enum.Enum):
    """Overflow behaviour of the cast (AVE ``sat`` BoolAttr)."""

    UNKNOWN = "unknown"
    SAT = "sat"
    NOSAT = "nosat"

    def __str__(self) -> str:
        return self.value


class RoundMode(enum.Enum):
    """Rounding applied on precision loss (HIVM ``round_mode``)."""

    CAST_ROUND = "cast_round"   # round to nearest, tie away from zero
    CAST_FLOOR = "cast_floor"   # round toward -inf
    CAST_CEIL = "cast_ceil"     # round toward +inf
    CAST_TRUNC = "cast_trunc"   # round toward zero

    def __str__(self) -> str:
        return self.value


_REG_SLOT_CODE = {RegSlot.ZERO: 0, RegSlot.ONE: 1, RegSlot.TWO: 2, RegSlot.THREE: 3}
_SAT_MODE_CODE = {SatMode.UNKNOWN: 0, SatMode.SAT: 1, SatMode.NOSAT: 2}
_ROUND_MODE_CODE = {
    RoundMode.CAST_ROUND: 0,
    RoundMode.CAST_FLOOR: 1,
    RoundMode.CAST_CEIL: 2,
    RoundMode.CAST_TRUNC: 3,
}


@dataclass(frozen=True)
class CastParams:
    """The three knobs of a ``tla.cast`` (``VectorSSA.to``).

    Args:
        reg_slot: destination packed-register position (:class:`RegSlot`).
        sat_mode: overflow behaviour (:class:`SatMode`).
        round_mode: rounding on precision loss (:class:`RoundMode`).
    """

    reg_slot: RegSlot = RegSlot.ZERO
    sat_mode: SatMode = SatMode.NOSAT
    round_mode: RoundMode = RoundMode.CAST_ROUND

    def codes(self) -> list[int]:
        """The three enum codes for the tla.cast DenseI32ArrayAttr, in order."""
        return [
            _REG_SLOT_CODE[self.reg_slot],
            _SAT_MODE_CODE[self.sat_mode],
            _ROUND_MODE_CODE[self.round_mode],
        ]
