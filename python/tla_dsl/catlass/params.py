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
