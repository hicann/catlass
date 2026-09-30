//===----------------------------------------------------------------------===//
// Cube-unit ops: the mmad family. The tla.cube region itself is in
// TlaRegionOps.td.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

def Tla_MmadOp : Tla_Op<"mmad", []> {
  let summary = "Matrix multiply-accumulate";
  let description = [{
    Perform a matrix multiply-accumulate on tensor views. Must be nested inside a
    ``tla.cube`` region.

    ``compute_order`` is always supplied by the frontend. The lowering emits
    ``hivm.set_ctrl`` on ``CTRL[51]`` (0 = ``M_FIRST``, 1 = ``N_FIRST``): when
    every mmad in a function agrees on the value it is emitted once at the
    function entry; if a function mixes values it is emitted ahead of each mmad
    runtime call instead.

    ``hf32_mode`` is always supplied by the frontend. The lowering emits
    ``hivm.set_ctrl`` on ``CTRL[46]`` (enable HF32 rounding) and ``CTRL[47]``
    (rounding mode, ``HF32_NEAREST_ZERO`` -> 1 / ``HF32_NEAREST_EVEN`` -> 0,
    mirroring ``AscendC::SetHF32Mode`` / ``AscendC::SetHF32TransMode``): when
    every mmad in a function agrees on the value it is emitted once at the
    function entry; if a function mixes values it is emitted ahead of each mmad
    runtime call instead.
  }];
  let arguments = (ins Tla_TensorType:$acc, Tla_TensorType:$lhs, Tla_TensorType:$rhs,
                       I1:$init_c, I64:$unit_flag,
                       Tla_ComputeOrderAttr:$compute_order,
                       Tla_HF32ModeAttr:$hf32_mode);
  let assemblyFormat = [{
    $acc `,` $lhs `,` $rhs `,` $init_c `,` $unit_flag attr-dict `:`
    qualified(type($acc)) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `,` type($init_c) `,` type($unit_flag)
  }];
  let hasVerifier = 1;
}

def Tla_MmadMxOp : Tla_Op<"mmad_mx", []> {
  let summary = "Microscaling (MX) matrix multiply-accumulate";
  let description = [{
    Perform a microscaling matrix multiply-accumulate on tensor views. Must be
    nested inside a ``tla.cube`` region.

    A distinct op rather than a flag on ``tla.mmad``, mirroring the hardware:
    the cube has a dedicated ``mad_mx`` instruction, which AscendC exposes as
    ``asc_mmad_mx``.

    Like that instruction, this op takes **no scale operand**. ``lhs``/``rhs``
    must have been loaded by ``tla.copy`` with its ``scale`` operand, which is
    what attaches the e8m0 block to the L0 tile; the scale then lives in a side
    buffer addressed from the L0 tile rather than being passed here.

    A packed-fp4 operand names its encoding in its own element type
    (``!tla.f4e2m1`` / ``!tla.f4e1m2``), so there is nothing to state here and
    nothing that can disagree with the load that wrote the tile.
  }];
  let arguments = (ins Tla_TensorType:$acc, Tla_TensorType:$lhs, Tla_TensorType:$rhs,
                       I1:$init_c, I64:$unit_flag,
                       Tla_ComputeOrderAttr:$compute_order);
  let assemblyFormat = [{
    $acc `,` $lhs `,` $rhs `,` $init_c `,` $unit_flag attr-dict `:`
    qualified(type($acc)) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `,` type($init_c) `,` type($unit_flag)
  }];
  let hasVerifier = 1;
}
