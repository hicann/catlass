//===----------------------------------------------------------------------===//
// Memory-to-memory data movement: copy, copy_mx and fill, plus the parameter
// bundle tla.copy takes on the L0C->dst path.
//
// tla.load / tla.store are not here: they move UB <-> vector register and are
// only valid inside a tla.vec.func, so they live in TlaSimdOps.td.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

def Tla_CopyL0C2DstParamsOp : Tla_Op<"CopyL0C2DstParams", [Pure]> {
  let summary = "CopyL0C2DstParams op";
  let arguments = (ins
  // attributes
    I64Attr:$unit_flag,
    BoolAttr:$relu_enable,
    Tla_QuantModeAttr:$quant_mode,
    Tla_L0C2UBModeAttr:$l0c2ub_mode,
  // operands
    Optional<AnyTypeOf<[F32, Tla_TensorType]>>:$quant_scale_or_tensor
  );
  let results = (outs Tla_CopyL0C2DstParamsType:$params);
}

def Tla_CopyOp : Tla_Op<"copy", []> {
  let summary = "Copy";
  let description = [{
    Copy between tensors/views in different address spaces. Cube data-path
    copies (GM->L1, L1->L0A, L1->L0B, L0C->GM, L0C->UB, L0C->L1, L1->UB) must be nested
    inside a ``tla.cube`` region; vector staging copies (GM->UB, UB->GM, UB->L1)
    must be nested inside a ``tla.vector`` region.

    The atomic operation is enabled if ``atomic_mode`` provided, and currently only ``AtomicMode.ADD`` is supported, which is lowered to HIVM ``SetAtomicOp``

    A microscaling (MX) L1->L0A/L0B load is ``tla.copy_mx``, which carries the
    operand's e8m0 scale block. The frontend spells that as ``tla.copy`` with a
    ``scale=`` argument, mirroring AscendC's five-argument ``LoadData``, but it
    lowers to the separate op.
  }];
  let arguments = (ins
    Tla_TensorType:$dst, Tla_TensorType:$src,
    Optional<AnyTypeOf<[Tla_CopyL0C2DstParamsType]>>:$params,
    OptionalAttr<Tla_AtomicModeAttr>:$atomic_mode);
  let assemblyFormat = [{
    $dst `,` $src (`,` $params^)? attr-dict `:` qualified(type($dst)) `,` qualified(type($src)) (`,` qualified(type($params))^)?
  }];
  let hasVerifier = 1;
}

def Tla_CopyMxOp : Tla_Op<"copy_mx", []> {
  let summary = "Microscaling (MX) L1->L0 copy";
  let description = [{
    Copy an MX operand tile from L1 to L0A/L0B, attaching its shared-exponent
    scale block. Must be nested inside a ``tla.cube`` region.

    A separate op from ``tla.copy`` so that the plain copy keeps a fixed operand
    shape. The frontend still spells this as ``tla.copy(dst, src, scale=...)``,
    mirroring AscendC, where the MX load is the five-argument overload of
    ``LoadData`` rather than a distinct entry point.

    On this hardware the scale is consumed by the *load*, not by the matmul:
    ``tla.mmad_mx`` takes no scale operand, and the MX association is
    established when the operand lands in L0.

    ``scale`` is an L1 tile of OCP ``e8m0`` shared exponents, one per
    ``MX_SCALE_GROUP_NUM`` (32) elements along K. Its element type is the
    dialect's own ``!tla.f8e8m0`` (``tla.Float8E8M0`` on the Python side); like
    the other cube element formats it carries no LLVM lowering, so at the
    memref boundary it is buffered as i8, one byte per exponent. It must be
    tagged ``zZMxScale`` for an L1A and ``nNMxScale`` for an L1B.

    The destination L0 tile is what makes the matmul an MX matmul, so it must be
    consumed by ``tla.mmad_mx`` rather than ``tla.mmad``.
  }];
  let arguments = (ins Tla_TensorType:$dst, Tla_TensorType:$src, Tla_TensorType:$scale);
  let assemblyFormat = [{
    $dst `,` $src `,` $scale attr-dict `:`
    qualified(type($dst)) `,` qualified(type($src)) `,` qualified(type($scale))
  }];
  let hasVerifier = 1;
}

def Tla_FillOp : Tla_Op<"fill", []> {
  let summary = "Fill a region of an L1 zN/nZ tile with a scalar bit pattern";
  let description = [{
    Fill ``dst`` with ``value`` over the region ``[coord, coord + shape)``.
    Cube only: the destination must be an L1 tile tagged ``zN`` or ``nZ``, with
    a fp4/fp8 element type, and ``value`` must be 0 -- the op exists to
    zero the K-pad tail of an MX operand tile, nothing else.

    ``value`` travels as its i32 bit pattern (a uint16 pattern for the
    fp4/fp8 storage). The frontend crops ``shape`` against the tile's origin
    shape and the given coord before this op is formed. The device helper
    aligns the coordinate up to the C0 boundary (32 bytes) along the layout's
    C0 axis before filling, so a fill that starts or ends inside a C0 block
    degrades to the enclosing aligned blocks and a zero-length fill is a
    no-op.
  }];
  let arguments = (ins Tla_TensorType:$dst, I32:$value,
                         Tla_CoordType:$coord, Tla_ShapeType:$shape);
  let assemblyFormat = [{
    $dst `,` $value `,` $coord `,` $shape attr-dict `:`
    qualified(type($dst)) `,` type($value) `,` type($coord) `,` type($shape)
  }];
  let hasVerifier = 1;
}
