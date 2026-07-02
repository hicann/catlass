include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// include "Dialect/Tla/IR/TlaOpTraits.h"
// SameOperandsAndResultShape
// let hasFolder = 1;

def TlaDialect : Dialect {
  let name = "tla";
  let summary = "Tla dialect schema for TlaIR matmul prototyping.";
  let description = [{
    Tla dialect schema for TlaIR matmul prototyping.
  }];
  let cppNamespace = "::tla";
}

class Tla_Op<string mnemonic, list<Trait> traits = []> : Op<TlaDialect, mnemonic, traits>;

def Tla_CrossFlagType : TypeDef<TlaDialect, "CrossFlag", []> {
  let mnemonic = "cross_flag";
  let summary = "Synchronization flag used for cross-core coordination.";
  let assemblyFormat = "";
}

def Tla_FlagType : TypeDef<TlaDialect, "Flag", []> {
  let mnemonic = "flag";
  let summary = "Synchronization flag used for pipe coordination.";
  let assemblyFormat = "";
}

def Tla_MaskType : TypeDef<TlaDialect, "Mask", []> {
  let mnemonic = "mask";
  let summary = "Vector element-wise predicate/mask for conditional vector ops.";
  let assemblyFormat = "";
}


def Tla_MutexType : TypeDef<TlaDialect, "Mutex", []> {
  let mnemonic = "mutex";
  let summary = "Synchronization mutex used for resource coordination.";
  let assemblyFormat = "";
}


def Tla_AddressSpaceTypeParameter
    : TypeParameter<"::AddressSpace", "Tla address-space keyword"> {
  let parser = [{ [&]() -> ::mlir::FailureOr<::AddressSpace> {
    ::llvm::StringRef kw;
    if ($_parser.parseKeyword(&kw))
      return ::mlir::failure();
    auto sym = ::symbolizeAddressSpace(kw);
    if (!sym)
      return $_parser.emitError($_parser.getCurrentLocation())
             << "unknown AddressSpace keyword: " << kw;
    return *sym;
  }() }];
  let printer = "$_printer << ::stringifyAddressSpace($_self)";
}

def Tla_LayoutTagTypeParameter
    : TypeParameter<"::LayoutTag", "Tla tensor layout keyword"> {
  let parser = [{ [&]() -> ::mlir::FailureOr<::LayoutTag> {
    ::llvm::StringRef kw;
    if ($_parser.parseKeyword(&kw))
      return ::mlir::failure();
    auto sym = ::symbolizeLayoutTag(kw);
    if (!sym)
      return $_parser.emitError($_parser.getCurrentLocation())
             << "unknown LayoutTag keyword: " << kw;
    return *sym;
  }() }];
  let printer = "$_printer << ::stringifyLayoutTag($_self)";
}

class Tla_IndexTreeType<string name, string typeMnemonic, string typeSummary>
    : TypeDef<TlaDialect, name, []> {
  let mnemonic = typeMnemonic;
  let summary = typeSummary;
  let parameters = (ins ArrayRefParameter<"int64_t", "encoded index-tree">:$tree);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Tla_ShapeType : Tla_IndexTreeType<"Shape", "shape", "Shape value for tile/broadcast.">;

def Tla_CoordType : Tla_IndexTreeType<"Coord", "coord", "Coordinate tuple for hierarchical tiling.">;

def Tla_StrideType : Tla_IndexTreeType<"Stride", "stride", "Stride tuple for layouts.">;

def Tla_ValueType : TypeDef<TlaDialect, "Value", []> {
  let mnemonic = "value";
  let summary = "Register-resident value (scalar or vector tile).";
  let parameters = (ins OptionalParameter<"::mlir::Type", "element type">:$elementType);
  let hasCustomAssemblyFormat = 1;
}

def Tla_ScalarValueType : AnyTypeOf<[AnyInteger, AnyFloat, Index],
    "TLA scalar SSA value type">;

def Tla_BoolAttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin boolean attribute placeholder.">;

def Tla_I64AttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin i64 attribute placeholder.">;

def Tla_StrAttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin string attribute placeholder.">;

def Tla_SymbolNameAttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin symbol name attribute placeholder.">;

def Tla_TypeAttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin type attribute placeholder.">;

class Tla_EnumAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TlaDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class Tla_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
}

def Tla_LayoutTag_RowMajor : I32EnumAttrCase<"row_major", 0>;
def Tla_LayoutTag_ColumnMajor : I32EnumAttrCase<"column_major", 1>;
def Tla_LayoutTag_zN : I32EnumAttrCase<"zN", 2>;
def Tla_LayoutTag_nZ : I32EnumAttrCase<"nZ", 3>;
def Tla_LayoutTag_zZ : I32EnumAttrCase<"zZ", 4>;
def Tla_LayoutTag_L0CLayout : I32EnumAttrCase<"L0Clayout", 5>;
def Tla_LayoutTagEnum : Tla_I32Enum<"LayoutTag", "Tensor layout tag for tla.tensor types.", [Tla_LayoutTag_RowMajor, Tla_LayoutTag_ColumnMajor, Tla_LayoutTag_zN, Tla_LayoutTag_nZ, Tla_LayoutTag_zZ, Tla_LayoutTag_L0CLayout]>;

def Tla_AddressSpace_Generic : I32EnumAttrCase<"generic", 0>;
def Tla_AddressSpace_Gm : I32EnumAttrCase<"gm", 1>;
def Tla_AddressSpace_L1 : I32EnumAttrCase<"l1", 2>;
def Tla_AddressSpace_L0a : I32EnumAttrCase<"l0a", 3>;
def Tla_AddressSpace_L0b : I32EnumAttrCase<"l0b", 4>;
def Tla_AddressSpace_L0c : I32EnumAttrCase<"l0c", 5>;
def Tla_AddressSpace_Ub : I32EnumAttrCase<"ub", 6>;
def Tla_AddressSpaceEnum : Tla_I32Enum<"AddressSpace", "Tla pointer/tensor/memref address space.", [Tla_AddressSpace_Generic, Tla_AddressSpace_Gm, Tla_AddressSpace_L1, Tla_AddressSpace_L0a, Tla_AddressSpace_L0b, Tla_AddressSpace_L0c, Tla_AddressSpace_Ub]>;

def Tla_PtrAddressSpaceAttr : Tla_EnumAttr<"PtrAddressSpace", "ptr_address_space"> {
  let parameters = (ins EnumParameter<Tla_AddressSpaceEnum>:$value);
  let assemblyFormat = "`<` $value `>`";
  let description = [{
    On-chip ``!tla.ptr`` address space; carried on pointer types (e.g. with ``tla.alloc_ptr``).
  }];
}


def Tla_PtrType : TypeDef<TlaDialect, "Ptr", []> {
  let summary = "Pointer value (pointee type, address space, element alignment in bytes).";
  let mnemonic = "ptr";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "Logical pointee / value element type.">:$pointee,
    Tla_AddressSpaceTypeParameter:$addrspace,
    "unsigned":$alignment
  );
  let assemblyFormat = "`<` $pointee `,` $addrspace `,` $alignment `>`";
}

def Tla_LayoutType : TypeDef<TlaDialect, "Layout", []> {
  let mnemonic = "layout";
  let summary = "Packed shape+stride+layout-tag descriptor (Tla-style).";
  let parameters = (ins
    TypeParameter<"::tla::ShapeType", "shape tree type">:$shape,
    TypeParameter<"::tla::StrideType", "stride tree type">:$stride,
    TypeParameter<"::tla::ShapeType", "origin shape">:$origin,
    Tla_LayoutTagTypeParameter:$layoutTag
  );
  let assemblyFormat = "`<` qualified($shape) `,` qualified($stride) `,` qualified($origin) `,` $layoutTag `>`";
  let genVerifyDecl = 1;
}

def Tla_TensorType : TypeDef<TlaDialect, "TlaTensor", []> {
  let summary = "First-class tensor/view with structured layout, coordinate, and pointer.";
  let description = [{
    Structured tensor descriptor:
    ``!tla.tensor<!tla.layout<...>, !tla.coord<...>, !tla.ptr<...>>``.
  }];
  let mnemonic = "tensor";
  let parameters = (ins
    TypeParameter<"::tla::LayoutType", "layout descriptor">:$layout,
    TypeParameter<"::tla::CoordType", "coord tree type">:$coord,
    TypeParameter<"::tla::PtrType", "backing pointer type">:$ptr
  );
  let assemblyFormat = "`<` qualified($layout) `,` qualified($coord) `,` qualified($ptr) `>`";
  let genVerifyDecl = 1;
}

def Tla_CrossMode_Npu : I32EnumAttrCase<"npu", 0>;
def Tla_CrossMode_VectorsCore : I32EnumAttrCase<"vectors_core", 1>;
def Tla_CrossMode_SingleCore : I32EnumAttrCase<"single_core", 2>;
def Tla_CrossModeEnum : Tla_I32Enum<"CrossMode", "Cross-core synchronization mode for tla.cross_flag.", [Tla_CrossMode_Npu, Tla_CrossMode_VectorsCore, Tla_CrossMode_SingleCore]>;
def Tla_CrossModeAttr : Tla_EnumAttr<"CrossMode", "cross_mode"> {
  let parameters = (ins EnumParameter<Tla_CrossModeEnum>:$cross_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Cross-core synchronization mode for tla.cross_flag.
  }];
}

def Tla_Pipe_Scalar : I32EnumAttrCase<"scalar", 0>;
def Tla_Pipe_Vector : I32EnumAttrCase<"vector", 1>;
def Tla_Pipe_Cube : I32EnumAttrCase<"cube", 2>;
def Tla_Pipe_Mte1 : I32EnumAttrCase<"mte1", 3>;
def Tla_Pipe_Mte2 : I32EnumAttrCase<"mte2", 4>;
def Tla_Pipe_Mte3 : I32EnumAttrCase<"mte3", 5>;
def Tla_Pipe_All : I32EnumAttrCase<"all", 6>;
def Tla_Pipe_Mte4 : I32EnumAttrCase<"mte4", 7>;
def Tla_Pipe_Mte5 : I32EnumAttrCase<"mte5", 8>;
def Tla_Pipe_V2 : I32EnumAttrCase<"v2", 9>;
def Tla_Pipe_Fix : I32EnumAttrCase<"fix", 10>;
def Tla_Pipe_VirtualMte2L1a : I32EnumAttrCase<"virtual_mte2_l1a", 11>;
def Tla_Pipe_VirtualMte2L1b : I32EnumAttrCase<"virtual_mte2_l1b", 12>;
def Tla_Pipe_Num : I32EnumAttrCase<"num", 13>;
def Tla_PipeEnum : Tla_I32Enum<"Pipe", "Pipe identifier for tla.set_flag and tla.wait_flag.", [Tla_Pipe_Scalar, Tla_Pipe_Vector, Tla_Pipe_Cube, Tla_Pipe_Mte1, Tla_Pipe_Mte2, Tla_Pipe_Mte3, Tla_Pipe_All, Tla_Pipe_Mte4, Tla_Pipe_Mte5, Tla_Pipe_V2, Tla_Pipe_Fix, Tla_Pipe_VirtualMte2L1a, Tla_Pipe_VirtualMte2L1b, Tla_Pipe_Num]>;
def Tla_PipeAttr : Tla_EnumAttr<"Pipe", "pipe"> {
  let parameters = (ins EnumParameter<Tla_PipeEnum>:$pipe);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Pipe identifier for tla.set_flag and tla.wait_flag.
  }];
}

// QuantMode Enum Attr
def Tla_QuantMode_NoQuant : I32EnumAttrCase<"NO_QUANT", 0>;
def Tla_QuantMode_PerTensor : I32EnumAttrCase<"PER_TENSOR", 1>;
def Tla_QuantMode_PerChannel : I32EnumAttrCase<"PER_CHANNEL", 2>;
def Tla_QuantModeEnum : Tla_I32Enum<"QuantMode", "Quant Mode when l0c2dst.", [
  Tla_QuantMode_NoQuant,
  Tla_QuantMode_PerTensor,
  Tla_QuantMode_PerChannel
]>;
def Tla_QuantModeAttr : Tla_EnumAttr<"QuantMode", "quant_mode"> {
  let parameters = (ins EnumParameter<Tla_QuantModeEnum>:$quant_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Copy L0C to Dst quant mode.
  }];
}

// L0C2UBMode Enum Attr
def Tla_L0C2UBMode_NoSplitVec0 : I32EnumAttrCase<"NO_SPLIT_VEC_0", 0>;
def Tla_L0C2UBMode_NoSplitVec1 : I32EnumAttrCase<"NO_SPLIT_VEC_1", 1>;
def Tla_L0C2UBMode_SplitM : I32EnumAttrCase<"SPLIT_M", 2>;
def Tla_L0C2UBMode_SplitN : I32EnumAttrCase<"SPLIT_N", 3>;
def Tla_L0C2UBModeEnum : Tla_I32Enum<"L0C2UBMode", "L0C 2 UB split mode.", [
  Tla_L0C2UBMode_NoSplitVec0,
  Tla_L0C2UBMode_NoSplitVec1,
  Tla_L0C2UBMode_SplitM,
  Tla_L0C2UBMode_SplitN
]>;
def Tla_L0C2UBModeAttr : Tla_EnumAttr<"L0C2UBMode", "l0c2ub_mode"> {
  let parameters = (ins EnumParameter<Tla_L0C2UBModeEnum>:$l0c2ub_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Copy L0C 2 UB split mode.
  }];
}

def Tla_CopyL0C2DstParamsType : TypeDef<TlaDialect, "CopyL0C2DstParams", []> {
  let mnemonic = "copy_l0c2dst_params";
  let summary = "CopyL0C2DstParams for tla.copy.";
  let assemblyFormat = "";
}

def Tla_BinaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::BinaryOps">;
def Tla_BlockDimTrait : NativeOpTrait<"::mlir::tla::trait::BlockDim">;
def Tla_BlockIdxTrait : NativeOpTrait<"::mlir::tla::trait::BlockIdx">;
def Tla_BroadcastTrait : NativeOpTrait<"::mlir::tla::trait::Broadcast">;
def Tla_ControlOpsTrait : NativeOpTrait<"::mlir::tla::trait::ControlOps">;
def Tla_CopyTrait : NativeOpTrait<"::mlir::tla::trait::Copy">;
def Tla_CrossCoreSetFlagTrait : NativeOpTrait<"::mlir::tla::trait::CrossCoreSetFlag">;
def Tla_CrossCoreWaitFlagTrait : NativeOpTrait<"::mlir::tla::trait::CrossCoreWaitFlag">;
def Tla_CrossFlagTrait : NativeOpTrait<"::mlir::tla::trait::CrossFlag">;
def Tla_CubeTrait : NativeOpTrait<"::mlir::tla::trait::Cube">;
def Tla_CubeTernaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::CubeTernaryOps">;
def Tla_DependentBinaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::DependentBinaryOps">;
def Tla_DependentUnaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::DependentUnaryOps">;
def Tla_FlagTrait : NativeOpTrait<"::mlir::tla::trait::Flag">;
def Tla_FmaTrait : NativeOpTrait<"::mlir::tla::trait::Fma">;
def Tla_IntermediateBinaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::IntermediateBinaryOps">;
def Tla_IntermediateMemoryOpsTrait : NativeOpTrait<"::mlir::tla::trait::IntermediateMemoryOps">;
def Tla_IntermediateUnaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::IntermediateUnaryOps">;
def Tla_LoadTrait : NativeOpTrait<"::mlir::tla::trait::Load">;
def Tla_MakeTileLikeTrait : NativeOpTrait<"::mlir::tla::trait::MakeTileLike">;
def Tla_MemoryOpsTrait : NativeOpTrait<"::mlir::tla::trait::MemoryOps">;
def Tla_AllocateTrait : NativeOpTrait<"::mlir::tla::trait::Allocate">;
def Tla_MmadTrait : NativeOpTrait<"::mlir::tla::trait::Mmad">;
def Tla_OpClassBinaryTrait : NativeOpTrait<"::mlir::tla::trait::OpClassBinary">;
def Tla_OpClassControlTrait : NativeOpTrait<"::mlir::tla::trait::OpClassControl">;
def Tla_OpClassMemoryTrait : NativeOpTrait<"::mlir::tla::trait::OpClassMemory">;
def Tla_OpClassUnaryTrait : NativeOpTrait<"::mlir::tla::trait::OpClassUnary">;
def Tla_PipeBarrierTrait : NativeOpTrait<"::mlir::tla::trait::PipeBarrier">;
def Tla_PipeCubeOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeCubeOps">;
def Tla_PipeDependentOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeDependentOps">;
def Tla_PipeIntermediateOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeIntermediateOps">;
def Tla_PipeScalarOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeScalarOps">;
def Tla_PipeVectorOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeVectorOps">;
def Tla_RegionControlOpsTrait : NativeOpTrait<"::mlir::tla::trait::RegionControlOps">;
def Tla_RegionOpsTrait : NativeOpTrait<"::mlir::tla::trait::RegionOps">;
def Tla_ScalarControlOpsTrait : NativeOpTrait<"::mlir::tla::trait::ScalarControlOps">;
def Tla_SetFlagTrait : NativeOpTrait<"::mlir::tla::trait::SetFlag">;
def Tla_StoreTrait : NativeOpTrait<"::mlir::tla::trait::Store">;
def Tla_SubtileTrait : NativeOpTrait<"::mlir::tla::trait::Subtile">;
def Tla_TernaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::TernaryOps">;
def Tla_TileTrait : NativeOpTrait<"::mlir::tla::trait::Tile">;
def Tla_TileCountTrait : NativeOpTrait<"::mlir::tla::trait::TileCount">;
def Tla_UnaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::UnaryOps">;
def Tla_VectorTrait : NativeOpTrait<"::mlir::tla::trait::Vector">;
def Tla_VectorBinaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::VectorBinaryOps">;
def Tla_WaitFlagTrait : NativeOpTrait<"::mlir::tla::trait::WaitFlag">;

def Tla_BlockDimOp : Tla_Op<"arch.block_dim", [Pure]> {
  let summary = "Block dimension";
  let description = [{
    Return the grid block dimension.
  }];
  let results = (outs Index:$dim);
  let assemblyFormat = [{
    attr-dict `->` type($dim)
  }];
}

def Tla_BlockIdxOp : Tla_Op<"arch.block_idx", [Pure]> {
  let summary = "Block index";
  let description = [{
    Return the current block index.
  }];
  let results = (outs Index:$index);
  let assemblyFormat = [{
    attr-dict `->` type($index)
  }];
}

def Tla_SubBlockIdxOp : Tla_Op<"arch.sub_block_idx", [Pure]> {
  let summary = "Sub-block index";
  let description = [{
    Return the current sub-block index.
  }];
  let results = (outs Index:$index);
  let assemblyFormat = [{
    attr-dict `->` type($index)
  }];
}

def Tla_IntToPtrOp : Tla_Op<"inttoptr", [Pure]> {
  let summary = "Integer SSA to tla.ptr (``tla.inttoptr`` parity)";
  let description = [{
    Reinterprets the bit pattern of an integer-like SSA value as a pointer of the
    given ``!tla.ptr`` result type. This op is carried in Tla IR for frontend parity;
    lowering to downstream representations is not part of the TlaIR emission path.
  }];
  let arguments = (ins AnyType:$addr);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    $addr attr-dict `:` type($addr) `->` qualified(type($ptr))
  }];
}

def Tla_RecastPtrOp : Tla_Op<"recast_ptr", [Pure]> {
  let summary = "Recast pointer element type (``tla.recast_ptr`` parity, dtype-only)";
  let description = [{
    Reinterprets a ``!tla.ptr`` SSA as a different logical pointee type while preserving
    address space and alignment.
    **No swizzle** metadata (Tla does not model swizzle on ``!tla.ptr``).
  }];
  let arguments = (ins Tla_PtrType:$src);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` qualified(type($ptr))
  }];
}

def Tla_BroadcastOp : Tla_Op<"broadcast", [Pure]> {
  let summary = "Broadcast";
  let description = [{
    Broadcast a scalar value into a vector tile value using a packed ``!tla.shape`` extent.
  }];
  let arguments = (ins Tla_ValueType:$value, Tla_ShapeType:$shape);
  let results = (outs Tla_ValueType:$result);
  let assemblyFormat = [{
    $value `,` $shape attr-dict `:` qualified(type($value)) `,` qualified(type($shape)) `->` qualified(type($result))
  }];
}

def Tla_FullOp : Tla_Op<"full", [Pure]> {
  let summary = "Fill vector lanes with scalar";
  let description = [{
    Broadcast a scalar value into the vector helper register type selected by the
    enclosing vector lowering context.
  }];
  let arguments = (ins AnyTypeOf<[I1, I8, I16, I32, I64, BF16, F16, F32]>:$value);
  let results = (outs Tla_TensorType:$result);
  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

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
    Copy between tensors/views in different address spaces.
  }];
  let arguments = (ins
    Tla_TensorType:$dst, Tla_TensorType:$src,
    Optional<AnyTypeOf<[Tla_CopyL0C2DstParamsType]>>:$params);
  let assemblyFormat = [{
    $dst `,` $src (`,` $params^)? attr-dict `:` qualified(type($dst)) `,` qualified(type($src)) (`,` qualified(type($params))^)?
  }];
}

def Tla_CrossCoreSetFlagOp : Tla_Op<"cross_core_set_flag", []> {
  let summary = "Cross-core flag set";
  let description = [{
    Set a cross-core synchronization flag.
  }];
  let arguments = (ins Tla_CrossFlagType:$flag);
  let assemblyFormat = [{
    $flag attr-dict `:` qualified(type($flag))
  }];
}

def Tla_CrossCoreWaitFlagOp : Tla_Op<"cross_core_wait_flag", []> {
  let summary = "Cross-core flag wait";
  let description = [{
    Wait on a cross-core synchronization flag.
  }];
  let arguments = (ins Tla_CrossFlagType:$flag);
  let assemblyFormat = [{
    $flag attr-dict `:` qualified(type($flag))
  }];
}

def Tla_CrossFlagOp : Tla_Op<"cross_flag", [Pure]> {
  let summary = "Cross-block flag literal";
  let description = [{
    Materialize a cross-core synchronization flag.
  }];
  let arguments = (ins StrAttr:$name, Tla_PipeAttr:$src_pipe, Tla_PipeAttr:$dst_pipe,
                       DefaultValuedOptionalAttr<I64Attr, "2">:$mode);
  let results = (outs Tla_CrossFlagType:$flag);
  let assemblyFormat = [{
    $name attr-dict `->` qualified(type($flag))
  }];
}

def Tla_CubeOp : Tla_Op<"cube", [NoTerminator, SingleBlock]> {
  let summary = "Cube region";
  let description = [{
    Launch a cube region for matrix multiply and related operations.
  }];
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $body attr-dict
  }];
}

def Tla_FlagOp : Tla_Op<"flag", [Pure]> {
  let summary = "Flag literal";
  let description = [{
    Materialize a static synchronization flag.
  }];
  let arguments = (ins StrAttr:$name, Tla_PipeAttr:$src_pipe, Tla_PipeAttr:$dst_pipe);
  let results = (outs Tla_FlagType:$flag);
  let assemblyFormat = [{
    $name attr-dict `->` qualified(type($flag))
  }];
}

def Tla_FmaOp : Tla_Op<"fma", [Pure]> {
  let summary = "FMA";
  let description = [{
    Fused multiply-add over register values.
  }];
  let arguments = (ins Tla_ValueType:$lhs, Tla_ValueType:$rhs, Tla_ValueType:$acc);
  let results = (outs Tla_ValueType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs)) `,` qualified(type($acc)) `->` qualified(type($result))
  }];
}

def Tla_FuncOp : Tla_Op<"func", []> {
  let summary = "Tla kernel function";
  let description = [{
    Defines a Tla kernel entry point. The body is an isolated region with Tla IR operations.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$function_type);
  let regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = 1;
}

def Tla_LoadOp : Tla_Op<"load", []> {
  let summary = "Load";
  let description = [{
    Load a tensor tile into vector SSA.
  }];
  let arguments = (ins Tla_TensorType:$source);
  let results = (outs Tla_TensorType:$result);
  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

// Scratch alloc uses a static ``size_bytes`` attribute (``i64``) on ``tla.alloc_ptr``.
def Tla_AllocPtrOp : Tla_Op<"alloc_ptr", [Pure]> {
  let summary = "Static on-chip scratch bytes";
  let description = [{
    Reserves ``size_bytes`` bytes of on-chip scratch. Address space and element alignment appear
    **only** on the result ``!tla.ptr<pointee, addrspace, align>`` type (no duplicate
    ``mem_space`` / ``alignment`` attributes). HIVM lowering materializes allocator-backed
    pointer uses as ``hivm.hir.pointer_cast`` plus ``tla.hivm_memref_as_ptr`` with assigned offsets.
  }];
  let arguments = (ins I64Attr:$size_bytes);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    `{` `size_bytes` `=` $size_bytes `}` attr-dict `->` qualified(type($ptr))
  }];
  let hasVerifier = 1;
}

def Tla_HivmMemrefAsPtrOp : Tla_Op<"hivm_memref_as_ptr", [Pure]> {
  let summary = "Bridge HIVM memref from pointer_cast to !tla.ptr";
  let description = [{
    Emitted by ``tla-alloc-ptr-to-hivm-pointer-cast``: wraps the rank-1 memref produced by
    ``hivm.hir.pointer_cast`` as a ``!tla.ptr`` for remaining Tla IR. Named op instead of
    ``builtin.unrealized_conversion_cast`` so failures and ``mlir-opt`` dumps identify this edge.
    Lowered away in ``tla-lower-to-std`` when tensor descriptors materialize the memref base.
    The operand is spelled as a generic type in ODS (avoids pulling ``BuiltinTypes.td`` into
    ``Tla.td``); ``HivmMemrefAsPtrOp::verify`` requires a rank-1 ``memref``.
  }];
  let arguments = (ins AnyType:$memref);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = "$memref attr-dict `:` type($memref) `->` qualified(type($ptr))";
  let hasVerifier = 1;
}

def Tla_MakeTensorLikeOp : Tla_Op<"make_tensor_like", []> {
  let summary = "Make tensor like";
  let description = [{
    Reconstruct a typed local tensor from a raw allocation pointer and a reference tensor.
  }];
  let arguments = (ins Tla_PtrType:$ptr, Tla_TensorType:$like, StrAttr:$layoutTag);
  let results = (outs Tla_TensorType:$tile);
  let assemblyFormat = [{
    $ptr `like` $like `layoutTag` `(` $layoutTag `)` attr-dict `:` qualified(type($ptr)) `,` qualified(type($like)) `->` qualified(type($tile))
  }];
}

def Tla_MakeTensorOp : Tla_Op<"make_tensor", [Pure]> {
  let summary = "Construct a tensor from an explicit pointer, layout, and coord";
  let description = [{
    Build a `!tla.tensor<layout, coord, ptr>` from explicit operands. Unlike
    `tla.make_tensor_like` (which clones layout/coord from a reference tensor), this op
    takes the layout and coord directly as operands, mirroring the tensor type's
    `(layout, coord, ptr)` structure. Address space and element type come from the `ptr`
    operand's `!tla.ptr` type; the layout tag, shape, stride, and origin come from the
    `!tla.layout` operand.
  }];
  let arguments = (ins Tla_PtrType:$ptr, Tla_LayoutType:$layout, Tla_CoordType:$coord);
  let results = (outs Tla_TensorType:$result);
  let assemblyFormat = [{
    $ptr `,` $layout `,` $coord attr-dict `:`
    qualified(type($ptr)) `,` qualified(type($layout)) `,` qualified(type($coord)) `->` qualified(type($result))
  }];
}

def Tla_MmadOp : Tla_Op<"mmad", []> {
  let summary = "Matrix multiply-accumulate";
  let description = [{
    Perform a matrix multiply-accumulate on tensor views.
  }];
  let arguments = (ins Tla_TensorType:$acc, Tla_TensorType:$lhs, Tla_TensorType:$rhs, I1:$init_c, I64:$unit_flag);
  let assemblyFormat = [{
    $acc `,` $lhs `,` $rhs `,` $init_c `,` $unit_flag attr-dict `:` 
    qualified(type($acc)) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `,` type($init_c) `,` type($unit_flag)
  }];
}

def Tla_PipeBarrierOp : Tla_Op<"pipe_barrier", []> {
  let summary = "Pipe barrier";
  let description = [{
    Apply a synchronization barrier to the specified pipe.
  }];
  let arguments = (ins Tla_PipeAttr:$pipe);
  let assemblyFormat = [{
    `[` $pipe `]` attr-dict
  }];
}

def Tla_MutexOp : Tla_Op<"mutex", [Pure]> {
  let summary = "Mutex literal";
  let description = [{
    Materialize a static mutex associated with a semantic resource.
    ``id = -1`` means the compiler may allocate the hardware MutexID later;
    ``0`` through ``31`` are user-provided hardware MutexIDs.
  }];
  let arguments = (ins StrAttr:$resource, I64Attr:$id);
  let results = (outs Tla_MutexType:$mutex);
  let assemblyFormat = [{
    $resource attr-dict `->` qualified(type($mutex))
  }];
}

def Tla_MutexLockOp : Tla_Op<"mutex_lock", []> {
  let summary = "Mutex lock";
  let description = [{
    Acquire a mutex from the specified pipe.
  }];
  let arguments = (ins Tla_MutexType:$mutex, Tla_PipeAttr:$pipe);
  let assemblyFormat = [{
    $mutex `[` $pipe `]` `:` qualified(type($mutex)) attr-dict
  }];
}

def Tla_MutexUnlockOp : Tla_Op<"mutex_unlock", []> {
  let summary = "Mutex unlock";
  let description = [{
    Release a mutex from the specified pipe.
  }];
  let arguments = (ins Tla_MutexType:$mutex, Tla_PipeAttr:$pipe);
  let assemblyFormat = [{
    $mutex `[` $pipe `]` `:` qualified(type($mutex)) attr-dict
  }];
}

def Tla_MakeShapeOp : Tla_Op<"make_shape", [Pure]> {
  let summary = "Shape value";
  let description = [{
    Build a ``!tla.shape`` from operands for each dynamic dimension; rank and static sizes are fixed by the result type.
  }];
  let arguments = (ins Variadic<Index>:$dyn_elems);
  let results = (outs Tla_ShapeType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Tla_MakeCoordOp : Tla_Op<"make_coord", [Pure]> {
  let summary = "Coord value";
  let description = [{
    Build a ``!tla.coord`` from operands for each dynamic leaf; structure and static leaves are fixed by the result type.
  }];
  let arguments = (ins Variadic<Index>:$dyn_elems);
  let results = (outs Tla_CoordType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Tla_MakeStrideOp : Tla_Op<"make_stride", [Pure]> {
  let summary = "Stride value";
  let description = [{
    Build a ``!tla.stride`` from operands for each dynamic leaf; structure and static leaves are fixed by the result type.
  }];
  let arguments = (ins Variadic<Index>:$dyn_elems);
  let results = (outs Tla_StrideType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Tla_MakeLayoutOp : Tla_Op<"make_layout", [Pure]> {
  let summary = "Layout value";
  let description = [{
    Combine a packed ``!tla.shape`` and ``!tla.stride`` into a ``!tla.layout``, analogous to ``tla.make_layout(shape, stride)``.
    The result type encodes tuple-tree structure as ``!tla.layout<!tla.shape<shape_repr>, !tla.stride<stride_repr>, layout_tag>``, or
    ``!tla.layout<!tla.shape<shape_repr>, !tla.stride<stride_repr>, !tla.shape<origin_shape_repr>, layout_tag>`` when an explicit origin shape is used.

    Optional ``origin_shape`` records the logical extent (e.g. pre-padding matrix size). When omitted,
    consumers should treat the layout's logical origin shape as identical to ``shape``.
  }];
  let arguments = (ins Tla_ShapeType:$shape,
                       Tla_StrideType:$stride,
                       Optional<Tla_ShapeType>:$origin_shape,
                       OptionalAttr<StrAttr>:$layoutTag);
  let results = (outs Tla_LayoutType:$result);
  let assemblyFormat = [{
    $shape `,` $stride (`origin` $origin_shape^)? attr-dict `:`
    qualified(type($shape)) `,` qualified(type($stride))
    (`origin` qualified(type($origin_shape))^)? `->` qualified(type($result))
  }];
}

def Tla_ReturnOp : Tla_Op<"return", [Terminator]> {
  let summary = "Return";
  let description = [{
    Terminator for tla.func regions.
  }];
  let assemblyFormat = [{
    attr-dict
  }];
}

def Tla_SetFlagOp : Tla_Op<"set_flag", []> {
  let summary = "Set flag";
  let description = [{
    Signal a synchronization flag from one pipe to another.
  }];
  let arguments = (ins Tla_FlagType:$flag);
  let assemblyFormat = [{
    $flag `:` qualified(type($flag)) attr-dict
  }];
}

def Tla_StoreOp : Tla_Op<"store", []> {
  let summary = "Store";
  let description = [{
    Register-fragment store marker on a tensor tile. An optional mask predicates
    which lanes are written (masked-out lanes are left untouched).
  }];
  let arguments = (ins Tla_TensorType:$dest, Tla_TensorType:$source,
                       Optional<Tla_MaskType>:$mask);
  let assemblyFormat = [{
    $dest `,` $source (`mask` $mask^)? attr-dict `:` type($dest) `,` type($source)
    (`mask` qualified(type($mask))^)?
  }];
}

def Tla_CreateMaskOp : Tla_Op<"create_mask", [Pure]> {
  let summary = "Create a vector predicate mask from a fixed pattern";
  let description = [{
    Create an element-wise predicate mask from a fixed AVE ``pge`` pattern, e.g.
    ``ALL``, ``ALLF``, ``VL8`` (lowest 8 lanes), ``H`` (lowest half), ``Q``
    (lowest quarter), ``M3``/``M4`` (multiples of 3/4).

    ``dtype`` fixes the mask lane count (256 bytes / dtype size) and must match
    the enclosing vector region's element width.
  }];
  let arguments = (ins StrAttr:$pattern, TypeAttr:$dtype);
  let results = (outs Tla_MaskType:$result);
  let assemblyFormat = [{
    attr-dict `:` qualified(type($result))
  }];
}

def Tla_UpdateMaskOp : Tla_Op<"update_mask", [Pure]> {
  let summary = "Tail predicate mask plus the remaining element count";
  let description = [{
    Tail-loop helper mapping to AVE ``plt``. Given the number of elements still
    to process (``true_shape``), produce a predicate mask (lane ``i`` is active
    iff ``i < true_shape``, saturating to all-true once ``true_shape`` reaches
    the lane count) together with the remaining count after this iteration,
    ``new_true_shape = true_shape - lanes`` where ``lanes = 256 bytes / dtype
    size`` (= 64 for f32). Thread ``new_true_shape`` back through the loop's
    carried values to mask successive chunks, including the partial tail.

    ``dtype`` fixes the lane count and therefore the decrement; it must match the
    enclosing vector region's element width.
  }];
  let arguments = (ins Index:$true_shape, TypeAttr:$dtype);
  let results = (outs Tla_MaskType:$mask, Index:$new_true_shape);
  let assemblyFormat = [{
    $true_shape `,` $dtype attr-dict `:` qualified(type($mask)) `,` type($new_true_shape)
  }];
}

def Tla_TileViewOp : Tla_Op<"tile_view", []> {
  let summary = "Tile view";
  let description = [{
    Create a tensor view over either a builtin memref buffer or an existing tla.tensor
    using packed ``!tla.shape`` and ``!tla.coord`` values.
  }];
  let arguments = (ins AnyTypeOf<[Tla_TensorType, AnyMemRef]>:$source, Tla_ShapeType:$shape, Tla_CoordType:$coord, OptionalAttr<StrAttr>:$layouttag);
  let results = (outs Tla_TensorType:$view);
  let assemblyFormat = [{
    $source `,` $shape `,` $coord attr-dict `:` qualified(type($source)) `,` qualified(type($shape)) `,` qualified(type($coord)) `->` qualified(type($view))
  }];
}

class Tla_VectorBinaryOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    An optional mask predicates which lanes are computed; masked-out lanes are
    undefined/zeroed.
  }];
  let arguments = (ins Tla_TensorType:$lhs, Tla_TensorType:$rhs,
                       Optional<Tla_MaskType>:$mask);
  let results = (outs Tla_TensorType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`mask` $mask^)? attr-dict `:`
    type($lhs) `,` type($rhs) (`mask` qualified(type($mask))^)? `->` type($result)
  }];
}

def Tla_AddOp : Tla_VectorBinaryOp<"add",
    "Vector add for tensor fragments",
    "Element-wise add on tensor fragments / vector tiles.">;

def Tla_SubOp : Tla_VectorBinaryOp<"sub",
    "Vector subtract for tensor fragments",
    "Element-wise subtract on tensor fragments / vector tiles.">;

def Tla_MulOp : Tla_VectorBinaryOp<"mul",
    "Vector multiply for tensor fragments",
    "Element-wise multiply on tensor fragments / vector tiles.">;

def Tla_DivOp : Tla_VectorBinaryOp<"div",
    "Vector divide for tensor fragments",
    "Element-wise divide on tensor fragments / vector tiles.">;

def Tla_MaxOp : Tla_VectorBinaryOp<"max",
    "Vector maximum for tensor fragments",
    "Element-wise maximum on tensor fragments / vector tiles.">;

def Tla_MinOp : Tla_VectorBinaryOp<"min",
    "Vector minimum for tensor fragments",
    "Element-wise minimum on tensor fragments / vector tiles.">;

def Tla_WhereOp : Tla_Op<"where", [Pure]> {
  let summary = "Vector select for tensor fragments";
  let description = [{
    Element-wise select on tensor fragments / vector tiles. For each lane the
    result holds the corresponding lane of ``x`` where the predicate mask is
    active and ``y`` elsewhere. ``x`` and ``y`` must share the same tensor type.
  }];
  let arguments = (ins Tla_MaskType:$mask, Tla_TensorType:$x, Tla_TensorType:$y);
  let results = (outs Tla_TensorType:$result);
  let assemblyFormat = [{
    $mask `,` $x `,` $y attr-dict `:`
    qualified(type($mask)) `,` type($x) `,` type($y) `->` type($result)
  }];
}

class Tla_VectorScalarBinaryOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    The right-hand side is a scalar SSA value. An optional mask predicates which
    lanes are computed; masked-out lanes are undefined/zeroed.
  }];
  let arguments = (ins Tla_TensorType:$lhs, Tla_ScalarValueType:$rhs,
                       Optional<Tla_MaskType>:$mask);
  let results = (outs Tla_TensorType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`mask` $mask^)? attr-dict `:`
    type($lhs) `,` type($rhs) (`mask` qualified(type($mask))^)? `->` type($result)
  }];
}

def Tla_AddsOp : Tla_VectorScalarBinaryOp<"adds",
    "Vector-scalar add for tensor fragments",
    "Element-wise add of a scalar to tensor fragments / vector tiles.">;

def Tla_SubsOp : Tla_VectorScalarBinaryOp<"subs",
    "Vector-scalar subtract for tensor fragments",
    "Element-wise subtract of a scalar from tensor fragments / vector tiles.">;

def Tla_MulsOp : Tla_VectorScalarBinaryOp<"muls",
    "Vector-scalar multiply for tensor fragments",
    "Element-wise multiply of tensor fragments / vector tiles by a scalar.">;

def Tla_MaxsOp : Tla_VectorScalarBinaryOp<"maxs",
    "Vector-scalar maximum for tensor fragments",
    "Element-wise maximum of tensor fragments / vector tiles and a scalar.">;

def Tla_MinsOp : Tla_VectorScalarBinaryOp<"mins",
    "Vector-scalar minimum for tensor fragments",
    "Element-wise minimum of tensor fragments / vector tiles and a scalar.">;

def Tla_DivsOp : Tla_VectorScalarBinaryOp<"divs",
    "Vector-scalar divide for tensor fragments",
    "Element-wise divide of tensor fragments / vector tiles by a scalar.">;

def Tla_VectorOp : Tla_Op<"vector", [NoTerminator, SingleBlock]> {
  let summary = "Vector region";
  let description = [{
    Launch a vector region for post-processing and epilogue ops.
  }];
  let regions = (region AnyRegion:$body);
}

def Tla_VecFuncOp : Tla_Op<"vec.func", [NoTerminator, SingleBlock]> {
  let summary = "Vector compute function region";
  let description = [{
    Group register-resident vector compute ops inside a vector region.
  }];
  let regions = (region AnyRegion:$body);
}

def Tla_WaitFlagOp : Tla_Op<"wait_flag", []> {
  let summary = "Wait flag";
  let description = [{
    Wait on a synchronization flag from one pipe to another.
  }];
  let arguments = (ins Tla_FlagType:$flag);
  let assemblyFormat = [{
    $flag `:` qualified(type($flag)) attr-dict
  }];
}
