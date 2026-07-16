#include "Dialect/Tla/IR/TlaOps.h"
#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringSwitch.h"

#define GET_OP_CLASSES
#include "tla/Ops.cpp.inc"

namespace tla {

template <typename TreeType>
static mlir::LogicalResult getIndexTreeLeavesForVerify(
    mlir::Operation *op, TreeType treeType,
    llvm::SmallVectorImpl<int64_t> &leaves, llvm::StringRef name) {
  if (failed(getTlaIndexTreeLeaves(treeType.getTree(), leaves)))
    return op->emitOpError() << "failed to decode " << name;
  return mlir::success();
}

static bool isSupportedCmpElementType(mlir::Type elementType) {
  if (elementType.isF16() || elementType.isF32())
    return true;
  auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType);
  return intType && (intType.isSignless() || intType.isUnsigned()) &&
         intType.getWidth() == 32;
}

static bool isSupportedCmpMode(llvm::StringRef mode) {
  return llvm::StringSwitch<bool>(mode)
      .Cases("lt", "le", "gt", "ge", true)
      .Cases("eq", "ne", true)
      .Default(false);
}

mlir::LogicalResult AllocPtrOp::verify() {
  auto resTy = llvm::dyn_cast<PtrType>(getResult().getType());
  if (!resTy)
    return emitOpError("result must be !tla.ptr");
  if (resTy.getAlignment() == 0)
    return emitOpError("result pointer alignment must be positive");
  auto ms = resTy.getAddrspace();
  if (ms == AddressSpace::generic || ms == AddressSpace::gm)
    return emitOpError("alloc_ptr requires on-chip !tla.ptr (l1, l0a, l0b, l0c, ub)");
  int64_t sizeBytes = getSizeBytesAttr().getInt();
  if (sizeBytes <= 0)
    return emitOpError("size_bytes must be positive");
  int64_t elemBytes = getByteSizeOfFixedWidthScalarType(resTy.getPointee());
  if (elemBytes <= 0)
    return emitOpError("alloc_ptr pointee must be a fixed-width scalar type");
  if (sizeBytes % elemBytes != 0)
    return emitOpError("size_bytes must be a multiple of result pointee type size");
  return mlir::success();
}

mlir::LogicalResult TensorPtrOp::verify() {
  auto resTy = mlir::dyn_cast<PtrType>(getPtr().getType());
  if (!resTy)
    return emitOpError("result must be !tla.ptr");
  if (auto tensorTy = mlir::dyn_cast<TlaTensorType>(getSrc().getType())) {
    if (tensorTy.getPtr() != resTy)
      return emitOpError("result ptr type must match the tensor's embedded pointer type");
  }
  return mlir::success();
}

mlir::LogicalResult PtrAddOp::verify() {
  auto srcTy = mlir::dyn_cast<PtrType>(getPtr().getType());
  auto resTy = mlir::dyn_cast<PtrType>(getResult().getType());
  if (!srcTy || !resTy)
    return emitOpError("operands and result must be !tla.ptr");
  if (srcTy.getPointee() != resTy.getPointee())
    return emitOpError("result pointee type must match the source pointer's pointee");
  if (srcTy.getAddrspace() != resTy.getAddrspace())
    return emitOpError("result address space must match the source pointer's address space");
  return mlir::success();
}

// Walk the enclosing ops looking for an ancestor of type AncestorOp. The
// required region may be several levels up (e.g. a compute op nested inside a
// scf.for loop inside a tla.vec.func), so this checks all transitive parents
// rather than just the immediate one.
template <typename AncestorOp> static bool hasEnclosing(mlir::Operation *op) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (mlir::isa<AncestorOp>(parent))
      return true;
  return false;
}

// The region-wrapper requirement is a frontend/authoring constraint, enforced
// while ops still live in the tla.func container. Once tla-lower-func lowers
// tla.func to func.func -- and tla-vector-region / TlaSplitMixedFuncPass
// inline the frontend tla.cube / tla.vector wrappers into the resulting
// func.func (carrying the AIC/AIV/MIX core context on function attributes such
// as hivm.func_core_type / hivm.part_of_mix / hacc.entry rather than a lexical
// region) -- the lexical wrapper is legitimately gone. Ops already inside a
// lowered func.func are therefore exempt; the constraint is fully enforced at
// parse time and in the frontend, where ops are still under tla.func.
static bool isInLoweredFunc(mlir::Operation *op) {
  return op->getParentOfType<mlir::func::FuncOp>() != nullptr;
}

template <typename AncestorOp> static bool hasEnclosingRegion(mlir::Operation *op) {
  return hasEnclosing<AncestorOp>(op) || isInLoweredFunc(op);
}

mlir::LogicalResult MmadOp::verify() {
  if (!hasEnclosingRegion<CubeOp>(getOperation()))
    return emitOpError("must be nested inside a tla.cube region");
  return mlir::success();
}

mlir::LogicalResult VecFuncOp::verify() {
  if (!hasEnclosingRegion<VectorOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vector region");
  return mlir::success();
}

mlir::LogicalResult InterleaveOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");

  if (getSrc0().getType() != getSrc1().getType())
    return emitOpError("requires src0 and src1 to have identical !tla.tensor types");

  if (getDst0().getType() != getSrc0().getType() ||
      getDst1().getType() != getSrc0().getType())
    return emitOpError("requires dst0/dst1 to have the same !tla.tensor type as inputs");

  return mlir::success();
}

mlir::LogicalResult DeInterleaveOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");

  if (getSrc0().getType() != getSrc1().getType())
    return emitOpError("requires src0 and src1 to have identical !tla.tensor types");

  if (getDst0().getType() != getSrc0().getType() ||
      getDst1().getType() != getSrc0().getType())
    return emitOpError("requires dst0/dst1 to have the same !tla.tensor type as inputs");

  return mlir::success();
}

// Vector compute ops (element-wise vector-vector and vector-scalar arithmetic)
// must live inside a tla.vec.func region.
#define TLA_VERIFY_IN_VEC_FUNC(OpTy)                                            \
  mlir::LogicalResult OpTy::verify() {                                          \
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))                        \
      return emitOpError("must be nested inside a tla.vec.func region");        \
    return mlir::success();                                                     \
  }

TLA_VERIFY_IN_VEC_FUNC(AddOp)
TLA_VERIFY_IN_VEC_FUNC(SubOp)
TLA_VERIFY_IN_VEC_FUNC(MulOp)
TLA_VERIFY_IN_VEC_FUNC(DivOp)
TLA_VERIFY_IN_VEC_FUNC(MaxOp)
TLA_VERIFY_IN_VEC_FUNC(MinOp)
TLA_VERIFY_IN_VEC_FUNC(AddsOp)
TLA_VERIFY_IN_VEC_FUNC(SubsOp)
TLA_VERIFY_IN_VEC_FUNC(MulsOp)
TLA_VERIFY_IN_VEC_FUNC(DivsOp)
TLA_VERIFY_IN_VEC_FUNC(MaxsOp)
TLA_VERIFY_IN_VEC_FUNC(MinsOp)
TLA_VERIFY_IN_VEC_FUNC(LoadOp)
TLA_VERIFY_IN_VEC_FUNC(StoreOp)
TLA_VERIFY_IN_VEC_FUNC(FullOp)
TLA_VERIFY_IN_VEC_FUNC(WhereOp)
TLA_VERIFY_IN_VEC_FUNC(CreateMaskOp)
TLA_VERIFY_IN_VEC_FUNC(UpdateMaskOp)
TLA_VERIFY_IN_VEC_FUNC(MaskNotOp)
TLA_VERIFY_IN_VEC_FUNC(MaskAndOp)
TLA_VERIFY_IN_VEC_FUNC(MaskOrOp)
TLA_VERIFY_IN_VEC_FUNC(MaskXorOp)
TLA_VERIFY_IN_VEC_FUNC(RegNotOp)
TLA_VERIFY_IN_VEC_FUNC(RegAndOp)
TLA_VERIFY_IN_VEC_FUNC(RegOrOp)
TLA_VERIFY_IN_VEC_FUNC(RegXorOp)
TLA_VERIFY_IN_VEC_FUNC(ExpOp)
TLA_VERIFY_IN_VEC_FUNC(LogOp)
TLA_VERIFY_IN_VEC_FUNC(SqrtOp)
TLA_VERIFY_IN_VEC_FUNC(AbsOp)
TLA_VERIFY_IN_VEC_FUNC(NegOp)
TLA_VERIFY_IN_VEC_FUNC(ReduceOp)

#undef TLA_VERIFY_IN_VEC_FUNC

mlir::LogicalResult ArangeOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");
  auto order = getOrderAttr().getValue();
  if (order != "increase" && order != "decrease")
    return emitOpError("unsupported arange order '")
           << order << "'; expected 'increase' or 'decrease'";
  return mlir::success();
}

// Synchronization/mutex/barrier ops must live inside a tla.cube or tla.vector
// region (either core-kind region; not the func-level scope).
#define TLA_VERIFY_IN_CUBE_OR_VECTOR(OpTy)                                     \
  mlir::LogicalResult OpTy::verify() {                                          \
    if (!hasEnclosingRegion<CubeOp>(getOperation()) &&                         \
        !hasEnclosingRegion<VectorOp>(getOperation()))                         \
      return emitOpError(                                                       \
          "must be nested inside a tla.cube or tla.vector region");            \
    return mlir::success();                                                     \
  }

TLA_VERIFY_IN_CUBE_OR_VECTOR(SetFlagOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(WaitFlagOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(CrossCoreSetFlagOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(CrossCoreWaitFlagOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(MutexLockOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(MutexUnlockOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(PipeBarrierOp)

#undef TLA_VERIFY_IN_CUBE_OR_VECTOR

mlir::LogicalResult LocalMemBarOp::verify() {
  auto kind = getBarrierKind();
  if (kind < 0 || kind > 11)
    return emitOpError("barrier_kind ") << kind << " is out of range [0, 11]";
  if (!hasEnclosingRegion<CubeOp>(getOperation()) &&
      !hasEnclosingRegion<VectorOp>(getOperation()))
    return emitOpError(
        "must be nested inside a tla.cube or tla.vector region");
  return mlir::success();
}

mlir::LogicalResult CopyOp::verify() {
  auto srcTy = mlir::dyn_cast<TlaTensorType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<TlaTensorType>(getDst().getType());
  if (!srcTy || !dstTy)
    return mlir::success(); // Operand type verifier handles malformed tensors.
  AddressSpace src = srcTy.getPtr().getAddrspace();
  AddressSpace dst = dstTy.getPtr().getAddrspace();

  // Cube data-path copies: GM->L1, L1->L0A, L1->L0B, L0C->GM, L0C->UB, L1->UB.
  bool cubeRoute = (src == AddressSpace::gm && dst == AddressSpace::l1) ||
                   (src == AddressSpace::l1 && dst == AddressSpace::l0a) ||
                   (src == AddressSpace::l1 && dst == AddressSpace::l0b) ||
                   (src == AddressSpace::l0c && dst == AddressSpace::gm) ||
                   (src == AddressSpace::l0c && dst == AddressSpace::ub) ||
                   (src == AddressSpace::l1 && dst == AddressSpace::ub);
  // Vector staging copies: GM->UB, UB->GM, UB->L1.
  bool vectorRoute = (src == AddressSpace::gm && dst == AddressSpace::ub) ||
                     (src == AddressSpace::ub && dst == AddressSpace::gm) ||
                     (src == AddressSpace::ub && dst == AddressSpace::l1);

  if (cubeRoute && !hasEnclosingRegion<CubeOp>(getOperation()))
    return emitOpError("copy between GM/L1/L0A/L0B/L0C/UB must be nested inside "
                       "a tla.cube region");
  if (vectorRoute && !hasEnclosingRegion<VectorOp>(getOperation()))
    return emitOpError(
        "copy between GM/UB/L1 must be nested inside a tla.vector region");
  return mlir::success();
}

mlir::LogicalResult CmpOp::verify() {
  if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
    return emitOpError("must be nested inside a tla.vec.func region");
  if (!isSupportedCmpMode(getMode()))
    return emitOpError()
           << "mode must be one of lt, le, gt, ge, eq, ne, got \""
           << getMode() << "\"";

  auto lhsType = getLhs().getType();
  auto lhsPtr = lhsType.getPtr();
  mlir::Type lhsElementType = lhsPtr.getPointee();
  if (!isSupportedCmpElementType(lhsElementType))
    return emitOpError() << "unsupported compare element type "
                         << lhsElementType;

  auto rhsType = getRhs().getType();
  auto rhsTensorType = mlir::dyn_cast<::tla::TlaTensorType>(rhsType);
  if (!rhsTensorType) {
    if (rhsType != lhsElementType)
      return emitOpError() << "scalar operand must have element type "
                           << lhsElementType << ", got " << rhsType;
    return mlir::success();
  }

  auto rhsPtr = rhsTensorType.getPtr();
  mlir::Type rhsElementType = rhsPtr.getPointee();
  if (lhsElementType != rhsElementType)
    return emitOpError() << "operands must have the same element type, got "
                         << lhsElementType << " and " << rhsElementType;

  return mlir::success();
}

static mlir::ParseResult parseIndexTreeValueOp(mlir::OpAsmParser &parser,
                                               mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> dynElemsOperands;
  llvm::SMLoc dynElemsOperandsLoc = parser.getCurrentLocation();
  mlir::Type resultType;

  if (parser.parseOperandList(dynElemsOperands))
    return mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();
  if (parser.parseArrow())
    return mlir::failure();
  if (parser.parseType(resultType))
    return mlir::failure();

  result.addTypes(resultType);
  if (parser.resolveOperands(dynElemsOperands, parser.getBuilder().getIndexType(),
                             dynElemsOperandsLoc, result.operands))
    return mlir::failure();
  return mlir::success();
}

template <typename OpTy> static void printIndexTreeValueOp(OpTy op, mlir::OpAsmPrinter &printer) {
  if (!op.getDynElems().empty()) {
    printer << ' ';
    printer << op.getDynElems();
  }
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " -> ";
  printer.printType(op.getResult().getType());
}

mlir::ParseResult MakeShapeOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  return parseIndexTreeValueOp(parser, result);
}

void MakeShapeOp::print(mlir::OpAsmPrinter &printer) { printIndexTreeValueOp(*this, printer); }

mlir::ParseResult MakeCoordOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  return parseIndexTreeValueOp(parser, result);
}

void MakeCoordOp::print(mlir::OpAsmPrinter &printer) { printIndexTreeValueOp(*this, printer); }

mlir::ParseResult MakeStrideOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  return parseIndexTreeValueOp(parser, result);
}

void MakeStrideOp::print(mlir::OpAsmPrinter &printer) { printIndexTreeValueOp(*this, printer); }

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::StringAttr symNameAttr;
  if (parser.parseSymbolName(symNameAttr, "sym_name", result.attributes))
    return llvm::failure();

  llvm::SmallVector<mlir::OpAsmParser::Argument, 4> arguments;
  if (parser.parseArgumentList(arguments, mlir::OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return llvm::failure();

  llvm::SmallVector<mlir::Type, 4> inputTypes;
  inputTypes.reserve(arguments.size());
  for (const auto &arg : arguments) {
    if (!arg.type)
      return parser.emitError(arg.ssaName.location) << "expected type for function argument";
    inputTypes.push_back(arg.type);
  }

  llvm::SmallVector<mlir::Type, 2> resultTypes;
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return llvm::failure();

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return llvm::failure();

  auto functionType = parser.getBuilder().getFunctionType(inputTypes, resultTypes);
  result.attributes.set("function_type", mlir::TypeAttr::get(functionType));

  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, arguments))
    return llvm::failure();

  return llvm::success();
}

void FuncOp::print(mlir::OpAsmPrinter &printer) {
  printer << " @" << getSymName() << "(";

  auto fnType = llvm::dyn_cast<mlir::FunctionType>(getFunctionType());
  llvm::ArrayRef<mlir::Type> inputs = fnType ? fnType.getInputs() : llvm::ArrayRef<mlir::Type>{};
  mlir::Block *entry = getBody().empty() ? nullptr : &getBody().front();

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i)
      printer << ", ";
    if (entry && i < entry->getNumArguments()) {
      printer << entry->getArgument(i);
    } else {
      printer << "%arg" << i;
    }
    printer << ": " << inputs[i];
  }
  printer << ")";

  if (fnType && !fnType.getResults().empty()) {
    printer << " -> (";
    for (size_t i = 0; i < fnType.getResults().size(); ++i) {
      if (i)
        printer << ", ";
      printer.printType(fnType.getResults()[i]);
    }
    printer << ")";
  }

  printer.printOptionalAttrDictWithKeyword((*this)->getAttrs(), {"sym_name", "function_type"});
  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

mlir::LogicalResult ScalarLoadOp::verify() {
  if (!mlir::isa<TlaTensorType, mlir::MemRefType>(getSource().getType()))
    return emitOpError("source must be !tla.tensor or builtin memref");

  mlir::Type expected;
  size_t rank = 0;
  if (auto srcTy = mlir::dyn_cast<TlaTensorType>(getSource().getType())) {
    if (srcTy.getPtr().getAddrspace() != AddressSpace::gm)
      return emitOpError("source !tla.tensor must be in gm address space");
    auto layoutTag = srcTy.getLayout().getLayoutTag();
    if (layoutTag != LayoutTag::row_major && layoutTag != LayoutTag::column_major)
      return emitOpError("source !tla.tensor layout must be row_major or column_major");
    expected = srcTy.getPtr().getPointee();
    llvm::SmallVector<int64_t, 4> shapeLeaves;
    if (failed(getIndexTreeLeavesForVerify(getOperation(), srcTy.getLayout().getShape(),
                                           shapeLeaves, "shape")))
      return mlir::failure();
    rank = shapeLeaves.size();
  } else {
    auto memrefTy = mlir::cast<mlir::MemRefType>(getSource().getType());
    expected = memrefTy.getElementType();
    rank = static_cast<size_t>(memrefTy.getRank());
  }

  if (getResult().getType() != expected)
    return emitOpError("result type must match tensor element type, expected ")
           << expected << ", got " << getResult().getType();

  auto indexCount = getIndices().size();
  // Indices must match the logical rank: no row-omitted shorthand for rank-2.
  if (!((rank == 1 && indexCount == 1) || (rank == 2 && indexCount == 2)))
    return emitOpError(
        "scalar_load expects rank-1/2 source with matching indices (rank-1: 1; rank-2: 2)");
  for (mlir::Value idx : getIndices()) {
    if (!idx.getType().isIndex())
      return emitOpError("indices must be index-typed");
  }
  return mlir::success();
}

mlir::LogicalResult ScalarStoreOp::verify() {
  if (!mlir::isa<TlaTensorType, mlir::MemRefType>(getDest().getType()))
    return emitOpError("dest must be !tla.tensor or builtin memref");

  mlir::Type expected;
  size_t rank = 0;
  if (auto destTy = mlir::dyn_cast<TlaTensorType>(getDest().getType())) {
    if (destTy.getPtr().getAddrspace() != AddressSpace::gm)
      return emitOpError("dest !tla.tensor must be in gm address space");
    auto layoutTag = destTy.getLayout().getLayoutTag();
    if (layoutTag != LayoutTag::row_major && layoutTag != LayoutTag::column_major)
      return emitOpError("dest !tla.tensor layout must be row_major or column_major");
    expected = destTy.getPtr().getPointee();
    llvm::SmallVector<int64_t, 4> shapeLeaves;
    if (failed(getIndexTreeLeavesForVerify(getOperation(), destTy.getLayout().getShape(),
                                           shapeLeaves, "shape")))
      return mlir::failure();
    rank = shapeLeaves.size();
  } else {
    auto memrefTy = mlir::cast<mlir::MemRefType>(getDest().getType());
    expected = memrefTy.getElementType();
    rank = static_cast<size_t>(memrefTy.getRank());
  }

  if (getValue().getType() != expected)
    return emitOpError("value type must match tensor element type, expected ")
           << expected << ", got " << getValue().getType();

  auto indexCount = getIndices().size();
  if (!((rank == 1 && indexCount == 1) || (rank == 2 && indexCount == 2)))
    return emitOpError(
        "scalar_store expects rank-1/2 dest with matching indices (rank-1: 1; rank-2: 2)");
  for (mlir::Value idx : getIndices()) {
    if (!idx.getType().isIndex())
      return emitOpError("indices must be index-typed");
  }
  return mlir::success();
}

} // namespace tla
