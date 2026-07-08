#include "Dialect/Tla/IR/TlaOps.h"
#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  auto i8 = mlir::IntegerType::get(getContext(), 8);
  if (resTy.getPointee() != i8)
    return emitOpError("alloc_ptr requires !tla.ptr<i8, ...> result");
  if (resTy.getAlignment() == 0)
    return emitOpError("result pointer alignment must be positive");
  auto ms = resTy.getAddrspace();
  if (ms == AddressSpace::generic || ms == AddressSpace::gm)
    return emitOpError("alloc_ptr requires on-chip !tla.ptr (l1, l0a, l0b, l0c, ub)");
  if (getSizeBytesAttr().getInt() <= 0)
    return emitOpError("size_bytes must be positive");
  return mlir::success();
}

mlir::LogicalResult HivmMemrefAsPtrOp::verify() {
  auto mr = mlir::dyn_cast<mlir::MemRefType>(getMemref().getType());
  if (!mr)
    return emitOpError("operand must be a memref");
  if (mr.getRank() != 1)
    return emitOpError("expected rank-1 memref (HIVM pointer_cast lowering)");
  if (!mlir::isa<PtrType>(getResult().getType()))
    return emitOpError("result must be !tla.ptr");
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
// tla.func to func.func -- and convert-tla-to-vector / TlaSplitMixedFuncPass
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

} // namespace tla
