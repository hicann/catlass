#include "Dialect/Tla/IR/TlaOps.h"
#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "tla/Ops.cpp.inc"

namespace tla {

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
