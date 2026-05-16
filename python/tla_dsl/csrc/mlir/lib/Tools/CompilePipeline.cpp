#include "Tools/CompilePipeline.h"

#include "Dialect/Tla/IR/TlaDialect.h"
#include "Dialect/Tla/IR/TlaOps.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "Passes.h"
#include "Tools/AddressSpaceConversion.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"

#include <cstdlib>
#include <memory>

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
using namespace mlir;

namespace {

struct GMRank1ArgInfo {
  bool isGMRank1Static = false;
  int64_t staticSize = -1;
};

using GMFunctionArgMap = llvm::StringMap<SmallVector<GMRank1ArgInfo, 8>>;

static bool isTlaTileType(Type type) { return llvm::isa<::tla::TlaTensorType>(type); }

static bool hasUnloweredTlaTileSignatures(ModuleOp module, std::string &detail) {
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    auto fnType = funcOp.getFunctionType();
    for (Type input : fnType.getInputs()) {
      if (!isTlaTileType(input))
        continue;
      detail = "func.func @" + funcOp.getSymName().str() + " argument";
      return true;
    }
    for (Type result : fnType.getResults()) {
      if (!isTlaTileType(result))
        continue;
      detail = "func.func @" + funcOp.getSymName().str() + " result";
      return true;
    }
  }

  bool hasUnloweredCall = false;
  module.walk([&](func::CallOp callOp) {
    for (Type operandType : callOp.getOperandTypes()) {
      if (!isTlaTileType(operandType))
        continue;
      detail = "func.call @" + callOp.getCallee().str() + " operand";
      hasUnloweredCall = true;
      return WalkResult::interrupt();
    }
    for (Type resultType : callOp.getResultTypes()) {
      if (!isTlaTileType(resultType))
        continue;
      detail = "func.call @" + callOp.getCallee().str() + " result";
      hasUnloweredCall = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasUnloweredCall;
}

static bool rewriteTlaTileTypesToLLVMPointer(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  bool changed = false;

  module.walk([&](func::FuncOp funcOp) {
    auto fnType = funcOp.getFunctionType();
    SmallVector<Type, 8> newInputs;
    SmallVector<Type, 4> newResults;
    newInputs.reserve(fnType.getNumInputs());
    newResults.reserve(fnType.getNumResults());

    bool localChanged = false;
    for (Type input : fnType.getInputs()) {
      if (isTlaTileType(input)) {
        newInputs.push_back(ptrType);
        localChanged = true;
      } else {
        newInputs.push_back(input);
      }
    }
    for (Type result : fnType.getResults()) {
      if (isTlaTileType(result)) {
        newResults.push_back(ptrType);
        localChanged = true;
      } else {
        newResults.push_back(result);
      }
    }
    if (!localChanged)
      return;

    changed = true;
    funcOp.setType(FunctionType::get(ctx, newInputs, newResults));
    if (!funcOp.isDeclaration() && !funcOp.getBody().empty()) {
      Block &entry = funcOp.getBody().front();
      for (unsigned i = 0; i < entry.getNumArguments() && i < newInputs.size(); ++i) {
        if (entry.getArgument(i).getType() != newInputs[i])
          entry.getArgument(i).setType(newInputs[i]);
      }
    }
  });

  SmallVector<func::CallOp, 16> calls;
  module.walk([&](func::CallOp callOp) { calls.push_back(callOp); });
  for (func::CallOp callOp : calls) {
    bool localChanged = false;
    for (Type operandType : callOp.getOperandTypes()) {
      if (isTlaTileType(operandType)) {
        localChanged = true;
        break;
      }
    }
    if (!localChanged) {
      for (Type resultType : callOp.getResultTypes()) {
        if (isTlaTileType(resultType)) {
          localChanged = true;
          break;
        }
      }
    }
    if (!localChanged)
      continue;

    changed = true;
    OpBuilder builder(callOp);
    auto newCall = builder.create<func::CallOp>(callOp.getLoc(), callOp.getCallee(),
                                                callOp.getResultTypes(), callOp.getOperands());
    callOp.replaceAllUsesWith(newCall.getResults());
    callOp.erase();
  }

  return changed;
}

static GMFunctionArgMap collectGMFunctionArgInfo(ModuleOp module) {
  GMFunctionArgMap infoByFunc;
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    SmallVector<GMRank1ArgInfo, 8> argInfo;
    argInfo.reserve(funcOp.getNumArguments());
    for (Type input : funcOp.getArgumentTypes()) {
      GMRank1ArgInfo info;
      auto memrefType = llvm::dyn_cast<MemRefType>(input);
      if (memrefType && memrefType.getRank() == 1 && memrefType.getMemorySpaceAsInt() == 1 &&
          memrefType.hasStaticShape()) {
        info.isGMRank1Static = true;
        info.staticSize = memrefType.getShape()[0];
      }
      argInfo.push_back(info);
    }
    infoByFunc[funcOp.getSymName()] = std::move(argInfo);
  }
  return infoByFunc;
}

static FailureOr<MemRefType> bridgeTlaMemrefType(Type tlaMemrefType) {
  auto tlaMemref = dyn_cast<::tla::MemrefType>(tlaMemrefType);
  if (!tlaMemref)
    return failure();
  for (int64_t dim : tlaMemref.getShape()) {
    if (dim == ShapedType::kDynamic)
      return failure();
  }

  MLIRContext *ctx = tlaMemrefType.getContext();
  FailureOr<int64_t> memorySpaceValue =
      ::tla::mapTlaAddressSpaceToMlirMemRefSpaceValue(tlaMemref.getAddressSpace());
  if (failed(memorySpaceValue))
    return failure();

  Attribute memorySpace = IntegerAttr::get(IntegerType::get(ctx, 64), *memorySpaceValue);
  return MemRefType::get(tlaMemref.getShape(), tlaMemref.getElementType(), AffineMap(),
                         memorySpace);
}

static FailureOr<uint64_t> evaluateI64Value(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constant.getValue()))
      return intAttr.getInt();
    return failure();
  }
  if (auto indexCast = value.getDefiningOp<arith::IndexCastOp>())
    return evaluateI64Value(indexCast.getIn());
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = evaluateI64Value(add.getLhs());
    auto rhs = evaluateI64Value(add.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();
    return *lhs + *rhs;
  }
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    auto lhs = evaluateI64Value(mul.getLhs());
    auto rhs = evaluateI64Value(mul.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();
    return (*lhs) * (*rhs);
  }
  if (auto div = value.getDefiningOp<arith::DivUIOp>()) {
    auto lhs = evaluateI64Value(div.getLhs());
    auto rhs = evaluateI64Value(div.getRhs());
    if (failed(lhs) || failed(rhs) || *rhs == 0)
      return failure();
    return (*lhs) / (*rhs);
  }
  return failure();
}

static func::FuncOp getOrCreateRuntimeCall(ModuleOp module, StringRef name,
                                           ArrayRef<Type> operandTypes,
                                           ArrayRef<Type> resultTypes = {}) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;

  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = builder.getFunctionType(operandTypes, resultTypes);
  auto func = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
  func.setPrivate();
  return func;
}

static bool
prepareTlaMemrefsForLLVMTranslation(ModuleOp module,
                                    llvm::StringMap<SmallVector<uint64_t, 8>> &ubByteOffsetsByFunc,
                                    std::string &error) {
  struct MaterializedAllocation {
    Value source;
    Type tlaMemrefType;
    Value builtinAlloc;
  };

  DenseMap<Value, Value> builtinMemrefByTlaValue;
  SmallVector<UnrealizedConversionCastOp, 8> allocationBridgeCasts;
  SmallVector<MaterializedAllocation, 8> materializedAllocations;

  module.walk([&](UnrealizedConversionCastOp castOp) {
    if (castOp.getNumOperands() != 1 || castOp.getNumResults() != 1)
      return;
    if (!castOp.getOperand(0).getType().isInteger(64))
      return;
    if (failed(bridgeTlaMemrefType(castOp.getResult(0).getType())))
      return;
    allocationBridgeCasts.push_back(castOp);
  });

  for (UnrealizedConversionCastOp castOp : allocationBridgeCasts) {
    if (!castOp || !castOp->getBlock())
      continue;

    Value builtinAlloc;
    bool reusedExistingAlloc = false;
    for (const auto &entry : materializedAllocations) {
      if (entry.source == castOp.getOperand(0) &&
          entry.tlaMemrefType == castOp.getResult(0).getType()) {
        builtinAlloc = entry.builtinAlloc;
        reusedExistingAlloc = true;
        break;
      }
    }

    FailureOr<MemRefType> builtinType = bridgeTlaMemrefType(castOp.getResult(0).getType());
    if (failed(builtinType)) {
      error = "Failed to prepare memref allocation cast for LLVM translation.";
      return false;
    }

    if (!reusedExistingAlloc) {
      OpBuilder builder(castOp);
      builtinAlloc = builder.create<::mlir::memref::AllocaOp>(castOp.getLoc(), *builtinType);
      materializedAllocations.push_back(
          {castOp.getOperand(0), castOp.getResult(0).getType(), builtinAlloc});
    }
    builtinMemrefByTlaValue[castOp.getResult(0)] = builtinAlloc;
    if (!reusedExistingAlloc && builtinType->getMemorySpaceAsInt() == 6) {
      auto maybeOffset = evaluateI64Value(castOp.getOperand(0));
      if (failed(maybeOffset)) {
        error = "Failed to evaluate UB allocation byte offset for LLVM translation.";
        return false;
      }
      auto parentFunc = castOp->getParentOfType<func::FuncOp>();
      if (!parentFunc) {
        error = "Failed to resolve parent func for UB allocation.";
        return false;
      }
      ubByteOffsetsByFunc[parentFunc.getSymName()].push_back(*maybeOffset);
    }
  }

  SmallVector<UnrealizedConversionCastOp, 16> bridgeCasts;
  module.walk([&](UnrealizedConversionCastOp castOp) { bridgeCasts.push_back(castOp); });
  for (UnrealizedConversionCastOp castOp : bridgeCasts) {
    if (!castOp || !castOp->getBlock() || castOp.getNumOperands() != 1 ||
        castOp.getNumResults() != 1)
      continue;

    Value operand = castOp.getOperand(0);
    auto mapped = builtinMemrefByTlaValue.find(operand);
    if (mapped == builtinMemrefByTlaValue.end())
      continue;

    if (castOp.getResult(0).getType() != mapped->second.getType())
      continue;
    castOp.getResult(0).replaceAllUsesWith(mapped->second);
    castOp.erase();
  }

  SmallVector<::tla::LoadOp, 8> loadOps;
  SmallVector<::tla::StoreOp, 8> storeOps;
  module.walk([&](::tla::LoadOp op) { loadOps.push_back(op); });
  module.walk([&](::tla::StoreOp op) { storeOps.push_back(op); });

  for (::tla::LoadOp op : loadOps) {
    if (!op || !op->getBlock() || op->getNumOperands() != 2 || op->getNumResults() != 0)
      continue;

    auto dstIt = builtinMemrefByTlaValue.find(op->getOperand(0));
    if (dstIt == builtinMemrefByTlaValue.end())
      continue;

    auto srcType = llvm::dyn_cast<MemRefType>(op->getOperand(1).getType());
    auto dstType = llvm::dyn_cast<MemRefType>(dstIt->second.getType());
    if (!srcType || !dstType || srcType.getRank() != 1 || dstType.getRank() != 1 ||
        !srcType.getElementType().isF32() || !dstType.getElementType().isF32())
      continue;

    OpBuilder builder(op);
    auto i32Type = builder.getI32Type();
    auto f32Type = builder.getF32Type();
    auto i64Type = builder.getI64Type();
    auto padMode = builder.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);
    auto padValue =
        builder.create<arith::ConstantOp>(op.getLoc(), f32Type, builder.getFloatAttr(f32Type, 0.0));
    auto leftPadding = builder.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
    auto evictionPolicy = builder.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);
    auto callee = getOrCreateRuntimeCall(module, "_mlir_ciface_load_gm_to_ubuf_1d_float",
                                         {srcType, dstType, i32Type, f32Type, i64Type, i32Type});
    builder.create<func::CallOp>(op.getLoc(), callee,
                                 ValueRange{op->getOperand(1), dstIt->second, padMode, padValue,
                                            leftPadding, evictionPolicy});
    op.erase();
  }

  for (::tla::StoreOp op : storeOps) {
    if (!op || !op->getBlock() || op->getNumOperands() != 2 || op->getNumResults() != 0)
      continue;

    auto srcIt = builtinMemrefByTlaValue.find(op->getOperand(1));
    if (srcIt == builtinMemrefByTlaValue.end())
      continue;

    auto dstType = llvm::dyn_cast<MemRefType>(op->getOperand(0).getType());
    auto srcType = llvm::dyn_cast<MemRefType>(srcIt->second.getType());
    if (!srcType || !dstType || srcType.getRank() != 1 || dstType.getRank() != 1 ||
        !srcType.getElementType().isF32() || !dstType.getElementType().isF32())
      continue;

    OpBuilder builder(op);
    auto i32Type = builder.getI32Type();
    auto atomicKind = builder.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);
    auto callee = getOrCreateRuntimeCall(module, "_mlir_ciface_store_ubuf_to_gm_1d_float",
                                         {srcType, dstType, i32Type});
    builder.create<func::CallOp>(op.getLoc(), callee,
                                 ValueRange{srcIt->second, op->getOperand(0), atomicKind});
    op.erase();
  }

  for (UnrealizedConversionCastOp castOp : allocationBridgeCasts) {
    if (!castOp || !castOp->getBlock())
      continue;
    if (castOp.getResult(0).use_empty())
      castOp.erase();
  }

  return true;
}

static llvm::StructType *getRank1MemrefDescriptorType(ArrayRef<llvm::Type *> fields) {
  assert(fields.size() == 5 && "expected 5 flattened fields for rank-1 memref");
  auto *sizeArrayTy = llvm::ArrayType::get(fields[3], 1);
  auto *strideArrayTy = llvm::ArrayType::get(fields[4], 1);
  return llvm::StructType::get(fields[0], fields[1], fields[2], sizeArrayTy, strideArrayTy);
}

static llvm::StructType *getRank2MemrefDescriptorType(ArrayRef<llvm::Type *> fields) {
  assert(fields.size() == 7 && "expected 7 flattened fields for rank-2 memref");
  auto *sizeArrayTy = llvm::ArrayType::get(fields[3], 2);
  auto *strideArrayTy = llvm::ArrayType::get(fields[5], 2);
  return llvm::StructType::get(fields[0], fields[1], fields[2], sizeArrayTy, strideArrayTy);
}

static llvm::Value *createRank1DescriptorAlloca(llvm::IRBuilder<> &builder,
                                                ArrayRef<llvm::Value *> fields, llvm::Twine name) {
  assert(fields.size() == 5 && "expected 5 flattened fields for rank-1 memref");
  SmallVector<llvm::Type *, 5> fieldTypes;
  fieldTypes.reserve(fields.size());
  for (llvm::Value *field : fields)
    fieldTypes.push_back(field->getType());

  auto *descTy = getRank1MemrefDescriptorType(fieldTypes);
  llvm::Value *undef = llvm::UndefValue::get(descTy);
  llvm::Value *desc = builder.CreateInsertValue(undef, fields[0], {0});
  desc = builder.CreateInsertValue(desc, fields[1], {1});
  desc = builder.CreateInsertValue(desc, fields[2], {2});
  desc = builder.CreateInsertValue(desc, fields[3], {3, 0});
  desc = builder.CreateInsertValue(desc, fields[4], {4, 0});

  llvm::Value *slot = builder.CreateAlloca(descTy, nullptr, name);
  builder.CreateStore(desc, slot);
  return slot;
}

static llvm::Value *createRank2DescriptorAlloca(llvm::IRBuilder<> &builder,
                                                ArrayRef<llvm::Value *> fields, llvm::Twine name) {
  assert(fields.size() == 7 && "expected 7 flattened fields for rank-2 memref");
  SmallVector<llvm::Type *, 7> fieldTypes;
  fieldTypes.reserve(fields.size());
  for (llvm::Value *field : fields)
    fieldTypes.push_back(field->getType());

  auto *descTy = getRank2MemrefDescriptorType(fieldTypes);
  llvm::Value *undef = llvm::UndefValue::get(descTy);
  llvm::Value *desc = builder.CreateInsertValue(undef, fields[0], {0});
  desc = builder.CreateInsertValue(desc, fields[1], {1});
  desc = builder.CreateInsertValue(desc, fields[2], {2});
  desc = builder.CreateInsertValue(desc, fields[3], {3, 0});
  desc = builder.CreateInsertValue(desc, fields[4], {3, 1});
  desc = builder.CreateInsertValue(desc, fields[5], {4, 0});
  desc = builder.CreateInsertValue(desc, fields[6], {4, 1});

  llvm::Value *slot = builder.CreateAlloca(descTy, nullptr, name);
  builder.CreateStore(desc, slot);
  return slot;
}

static llvm::Function *getOrCreateLoadWrapper(llvm::Module &module, llvm::Function &callee) {
  constexpr llvm::StringLiteral kWrapperName = "load_gm_to_ubuf_1d_float";
  if (llvm::Function *existing = module.getFunction(kWrapperName))
    return existing;

  llvm::LLVMContext &ctx = module.getContext();
  llvm::Type *i64Ty = llvm::Type::getInt64Ty(ctx);
  llvm::Type *i32Ty = llvm::Type::getInt32Ty(ctx);
  llvm::Type *f32Ty = llvm::Type::getFloatTy(ctx);
  llvm::Type *gmPtrTy = llvm::PointerType::get(ctx, 1);
  llvm::Type *ubPtrTy = llvm::PointerType::get(ctx, 6);
  SmallVector<llvm::Type *, 13> wrapperArgs = {gmPtrTy, gmPtrTy, i64Ty, i64Ty, i64Ty,
                                               ubPtrTy, ubPtrTy, i64Ty, i64Ty, i64Ty,
                                               i32Ty,   f32Ty,   i64Ty};
  auto *wrapperTy = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), wrapperArgs, false);
  auto *wrapper =
      llvm::Function::Create(wrapperTy, llvm::GlobalValue::PrivateLinkage, kWrapperName, module);
  wrapper->addFnAttr(llvm::Attribute::AlwaysInline);

  auto argIt = wrapper->arg_begin();
  llvm::Value *srcAllocated = &*argIt++;
  llvm::Value *srcAligned = &*argIt++;
  llvm::Value *srcOffset = &*argIt++;
  llvm::Value *srcSize = &*argIt++;
  llvm::Value *srcStride = &*argIt++;
  llvm::Value *dstAllocated = &*argIt++;
  llvm::Value *dstAligned = &*argIt++;
  llvm::Value *dstOffset = &*argIt++;
  llvm::Value *dstSize = &*argIt++;
  llvm::Value *dstStride = &*argIt++;
  llvm::Value *padMode = &*argIt++;
  llvm::Value *padValue = &*argIt++;
  llvm::Value *leftPadding = &*argIt++;

  llvm::BasicBlock *entry = llvm::BasicBlock::Create(ctx, "entry", wrapper);
  llvm::IRBuilder<> builder(entry);
  llvm::Value *srcDesc = createRank1DescriptorAlloca(
      builder, {srcAllocated, srcAligned, srcOffset, srcSize, srcStride}, "src.desc");
  llvm::Value *dstDesc = createRank1DescriptorAlloca(
      builder, {dstAllocated, dstAligned, dstOffset, dstSize, dstStride}, "dst.desc");
  llvm::Value *evictionPolicy = llvm::ConstantInt::get(i32Ty, 0);
  builder.CreateCall(&callee, {srcDesc, dstDesc, padMode, padValue, leftPadding, evictionPolicy});
  builder.CreateRetVoid();
  return wrapper;
}

static llvm::Function *getOrCreateStoreWrapper(llvm::Module &module, llvm::Function &callee) {
  constexpr llvm::StringLiteral kWrapperName = "store_ubuf_to_gm_1d_float";
  if (llvm::Function *existing = module.getFunction(kWrapperName))
    return existing;

  llvm::LLVMContext &ctx = module.getContext();
  llvm::Type *i64Ty = llvm::Type::getInt64Ty(ctx);
  llvm::Type *i32Ty = llvm::Type::getInt32Ty(ctx);
  llvm::Type *gmPtrTy = llvm::PointerType::get(ctx, 1);
  llvm::Type *ubPtrTy = llvm::PointerType::get(ctx, 6);
  SmallVector<llvm::Type *, 11> wrapperArgs = {ubPtrTy, ubPtrTy, i64Ty, i64Ty, i64Ty, gmPtrTy,
                                               gmPtrTy, i64Ty,   i64Ty, i64Ty, i32Ty};
  auto *wrapperTy = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), wrapperArgs, false);
  auto *wrapper =
      llvm::Function::Create(wrapperTy, llvm::GlobalValue::PrivateLinkage, kWrapperName, module);
  wrapper->addFnAttr(llvm::Attribute::AlwaysInline);

  auto argIt = wrapper->arg_begin();
  llvm::Value *srcAllocated = &*argIt++;
  llvm::Value *srcAligned = &*argIt++;
  llvm::Value *srcOffset = &*argIt++;
  llvm::Value *srcSize = &*argIt++;
  llvm::Value *srcStride = &*argIt++;
  llvm::Value *dstAllocated = &*argIt++;
  llvm::Value *dstAligned = &*argIt++;
  llvm::Value *dstOffset = &*argIt++;
  llvm::Value *dstSize = &*argIt++;
  llvm::Value *dstStride = &*argIt++;
  llvm::Value *atomicKind = &*argIt++;

  llvm::BasicBlock *entry = llvm::BasicBlock::Create(ctx, "entry", wrapper);
  llvm::IRBuilder<> builder(entry);
  llvm::Value *srcDesc = createRank1DescriptorAlloca(
      builder, {srcAllocated, srcAligned, srcOffset, srcSize, srcStride}, "src.desc");
  llvm::Value *dstDesc = createRank1DescriptorAlloca(
      builder, {dstAllocated, dstAligned, dstOffset, dstSize, dstStride}, "dst.desc");
  builder.CreateCall(&callee, {srcDesc, dstDesc, atomicKind});
  builder.CreateRetVoid();
  return wrapper;
}

static llvm::Function *getOrCreateVaddWrapper(llvm::Module &module, llvm::Function &callee) {
  constexpr llvm::StringLiteral kWrapperName = "vadd_1d_float";
  if (llvm::Function *existing = module.getFunction(kWrapperName))
    return existing;

  llvm::LLVMContext &ctx = module.getContext();
  llvm::Type *i64Ty = llvm::Type::getInt64Ty(ctx);
  llvm::Type *ubPtrTy = llvm::PointerType::get(ctx, 6);
  SmallVector<llvm::Type *, 20> wrapperArgs = {
      ubPtrTy, ubPtrTy, i64Ty, i64Ty, i64Ty, ubPtrTy, ubPtrTy, i64Ty, i64Ty, i64Ty,
      ubPtrTy, ubPtrTy, i64Ty, i64Ty, i64Ty, ubPtrTy, ubPtrTy, i64Ty, i64Ty, i64Ty};
  auto *wrapperTy = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), wrapperArgs, false);
  auto *wrapper =
      llvm::Function::Create(wrapperTy, llvm::GlobalValue::PrivateLinkage, kWrapperName, module);
  wrapper->addFnAttr(llvm::Attribute::AlwaysInline);

  SmallVector<llvm::Value *, 20> args;
  for (llvm::Argument &arg : wrapper->args())
    args.push_back(&arg);

  llvm::BasicBlock *entry = llvm::BasicBlock::Create(ctx, "entry", wrapper);
  llvm::IRBuilder<> builder(entry);
  llvm::Value *src0Desc =
      createRank1DescriptorAlloca(builder, ArrayRef<llvm::Value *>(args).slice(0, 5), "src0.desc");
  llvm::Value *src1Desc =
      createRank1DescriptorAlloca(builder, ArrayRef<llvm::Value *>(args).slice(5, 5), "src1.desc");
  llvm::Value *dstDesc =
      createRank1DescriptorAlloca(builder, ArrayRef<llvm::Value *>(args).slice(10, 5), "dst.desc");
  llvm::Value *tmpDesc =
      createRank1DescriptorAlloca(builder, ArrayRef<llvm::Value *>(args).slice(15, 5), "tmp.desc");
  builder.CreateCall(&callee, {src0Desc, src1Desc, dstDesc, tmpDesc});
  builder.CreateRetVoid();
  return wrapper;
}

static llvm::Function *getOrCreateRank2CopyWrapper(llvm::Module &module, llvm::Function &callee,
                                                   llvm::StringRef wrapperName,
                                                   unsigned srcAddrspace, unsigned dstAddrspace) {
  if (llvm::Function *existing = module.getFunction(wrapperName))
    return existing;

  llvm::LLVMContext &ctx = module.getContext();
  llvm::Type *i64Ty = llvm::Type::getInt64Ty(ctx);
  llvm::Type *srcPtrTy = llvm::PointerType::get(ctx, srcAddrspace);
  llvm::Type *dstPtrTy = llvm::PointerType::get(ctx, dstAddrspace);
  SmallVector<llvm::Type *, 14> wrapperArgs = {srcPtrTy, srcPtrTy, i64Ty,    i64Ty,    i64Ty,
                                               i64Ty,    i64Ty,    dstPtrTy, dstPtrTy, i64Ty,
                                               i64Ty,    i64Ty,    i64Ty,    i64Ty};
  llvm::FunctionType *calleeTy = callee.getFunctionType();
  for (unsigned i = 2; i < calleeTy->getNumParams(); ++i)
    wrapperArgs.push_back(calleeTy->getParamType(i));
  auto *wrapperTy = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), wrapperArgs, false);
  auto *wrapper =
      llvm::Function::Create(wrapperTy, llvm::GlobalValue::PrivateLinkage, wrapperName, module);
  wrapper->addFnAttr(llvm::Attribute::AlwaysInline);

  SmallVector<llvm::Value *, 14> args;
  for (llvm::Argument &arg : wrapper->args())
    args.push_back(&arg);

  llvm::BasicBlock *entry = llvm::BasicBlock::Create(ctx, "entry", wrapper);
  llvm::IRBuilder<> builder(entry);
  llvm::Value *srcDesc =
      createRank2DescriptorAlloca(builder, ArrayRef<llvm::Value *>(args).slice(0, 7), "src.desc");
  llvm::Value *dstDesc =
      createRank2DescriptorAlloca(builder, ArrayRef<llvm::Value *>(args).slice(7, 7), "dst.desc");
  SmallVector<llvm::Value *, 22> calleeArgs = {srcDesc, dstDesc};
  for (llvm::Value *arg : ArrayRef<llvm::Value *>(args).drop_front(14))
    calleeArgs.push_back(arg);
  builder.CreateCall(&callee, calleeArgs);
  builder.CreateRetVoid();
  return wrapper;
}

static llvm::Function *getOrCreateCopyTileGmToCbufWrapper(llvm::Module &module,
                                                          llvm::Function &callee) {
  constexpr llvm::StringLiteral wrapperName = "copy_tile_gm_row_major_to_cbuf_zn_c310";
  return getOrCreateRank2CopyWrapper(module, callee, wrapperName,
                                     /*srcAddrspace=*/1, /*dstAddrspace=*/2);
}

static llvm::Function *getOrCreateCopyTileCbufToCaWrapper(llvm::Module &module,
                                                          llvm::Function &callee) {
  constexpr llvm::StringLiteral wrapperName = "copy_tile_cbuf_zn_to_ca_zz_c310";
  return getOrCreateRank2CopyWrapper(module, callee, wrapperName,
                                     /*srcAddrspace=*/2, /*dstAddrspace=*/3);
}

static llvm::Function *getOrCreateCopyTileCbufToCbWrapper(llvm::Module &module,
                                                          llvm::Function &callee) {
  constexpr llvm::StringLiteral wrapperName = "copy_tile_cbuf_zn_to_cb_nz_c310";
  return getOrCreateRank2CopyWrapper(module, callee, wrapperName,
                                     /*srcAddrspace=*/2, /*dstAddrspace=*/4);
}

static llvm::Function *getOrCreateCopyTileCcToGmWrapper(llvm::Module &module,
                                                        llvm::Function &callee) {
  constexpr llvm::StringLiteral wrapperName = "copy_tile_cc_l0c_to_gm_row_major_c310";
  return getOrCreateRank2CopyWrapper(module, callee, wrapperName,
                                     /*srcAddrspace=*/5, /*dstAddrspace=*/1);
}

static uint64_t getElementByteWidthFromGetImmCallee(llvm::StringRef calleeName) {
  if (calleeName.ends_with("_bf16") || calleeName.ends_with("_f16") || calleeName.ends_with("_i16"))
    return 2;
  if (calleeName.ends_with("_f32") || calleeName.ends_with("_i32"))
    return 4;
  if (calleeName.ends_with("_f64") || calleeName.ends_with("_i64"))
    return 8;
  if (calleeName.ends_with("_i8") || calleeName.ends_with("_i1"))
    return 1;
  return 0;
}

static llvm::Value *resolveInsertedValueForIndices(llvm::Value *aggregate,
                                                   ArrayRef<unsigned> indices) {
  while (auto *insert = llvm::dyn_cast<llvm::InsertValueInst>(aggregate)) {
    if (insert->getIndices() == indices)
      return insert->getInsertedValueOperand();
    aggregate = insert->getAggregateOperand();
  }
  return nullptr;
}

static llvm::Value *peelDescriptorExtract(llvm::Value *value) {
  auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(value);
  if (!extract)
    return value;
  if (llvm::Value *inserted =
          resolveInsertedValueForIndices(extract->getAggregateOperand(), extract->getIndices()))
    return inserted;
  return value;
}

static void deleteDeadDescriptorTraffic(llvm::Module &module) {
  SmallVector<llvm::Instruction *, 64> worklist;
  for (llvm::Function &func : module) {
    for (llvm::BasicBlock &block : func) {
      for (llvm::Instruction &inst : llvm::make_early_inc_range(block)) {
        if (!llvm::isa<llvm::InsertValueInst, llvm::ExtractValueInst>(inst))
          continue;
        if (llvm::isInstructionTriviallyDead(&inst))
          worklist.push_back(&inst);
      }
    }
  }

  for (llvm::Instruction *inst : worklist) {
    if (inst && inst->getParent())
      llvm::RecursivelyDeleteTriviallyDeadInstructions(inst);
  }
}

static void rewriteUBAllocationsToSyntheticPointers(
    llvm::Module &module, const llvm::StringMap<SmallVector<uint64_t, 8>> &ubByteOffsetsByFunc) {
  for (llvm::Function &func : module) {
    if (func.isDeclaration() || func.hasPrivateLinkage())
      continue;
    auto offsetsIt = ubByteOffsetsByFunc.find(func.getName());
    if (offsetsIt == ubByteOffsetsByFunc.end())
      continue;
    ArrayRef<uint64_t> offsets = offsetsIt->second;
    size_t nextOffsetIndex = 0;
    SmallVector<llvm::Instruction *, 16> toErase;

    for (llvm::BasicBlock &block : func) {
      for (llvm::Instruction &inst : block) {
        auto *alloca = llvm::dyn_cast<llvm::AllocaInst>(&inst);
        if (!alloca || alloca->getAddressSpace() != 6)
          continue;
        if (nextOffsetIndex >= offsets.size())
          continue;
        llvm::Constant *replacement = nullptr;
        uint64_t byteOffset = offsets[nextOffsetIndex++];
        if (byteOffset == 0) {
          replacement = llvm::ConstantPointerNull::get(alloca->getType());
        } else {
          replacement = llvm::ConstantExpr::getIntToPtr(
              llvm::ConstantInt::get(llvm::Type::getInt64Ty(module.getContext()), byteOffset),
              alloca->getType());
        }
        alloca->replaceAllUsesWith(replacement);
        toErase.push_back(alloca);
      }
    }

    for (llvm::Instruction *inst : toErase) {
      if (inst && inst->getParent())
        inst->eraseFromParent();
    }
  }
}

static bool rewriteCifaceCallsWithWrappers(llvm::Module &module, std::string &error) {
  auto rewriteCallUsers = [&](llvm::StringRef calleeName, auto wrapperFactory,
                              unsigned droppedTrailingArgs) {
    llvm::Function *callee = module.getFunction(calleeName);
    if (!callee)
      return true;

    SmallVector<llvm::CallInst *, 8> callSites;
    for (llvm::User *user : callee->users()) {
      auto *call = llvm::dyn_cast<llvm::CallInst>(user);
      if (!call || call->getCalledFunction() != callee)
        continue;
      if (call->getFunction() == nullptr)
        continue;
      callSites.push_back(call);
    }
    if (callSites.empty())
      return true;

    llvm::Function *wrapper = wrapperFactory(module, *callee);
    for (llvm::CallInst *call : callSites) {
      if (!call || !call->getParent())
        continue;
      if (call->getFunction() == wrapper)
        continue;
      if (call->arg_size() < droppedTrailingArgs) {
        error = "LLVM runtime wrapper rewrite found an unexpected call arity.";
        return false;
      }

      SmallVector<llvm::Value *, 20> args(call->args());
      if (droppedTrailingArgs > 0)
        args.resize(args.size() - droppedTrailingArgs);
      for (llvm::Value *&arg : args)
        arg = peelDescriptorExtract(arg);

      llvm::IRBuilder<> builder(call);
      llvm::CallInst *newCall = builder.CreateCall(wrapper, args);
      call->replaceAllUsesWith(newCall);
      call->eraseFromParent();
    }
    return true;
  };

  if (!rewriteCallUsers("_mlir_ciface_load_gm_to_ubuf_1d_float", getOrCreateLoadWrapper,
                        /*droppedTrailingArgs=*/1))
    return false;
  if (!rewriteCallUsers("_mlir_ciface_store_ubuf_to_gm_1d_float", getOrCreateStoreWrapper,
                        /*droppedTrailingArgs=*/0))
    return false;
  if (!rewriteCallUsers("_mlir_ciface_vadd_1d_float", getOrCreateVaddWrapper,
                        /*droppedTrailingArgs=*/0))
    return false;
  if (!rewriteCallUsers("_mlir_ciface_copy_tile_gm_row_major_to_cbuf_zn_c310",
                        getOrCreateCopyTileGmToCbufWrapper,
                        /*droppedTrailingArgs=*/0))
    return false;
  if (!rewriteCallUsers("_mlir_ciface_copy_tile_cbuf_zn_to_ca_zz_c310",
                        getOrCreateCopyTileCbufToCaWrapper,
                        /*droppedTrailingArgs=*/0))
    return false;
  if (!rewriteCallUsers("_mlir_ciface_copy_tile_cbuf_zn_to_cb_nz_c310",
                        getOrCreateCopyTileCbufToCbWrapper,
                        /*droppedTrailingArgs=*/0))
    return false;
  if (!rewriteCallUsers("_mlir_ciface_copy_tile_cc_l0c_to_gm_row_major_c310",
                        getOrCreateCopyTileCcToGmWrapper,
                        /*droppedTrailingArgs=*/0))
    return false;
  deleteDeadDescriptorTraffic(module);
  return true;
}

static void addHivmKernelAnnotations(llvm::Module &module) {
  llvm::NamedMDNode *annotations = module.getOrInsertNamedMetadata("hivm.annotations");
  while (annotations->getNumOperands() > 0)
    annotations->eraseFromParent();
  annotations = module.getOrInsertNamedMetadata("hivm.annotations");

  llvm::LLVMContext &ctx = module.getContext();
  llvm::Metadata *kernelTag = llvm::MDString::get(ctx, "kernel");
  llvm::Metadata *kernelValue =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1));

  for (llvm::Function &func : module) {
    if (func.isDeclaration() || func.hasPrivateLinkage())
      continue;
    llvm::Metadata *funcRef = llvm::ValueAsMetadata::get(&func);
    auto *node = llvm::MDNode::get(ctx, {funcRef, kernelTag, kernelValue});
    annotations->addOperand(node);
  }

  if (annotations->getNumOperands() == 0)
    annotations->eraseFromParent();
}

static void assignHivmDeclarationAttributes(llvm::Module &module) {
  auto addAttrs = [&](llvm::StringRef name, std::initializer_list<llvm::Attribute::AttrKind> attrs,
                      bool inaccessibleMemOnly = false, bool readNone = false) {
    llvm::Function *func = module.getFunction(name);
    if (!func)
      return;
    for (llvm::Attribute::AttrKind attr : attrs)
      func->addFnAttr(attr);
    if (inaccessibleMemOnly)
      func->setOnlyAccessesInaccessibleMemory();
    if (readNone)
      func->setDoesNotAccessMemory();
  };

  addAttrs("llvm.hivm.GET.CTRL", {llvm::Attribute::NoUnwind}, /*inaccessibleMemOnly=*/true);
  addAttrs("llvm.hivm.SBITSET0", {llvm::Attribute::NoUnwind}, /*inaccessibleMemOnly=*/false,
           /*readNone=*/true);
  addAttrs("llvm.hivm.SET.CTRL", {llvm::Attribute::NoUnwind}, /*inaccessibleMemOnly=*/true);
  addAttrs("llvm.hivm.SET.FLAG.IMM", {llvm::Attribute::NoUnwind});
  addAttrs("llvm.hivm.WAIT.FLAG.IMM", {llvm::Attribute::NoUnwind});
  addAttrs("llvm.hivm.BARRIER", {llvm::Attribute::NoUnwind}, /*inaccessibleMemOnly=*/true);
}

static void normalizeLegacyLLVMAttributeSpellings(std::string &text) {
  auto replaceAll = [&](llvm::StringRef from, llvm::StringRef to) {
    size_t pos = 0;
    while ((pos = text.find(from.str(), pos)) != std::string::npos) {
      text.replace(pos, from.size(), to.str());
      pos += to.size();
    }
  };
  replaceAll("memory(inaccessiblemem: readwrite)", "inaccessiblememonly");
  replaceAll("memory(none)", "readnone");
}

static bool rewriteGMRawPointerABI(llvm::Module &module, const GMFunctionArgMap &gmArgInfoByFunc,
                                   std::string &error) {
  llvm::LLVMContext &ctx = module.getContext();
  llvm::Type *i64Ty = llvm::Type::getInt64Ty(ctx);
  SmallVector<llvm::Function *, 8> functions;
  for (llvm::Function &func : module)
    functions.push_back(&func);

  for (llvm::Function *func : functions) {
    if (!func || func->isDeclaration())
      continue;

    auto infoIt = gmArgInfoByFunc.find(func->getName());
    if (infoIt == gmArgInfoByFunc.end())
      continue;
    ArrayRef<GMRank1ArgInfo> argInfo = infoIt->second;
    if (argInfo.empty())
      continue;

    SmallVector<llvm::Type *, 8> newArgTypes;
    SmallVector<unsigned, 8> oldArgStarts;
    newArgTypes.reserve(argInfo.size());
    oldArgStarts.reserve(argInfo.size());

    unsigned oldArgIndex = 0;
    bool needsRewrite = false;
    for (const GMRank1ArgInfo &info : argInfo) {
      oldArgStarts.push_back(oldArgIndex);
      if (info.isGMRank1Static) {
        needsRewrite = true;
        if (oldArgIndex + 5 > func->arg_size()) {
          error = "GM raw-pointer ABI rewrite found an unexpected flattened memref arity.";
          return false;
        }
        newArgTypes.push_back(func->getArg(oldArgIndex + 1)->getType());
        oldArgIndex += 5;
      } else {
        if (oldArgIndex >= func->arg_size()) {
          error = "GM raw-pointer ABI rewrite found an unexpected function arity.";
          return false;
        }
        newArgTypes.push_back(func->getArg(oldArgIndex)->getType());
        oldArgIndex += 1;
      }
    }
    if (!needsRewrite)
      continue;
    if (oldArgIndex != func->arg_size()) {
      error = "GM raw-pointer ABI rewrite could not fully account for function arguments.";
      return false;
    }

    SmallVector<llvm::User *, 8> nonCallUsers;
    SmallVector<llvm::CallInst *, 8> callUsers;
    for (llvm::User *user : func->users()) {
      auto *call = llvm::dyn_cast<llvm::CallInst>(user);
      if (!call || call->getCalledFunction() != func) {
        nonCallUsers.push_back(user);
        continue;
      }
      callUsers.push_back(call);
    }
    if (!nonCallUsers.empty()) {
      error = "GM raw-pointer ABI rewrite only supports direct call users.";
      return false;
    }

    auto *newFuncTy = llvm::FunctionType::get(func->getReturnType(), newArgTypes, func->isVarArg());
    auto *newFunc = llvm::Function::Create(newFuncTy, func->getLinkage(), func->getAddressSpace(),
                                           func->getName() + ".gm_raw", &module);
    newFunc->copyAttributesFrom(func);
    newFunc->setComdat(func->getComdat());
    newFunc->setCallingConv(func->getCallingConv());
    newFunc->setSubprogram(func->getSubprogram());

    newFunc->splice(newFunc->begin(), func);

    auto newArgIt = newFunc->arg_begin();
    for (auto [inputOrdinal, info] : llvm::enumerate(argInfo)) {
      unsigned oldStart = oldArgStarts[inputOrdinal];
      if (info.isGMRank1Static) {
        llvm::Argument *rawPtrArg = &*newArgIt++;
        rawPtrArg->setName(func->getArg(oldStart + 1)->getName());
        func->getArg(oldStart + 0)->replaceAllUsesWith(rawPtrArg);
        func->getArg(oldStart + 1)->replaceAllUsesWith(rawPtrArg);
        func->getArg(oldStart + 2)->replaceAllUsesWith(llvm::ConstantInt::get(i64Ty, 0));
        func->getArg(oldStart + 3)
            ->replaceAllUsesWith(llvm::ConstantInt::get(i64Ty, info.staticSize));
        func->getArg(oldStart + 4)->replaceAllUsesWith(llvm::ConstantInt::get(i64Ty, 1));
      } else {
        llvm::Argument *newArg = &*newArgIt++;
        newArg->setName(func->getArg(oldStart)->getName());
        func->getArg(oldStart)->replaceAllUsesWith(newArg);
      }
    }

    for (llvm::CallInst *call : callUsers) {
      if (!call || !call->getParent())
        continue;
      SmallVector<llvm::Value *, 8> newArgs;
      newArgs.reserve(argInfo.size());
      unsigned callArgIndex = 0;
      for (const GMRank1ArgInfo &info : argInfo) {
        if (info.isGMRank1Static) {
          if (callArgIndex + 5 > call->arg_size()) {
            error = "GM raw-pointer ABI rewrite found an unexpected call arity.";
            return false;
          }
          newArgs.push_back(call->getArgOperand(callArgIndex + 1));
          callArgIndex += 5;
        } else {
          if (callArgIndex >= call->arg_size()) {
            error = "GM raw-pointer ABI rewrite found an unexpected call arity.";
            return false;
          }
          newArgs.push_back(call->getArgOperand(callArgIndex));
          callArgIndex += 1;
        }
      }
      llvm::IRBuilder<> builder(call);
      llvm::CallInst *newCall = builder.CreateCall(newFunc, newArgs);
      newCall->setCallingConv(call->getCallingConv());
      newCall->setTailCallKind(call->getTailCallKind());
      call->replaceAllUsesWith(newCall);
      call->eraseFromParent();
    }

    std::string originalName = func->getName().str();
    func->setName(originalName + ".flattened_gm");
    newFunc->setName(originalName);
    func->eraseFromParent();
  }

  return true;
}

} // namespace

namespace tla::tools {

void registerTlaCompileDialectsAndTranslations(DialectRegistry &registry) {
  registry.insert<arith::ArithDialect, mlir::DLTIDialect, func::FuncDialect, scf::SCFDialect,
                  LLVM::LLVMDialect, ::mlir::memref::MemRefDialect, ::tla::TlaDialect>();
  registry.insert<hacc::HACCDialect, hivm::HIVMDialect>();
  registerTlaCompileTranslationsAndInterfaces(registry);
}

void registerTlaCompileTranslationsAndInterfaces(DialectRegistry &registry) {
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
}

void loadTlaCompileDialects(MLIRContext &context) {
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<mlir::DLTIDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();
  context.getOrLoadDialect<::mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<::tla::TlaDialect>();
  context.getOrLoadDialect<hacc::HACCDialect>();
  context.getOrLoadDialect<hivm::HIVMDialect>();
}

void buildTlaCompilePassManagers(MLIRContext &context, PassManager &tlaPm, PassManager &llvmPm) {
  (void)context;
  ::tla::buildTlaPipeline(tlaPm);
  llvmPm.addPass(mlir::createConvertFuncToLLVMPass());
  llvmPm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  llvmPm.addPass(mlir::createArithToLLVMConversionPass());
  llvmPm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

bool runTlaCompilePipelines(ModuleOp module, llvm::StringRef emitMode, std::string &output,
                            std::string &error) {
  MLIRContext *context = module.getContext();
  PassManager tlaPm(context);
  PassManager llvmPm(context);
  buildTlaCompilePassManagers(*context, tlaPm, llvmPm);
  return runTlaCompilePipelinesWithManagers(module, emitMode, tlaPm, llvmPm, output, error,
                                            /*rewriteTileSignaturesToLLVMPointer=*/
                                            false);
}

bool runTlaCompilePipelinesWithManagers(ModuleOp module, llvm::StringRef emitMode,
                                        PassManager &tlaPm, PassManager &llvmPm,
                                        std::string &output, std::string &error,
                                        bool rewriteTileSignaturesToLLVMPointer) {
  if (failed(tlaPm.run(module))) {
    error = "Failed to run Tla pipeline.";
    return false;
  }

  if (emitMode == "mlir") {
    llvm::raw_string_ostream os(output);
    module.print(os);
    os.flush();
    return true;
  }

  if (emitMode == "llvm") {
    llvm::StringMap<SmallVector<uint64_t, 8>> ubByteOffsetsByFunc;
    GMFunctionArgMap gmArgInfoByFunc = collectGMFunctionArgInfo(module);
    if (!prepareTlaMemrefsForLLVMTranslation(module, ubByteOffsetsByFunc, error))
      return false;
    if (rewriteTileSignaturesToLLVMPointer) {
      rewriteTlaTileTypesToLLVMPointer(module);
    } else {
      std::string tileDetail;
      if (hasUnloweredTlaTileSignatures(module, tileDetail)) {
        error = "Failed to run LLVM conversion pipeline: unresolved tla.tile_view types "
                "remain after Tla pipeline (" +
                tileDetail +
                "). Move tile-signature lowering into tla::buildTlaPipeline "
                "(mlir/lib/Passes.cpp).";
        return false;
      }
    }
    if (failed(llvmPm.run(module))) {
      error = "Failed to run LLVM conversion pipeline.";
      return false;
    }
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
      error = "Failed to translate to LLVM IR.";
      return false;
    }
    rewriteUBAllocationsToSyntheticPointers(*llvmModule, ubByteOffsetsByFunc);
    if (!rewriteCifaceCallsWithWrappers(*llvmModule, error))
      return false;
    if (!rewriteGMRawPointerABI(*llvmModule, gmArgInfoByFunc, error))
      return false;
    assignHivmDeclarationAttributes(*llvmModule);
    addHivmKernelAnnotations(*llvmModule);
    llvm::raw_string_ostream os(output);
    llvmModule->print(os, nullptr);
    os.flush();
    normalizeLegacyLLVMAttributeSpellings(output);
    return true;
  }

  error = "Unsupported emit mode; expected 'mlir' or 'llvm'.";
  return false;
}

} // namespace tla::tools
