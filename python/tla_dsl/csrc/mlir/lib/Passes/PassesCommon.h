#pragma once

#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaOps.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "Tools/AddressSpaceConversion.h"

#include "Passes.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <utility>

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
using namespace mlir;

namespace tla {

inline constexpr StringLiteral kDltiTargetSystemSpecAttrName = "dlti.target_system_spec";
inline constexpr StringLiteral kMixModeAttrName = "mix_mode";
inline constexpr StringLiteral kParallelModeAttrName = "parallel_mode";
inline constexpr StringLiteral kHivmVectorFunctionAttrName = "hivm.vector_function";

enum class HivmCoreKind { AIC, AIV, MIX };

inline hivm::TModuleCoreType toModuleCoreType(HivmCoreKind coreKind) {
  switch (coreKind) {
  case HivmCoreKind::AIC:
    return hivm::TModuleCoreType::AIC;
  case HivmCoreKind::AIV:
    return hivm::TModuleCoreType::AIV;
  case HivmCoreKind::MIX:
    return hivm::TModuleCoreType::MIX;
  }
  llvm_unreachable("unknown HIVM core kind");
}

inline hivm::TFuncCoreType toFuncCoreType(HivmCoreKind coreKind) {
  switch (coreKind) {
  case HivmCoreKind::AIC:
    return hivm::TFuncCoreType::AIC;
  case HivmCoreKind::AIV:
    return hivm::TFuncCoreType::AIV;
  case HivmCoreKind::MIX:
    return hivm::TFuncCoreType::MIX;
  }
  llvm_unreachable("unknown HIVM core kind");
}

inline HivmCoreKind fromModuleCoreType(hivm::TModuleCoreType coreType) {
  switch (coreType) {
  case hivm::TModuleCoreType::AIC:
    return HivmCoreKind::AIC;
  case hivm::TModuleCoreType::AIV:
    return HivmCoreKind::AIV;
  case hivm::TModuleCoreType::MIX:
    return HivmCoreKind::MIX;
  }
  llvm_unreachable("unknown HIVM module core type");
}

inline HivmCoreKind fromFuncCoreType(hivm::TFuncCoreType coreType) {
  switch (coreType) {
  case hivm::TFuncCoreType::AIC:
    return HivmCoreKind::AIC;
  case hivm::TFuncCoreType::AIV:
    return HivmCoreKind::AIV;
  case hivm::TFuncCoreType::MIX:
    return HivmCoreKind::MIX;
  }
  llvm_unreachable("unknown HIVM func core type");
}

inline std::optional<HivmCoreKind> getModuleCoreKind(ModuleOp module) {
  if (!module)
    return std::nullopt;
  auto attr = module->getAttrOfType<hivm::TModuleCoreTypeAttr>(hivm::TModuleCoreTypeAttr::name);
  if (!attr)
    return std::nullopt;
  return fromModuleCoreType(attr.getModuleCoreType());
}

// Single source of truth: classify the execution unit a tla op requires.
// Returns std::nullopt for ops that do not constrain the core type.
//  - cube core (AIC): tla.cube / tla.mmad, or on-chip scratch allocated in L1.
//  - vector core (AIV): tla.vector / tla.vec.func, or on-chip scratch allocated
//    in UB.
inline std::optional<HivmCoreKind> getTlaOpCoreKind(Operation *op) {
  if (!op)
    return std::nullopt;
  if (isa<::tla::CubeOp, ::tla::MmadOp>(op))
    return HivmCoreKind::AIC;
  if (isa<::tla::VectorOp, ::tla::VecFuncOp>(op))
    return HivmCoreKind::AIV;
  if (auto alloc = dyn_cast<::tla::AllocPtrOp>(op)) {
    if (auto ptrTy = dyn_cast<::tla::PtrType>(alloc.getResult().getType())) {
      if (ptrTy.getAddrspace() == AddressSpace::ub)
        return HivmCoreKind::AIV;
      if (ptrTy.getAddrspace() == AddressSpace::l1)
        return HivmCoreKind::AIC;
    }
  }
  return std::nullopt;
}

// The function's core-type hint, persisted as the typed hivm.func_core_type
// attribute by TlaInferFuncCoreTypePass. This is the single carrier of the
// per-function AIC/AIV/MIX classification through the lowering pipeline.
inline std::optional<HivmCoreKind> funcCoreKindHint(Operation *op) {
  if (!op)
    return std::nullopt;
  auto attr = op->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
  if (!attr)
    return std::nullopt;
  return fromFuncCoreType(attr.getFuncCoreType());
}


inline bool isPrivateSymbol(Operation *op) {
  if (auto visibility = op->getAttrOfType<StringAttr>(SymbolTable::getVisibilityAttrName()))
    return visibility.getValue() == "private";
  return false;
}

inline std::optional<HivmCoreKind> getExpectedFunctionCoreKind(Operation *op) {
  if (std::optional<HivmCoreKind> hinted = funcCoreKindHint(op))
    return hinted;
  return getModuleCoreKind(op->getParentOfType<ModuleOp>());
}

inline bool hasC310TargetAttrs(ModuleOp module) {
  auto targetAttr = module->getAttrOfType<hacc::TargetAttr>(hacc::TargetAttr::name);
  return targetAttr && targetAttr.getTarget().getValue() == "Ascend950PR_9589" &&
         module->hasAttr(kDltiTargetSystemSpecAttrName);
}

inline void ensureC310TargetAttrs(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);
  module->setAttr(hacc::TargetAttr::name,
                  hacc::TargetAttr::get(ctx, StringAttr::get(ctx, "Ascend950PR_9589")));

  SmallVector<std::pair<StringRef, Attribute>, 14> specs = {
      {"AI_CORE_COUNT", builder.getI32IntegerAttr(32)},
      {"CUBE_CORE_COUNT", builder.getI32IntegerAttr(32)},
      {"VECTOR_CORE_COUNT", builder.getI32IntegerAttr(64)},
      {"UB_SIZE", builder.getI32IntegerAttr(2031616)},
      {"L1_SIZE", builder.getI32IntegerAttr(4194304)},
      {"L0A_SIZE", builder.getI32IntegerAttr(524288)},
      {"L0B_SIZE", builder.getI32IntegerAttr(524288)},
      {"L0C_SIZE", builder.getI32IntegerAttr(2097152)},
      {"UB_ALIGN_SIZE", builder.getI32IntegerAttr(256)},
      {"L1_ALIGN_SIZE", builder.getI32IntegerAttr(256)},
      {"L0C_ALIGN_SIZE", builder.getI32IntegerAttr(4096)},
      {"MINIMAL_D_CACHE_SIZE", builder.getI32IntegerAttr(262144)},
      {"MAXIMUM_D_CACHE_SIZE", builder.getI32IntegerAttr(983040)},
      {"ARCH", builder.getStringAttr("dav-c310")},
  };
  SmallVector<DataLayoutEntryInterface> entries;
  for (auto [name, value] : specs) {
    entries.push_back(DataLayoutEntryAttr::get(builder.getStringAttr(name), value));
  }

  auto targetSpec =
      cast<hacc::HACCTargetDeviceSpecInterface>(hacc::TargetDeviceSpecAttr::get(ctx, entries));
  hacc::utils::setNPUTargetSpec(module, targetSpec);
}

inline bool hasRequiredHaccEntryAttrs(Operation *op) {
  auto functionKind = op->getAttrOfType<hacc::HACCFuncTypeAttr>(hacc::HACCFuncTypeAttr::name);
  return op->hasAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY)) && functionKind &&
         functionKind.getFunctionKind() == hacc::HACCFuncType::DEVICE;
}

inline void setRequiredHaccEntryAttrs(Operation *op, MLIRContext *ctx) {
  op->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY), UnitAttr::get(ctx));
  op->setAttr(hacc::HACCFuncTypeAttr::name,
              hacc::HACCFuncTypeAttr::get(ctx, hacc::HACCFuncType::DEVICE));
}

inline void setC310RegbaseTargetAttr(Operation *op, MLIRContext *ctx) {
  op->setAttr(StringAttr::get(ctx, hivm_regbaseintrins::kDavinciTargetAttrName),
              hivm_regbaseintrins::SIMT_TargetAttr::get(ctx, "dav-c310"));
}

struct LowerTlaReturnToFuncReturnPattern : public OpRewritePattern<::tla::ReturnOp> {
  using OpRewritePattern<::tla::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::tla::ReturnOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

inline FailureOr<hivm::AddressSpace> mapTlaAddressSpaceToHivm(AddressSpace addressSpace) {
  switch (addressSpace) {
  case AddressSpace::generic:
    return hivm::AddressSpace::Zero;
  case AddressSpace::gm:
    return hivm::AddressSpace::GM;
  case AddressSpace::l1:
    return hivm::AddressSpace::L1;
  case AddressSpace::l0a:
    return hivm::AddressSpace::L0A;
  case AddressSpace::l0b:
    return hivm::AddressSpace::L0B;
  case AddressSpace::l0c:
    return hivm::AddressSpace::L0C;
  case AddressSpace::ub:
    return hivm::AddressSpace::UB;
  }
  return failure();
}

inline FailureOr<Attribute> mapTlaAddressSpaceToHivmMemspace(MLIRContext *ctx,
                                                             AddressSpace addressSpace) {
  FailureOr<hivm::AddressSpace> hivmSpace = mapTlaAddressSpaceToHivm(addressSpace);
  if (failed(hivmSpace))
    return failure();
  return hivm::AddressSpaceAttr::get(ctx, *hivmSpace);
}

inline bool hasZeroStaticCoords(ArrayRef<int64_t> coords) {
  return llvm::all_of(coords, [](int64_t coord) { return coord == 0; });
}

inline bool hasDefaultRowMajorStrides(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
  if (shape.empty() || shape.size() != strides.size())
    return false;

  int64_t expectedStride = 1;
  for (size_t index = shape.size(); index-- > 0;) {
    if (shape[index] == ShapedType::kDynamic || strides[index] == ShapedType::kDynamic)
      return false;
    if (strides[index] != expectedStride)
      return false;
    if (shape[index] != 0 && expectedStride > std::numeric_limits<int64_t>::max() / shape[index])
      return false;
    expectedStride *= shape[index];
  }
  return true;
}

inline FailureOr<MemRefType> bridgeTlaFuncMemrefType(Type tlaMemrefType) {
  auto tlaMemref = dyn_cast<::tla::MemrefType>(tlaMemrefType);
  if (!tlaMemref)
    return failure();

  FailureOr<Attribute> memorySpaceOr =
      mapTlaAddressSpaceToHivmMemspace(tlaMemrefType.getContext(), tlaMemref.getAddressSpace());
  if (failed(memorySpaceOr))
    return failure();
  return MemRefType::get(tlaMemref.getShape(), tlaMemref.getElementType(), AffineMap(),
                         *memorySpaceOr);
}

inline FailureOr<MemRefType> bridgeTlaFuncTensorType(Type tlaTensorType) {
  SmallVector<int64_t, 4> shape;
  SmallVector<int64_t, 4> strides;
  SmallVector<int64_t, 4> coords;
  SmallVector<int64_t, 4> originShape;
  std::string elementTypeStorage;
  std::string addressSpaceStorage;
  std::string layoutTagStorage;
  StringRef elementTypeText;
  StringRef addressSpace;
  StringRef layoutTag;

  auto tensorTy = dyn_cast<::tla::TlaTensorType>(tlaTensorType);
  if (!tensorTy)
    return failure();
  auto layout = tensorTy.getLayout();
  auto ptr = tensorTy.getPtr();
  if (!layout.getOrigin() || !ptr.getPointee())
    return failure();
  if (failed(::tla::getTlaIndexTreeLeaves(layout.getShape().getTree(), shape)) ||
      failed(::tla::getTlaIndexTreeLeaves(layout.getStride().getTree(), strides)) ||
      failed(::tla::getTlaIndexTreeLeaves(tensorTy.getCoord().getTree(), coords)) ||
      failed(::tla::getTlaIndexTreeLeaves(layout.getOrigin().getTree(), originShape)))
    return failure();
  llvm::raw_string_ostream os(elementTypeStorage);
  os << ptr.getPointee();
  os.flush();
  elementTypeText = StringRef(elementTypeStorage).trim();
  addressSpaceStorage = stringifyAddressSpace(ptr.getAddrspace()).str();
  layoutTagStorage = stringifyLayoutTag(layout.getLayoutTag()).str();
  addressSpace = addressSpaceStorage;
  layoutTag = layoutTagStorage;

  if (shape.empty() || strides.empty() || coords.empty() || originShape.empty() ||
      elementTypeText.empty() || addressSpace.empty() || layoutTag.empty())
    return failure();

  if (layoutTag != "row_major" && originShape.size() == coords.size() &&
      shape.size() != originShape.size()) {
    shape = originShape;
  }
  if (strides.size() != shape.size())
    strides = shape;
  if (coords.size() != shape.size()) {
    if (originShape.size() == coords.size() && shape.size() == originShape.size()) {
      // Keep the parsed coordinates.
    } else {
      return failure();
    }
  }

  MLIRContext *ctx = tlaTensorType.getContext();
  Type elementType = ptr.getPointee();
  auto tlaAddressSpace = symbolizeAddressSpace(addressSpace);
  if (!tlaAddressSpace)
    return failure();
  FailureOr<Attribute> memorySpaceOr = mapTlaAddressSpaceToHivmMemspace(ctx, *tlaAddressSpace);
  if (failed(memorySpaceOr))
    return failure();

  if (layoutTag == "row_major" &&
      !(hasZeroStaticCoords(coords) && hasDefaultRowMajorStrides(shape, strides))) {
    auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides);
    return MemRefType::get(shape, elementType, layout, *memorySpaceOr);
  }
  return MemRefType::get(shape, elementType, AffineMap(), *memorySpaceOr);
}

/// Whether `tla.func` → `func.func` lowering copies attributes other than `sym_name` /
/// `function_type` onto the new `func.func` (HACC pipeline needs them; std lowering omits).
enum class LowerTlaFuncToFuncAttrPolicy { CopyNonSignatureAttrs, OmitAttrs };

template <LowerTlaFuncToFuncAttrPolicy AttrPolicy>
struct LowerTlaFuncToFuncPattern : public OpRewritePattern<::tla::FuncOp> {
  using OpRewritePattern<::tla::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::tla::FuncOp op, PatternRewriter &rewriter) const override {
    auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name");
    auto typeAttr = op->getAttrOfType<TypeAttr>("function_type");
    if (!symNameAttr || !typeAttr) {
      op.emitError() << "expected tla.func to have sym_name and function_type";
      return failure();
    }

    auto funcType = llvm::dyn_cast<FunctionType>(typeAttr.getValue());
    if (!funcType) {
      op.emitError() << "expected tla.func function_type to be a FunctionType";
      return failure();
    }

    SmallVector<Type, 8> bridgedInputs;
    bridgedInputs.reserve(funcType.getNumInputs());
    for (Type input : funcType.getInputs()) {
      FailureOr<MemRefType> bridged = bridgeTlaFuncMemrefType(input);
      if (failed(bridged))
        bridged = bridgeTlaFuncTensorType(input);
      bridgedInputs.push_back(succeeded(bridged) ? *bridged : input);
    }
    SmallVector<Type, 4> bridgedResults;
    bridgedResults.reserve(funcType.getNumResults());
    for (Type result : funcType.getResults()) {
      FailureOr<MemRefType> bridged = bridgeTlaFuncMemrefType(result);
      if (failed(bridged))
        bridged = bridgeTlaFuncTensorType(result);
      bridgedResults.push_back(succeeded(bridged) ? *bridged : result);
    }

    auto bridgedFuncType = rewriter.getFunctionType(bridgedInputs, bridgedResults);
    auto func = rewriter.create<func::FuncOp>(op.getLoc(), symNameAttr.getValue(), bridgedFuncType);
    if constexpr (AttrPolicy == LowerTlaFuncToFuncAttrPolicy::CopyNonSignatureAttrs) {
      for (NamedAttribute attr : op->getAttrs()) {
        StringRef name = attr.getName().getValue();
        if (name == "sym_name" || name == "function_type")
          continue;
        func->setAttr(attr.getName(), attr.getValue());
      }
    }
    func.getBody().takeBody(op.getRegion());
    for (auto [arg, type] : llvm::zip_equal(func.getArguments(), bridgedFuncType.getInputs())) {
      arg.setType(type);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace tla
