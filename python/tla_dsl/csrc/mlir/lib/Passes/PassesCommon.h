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
using namespace mlir;

namespace tla {

namespace {

static constexpr StringLiteral kDltiTargetSystemSpecAttrName = "dlti.target_system_spec";
static constexpr StringLiteral kTlaExecUnitsAttrName = "tla.exec_units";
static constexpr StringLiteral kTlaModuleExecUnitsAttrName = "tla.module_exec_units";
static constexpr StringLiteral kTlaHasVectorRegionAttrName = "tla.has_vector_region";
static constexpr StringLiteral kMixModeAttrName = "mix_mode";
static constexpr StringLiteral kParallelModeAttrName = "parallel_mode";

enum class HivmCoreKind { AIC, AIV, MIX };

static hivm::TModuleCoreType toModuleCoreType(HivmCoreKind coreKind) {
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

static hivm::TFuncCoreType toFuncCoreType(HivmCoreKind coreKind) {
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

static HivmCoreKind fromModuleCoreType(hivm::TModuleCoreType coreType) {
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

static std::optional<HivmCoreKind> getModuleCoreKind(ModuleOp module) {
  if (!module)
    return std::nullopt;
  auto attr = module->getAttrOfType<hivm::TModuleCoreTypeAttr>(hivm::TModuleCoreTypeAttr::name);
  if (!attr)
    return std::nullopt;
  return fromModuleCoreType(attr.getModuleCoreType());
}

static HivmCoreKind promoteCoreKind(std::optional<HivmCoreKind> current, HivmCoreKind observed) {
  if (!current)
    return observed;
  if (*current == observed || *current == HivmCoreKind::MIX)
    return *current;
  return HivmCoreKind::MIX;
}

static std::optional<HivmCoreKind> coreKindFromExecUnitsAttr(Operation *op, StringRef attrName) {
  if (!op)
    return std::nullopt;
  auto attr = op->getAttrOfType<StringAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return llvm::StringSwitch<std::optional<HivmCoreKind>>(attr.getValue())
      .Case("cube", HivmCoreKind::AIC)
      .Case("vector", HivmCoreKind::AIV)
      .Case("cube_vector", HivmCoreKind::MIX)
      .Default(std::nullopt);
}

static std::optional<HivmCoreKind> coreKindFromExecUnitsAttr(Operation *op) {
  return coreKindFromExecUnitsAttr(op, kTlaExecUnitsAttrName);
}

static bool moduleCoreAllows(ModuleOp module, HivmCoreKind observed) {
  if (!module)
    return false;
  std::optional<HivmCoreKind> current = getModuleCoreKind(module);
  if (!current)
    return false;
  return *current == observed || *current == HivmCoreKind::MIX;
}

static bool isPrivateSymbol(Operation *op) {
  if (auto visibility = op->getAttrOfType<StringAttr>(SymbolTable::getVisibilityAttrName()))
    return visibility.getValue() == "private";
  return false;
}

static std::optional<HivmCoreKind> getExpectedFunctionCoreKind(Operation *op) {
  if (std::optional<HivmCoreKind> hinted = coreKindFromExecUnitsAttr(op))
    return hinted;
  return getModuleCoreKind(op->getParentOfType<ModuleOp>());
}

static void applyTlaExecUnitHints(ModuleOp module) {
  std::optional<HivmCoreKind> hinted =
      coreKindFromExecUnitsAttr(module.getOperation(), kTlaModuleExecUnitsAttrName);
  auto accumulateHint = [&](Operation *op) {
    if (isPrivateSymbol(op))
      return;
    std::optional<HivmCoreKind> funcHint = coreKindFromExecUnitsAttr(op);
    if (!funcHint)
      return;
    hinted = promoteCoreKind(hinted, *funcHint);
  };
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>())
    accumulateHint(funcOp);
  for (::tla::FuncOp funcOp : module.getOps<::tla::FuncOp>())
    accumulateHint(funcOp);
  if (!hinted)
    return;
  std::optional<HivmCoreKind> current = getModuleCoreKind(module);
  HivmCoreKind promoted = promoteCoreKind(current, *hinted);
  if (current && *current == promoted)
    return;
  module->setAttr(hivm::TModuleCoreTypeAttr::name,
                  hivm::TModuleCoreTypeAttr::get(module.getContext(), toModuleCoreType(promoted)));
}

static bool hasC310TargetAttrs(ModuleOp module) {
  auto targetAttr = module->getAttrOfType<hacc::TargetAttr>(hacc::TargetAttr::name);
  return targetAttr && targetAttr.getTarget().getValue() == "Ascend950PR_9589" &&
         module->hasAttr(kDltiTargetSystemSpecAttrName);
}

static void ensureC310TargetAttrs(ModuleOp module) {
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

static LogicalResult setModuleCoreKind(PatternRewriter &rewriter, ModuleOp module,
                                       HivmCoreKind coreKind) {
  std::optional<HivmCoreKind> current = getModuleCoreKind(module);
  HivmCoreKind promoted = promoteCoreKind(current, coreKind);
  if (current && *current == promoted)
    return failure();

  MLIRContext *ctx = module.getContext();
  rewriter.modifyOpInPlace(module, [&] {
    module->setAttr(hivm::TModuleCoreTypeAttr::name,
                    hivm::TModuleCoreTypeAttr::get(ctx, toModuleCoreType(promoted)));
  });
  return success();
}

static bool hasRequiredHaccEntryAttrs(Operation *op) {
  auto functionKind = op->getAttrOfType<hacc::HACCFuncTypeAttr>(hacc::HACCFuncTypeAttr::name);
  return op->hasAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY)) && functionKind &&
         functionKind.getFunctionKind() == hacc::HACCFuncType::DEVICE;
}

static bool shouldOmitPureVectorEntryCoreAttrs(Operation *op, HivmCoreKind coreKind) {
  if (!op || coreKind != HivmCoreKind::AIV)
    return false;
  if (isPrivateSymbol(op))
    return false;
  return true;
}

static void setRequiredHaccEntryAttrs(Operation *op, MLIRContext *ctx) {
  op->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY), UnitAttr::get(ctx));
  op->setAttr(hacc::HACCFuncTypeAttr::name,
              hacc::HACCFuncTypeAttr::get(ctx, hacc::HACCFuncType::DEVICE));
}
static bool hasExpectedFunctionAttrs(Operation *op, HivmCoreKind coreKind) {
  MLIRContext *ctx = op->getContext();
  if (shouldOmitPureVectorEntryCoreAttrs(op, coreKind)) {
    return hasRequiredHaccEntryAttrs(op) && !op->hasAttr(hivm::TFuncCoreTypeAttr::name) &&
           !op->hasAttr(kMixModeAttrName) && !op->hasAttr(kParallelModeAttrName);
  }
  StringRef expectedMixMode = coreKind == HivmCoreKind::AIV ? "aiv" : "mix";
  auto functionKind = op->getAttrOfType<hacc::HACCFuncTypeAttr>(hacc::HACCFuncTypeAttr::name);
  auto functionCoreType = op->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
  auto mixMode = op->getAttrOfType<StringAttr>(kMixModeAttrName);
  auto parallelMode = op->getAttrOfType<StringAttr>(kParallelModeAttrName);
  return op->hasAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY)) && functionKind &&
         functionKind.getFunctionKind() == hacc::HACCFuncType::DEVICE && functionCoreType &&
         functionCoreType.getFuncCoreType() == toFuncCoreType(coreKind) && mixMode &&
         mixMode == StringAttr::get(ctx, expectedMixMode) && parallelMode &&
         parallelMode == StringAttr::get(ctx, "simd");
}

static LogicalResult ensureFunctionAttrs(PatternRewriter &rewriter, Operation *op,
                                         HivmCoreKind coreKind) {
  if (isPrivateSymbol(op))
    return failure();
  if (hasExpectedFunctionAttrs(op, coreKind))
    return failure();

  if (coreKind == HivmCoreKind::AIV && !coreKindFromExecUnitsAttr(op)) {
    MLIRContext *ctx = op->getContext();
    rewriter.modifyOpInPlace(op, [&] { setRequiredHaccEntryAttrs(op, ctx); });
    return success();
  }

  if (shouldOmitPureVectorEntryCoreAttrs(op, coreKind)) {
    MLIRContext *ctx = op->getContext();
    rewriter.modifyOpInPlace(op, [&] { setRequiredHaccEntryAttrs(op, ctx); });
    return success();
  }

  MLIRContext *ctx = op->getContext();
  StringRef mixMode = coreKind == HivmCoreKind::AIV ? "aiv" : "mix";
  Attribute functionKind = hacc::HACCFuncTypeAttr::get(ctx, hacc::HACCFuncType::DEVICE);
  Attribute functionCoreType = hivm::TFuncCoreTypeAttr::get(ctx, toFuncCoreType(coreKind));

  rewriter.modifyOpInPlace(op, [&] {
    op->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY), UnitAttr::get(ctx));
    op->setAttr(hacc::HACCFuncTypeAttr::name, functionKind);
    op->setAttr(hivm::TFuncCoreTypeAttr::name, functionCoreType);
    op->setAttr(kMixModeAttrName, StringAttr::get(ctx, mixMode));
    op->setAttr(kParallelModeAttrName, StringAttr::get(ctx, "simd"));
  });
  return success();
}

template <typename OpTy, HivmCoreKind coreKind>
struct ObserveCoreOpPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    if (moduleCoreAllows(module, coreKind))
      return failure();
    if (failed(setModuleCoreKind(rewriter, module, coreKind)))
      return failure();
    // Dialect conversion requires a successful pattern to replace or update
    // its root operation. This pattern observes the root op but mutates the
    // module-level core attribute that makes the root dynamically legal.
    rewriter.modifyOpInPlace(op, [] {});
    return success();
  }
};

struct DefaultModuleCoreTypePattern : public OpRewritePattern<ModuleOp> {
  using OpRewritePattern<ModuleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModuleOp module, PatternRewriter &rewriter) const override {
    if (getModuleCoreKind(module))
      return failure();
    return setModuleCoreKind(rewriter, module, HivmCoreKind::AIC);
  }
};

template <typename FuncOpTy> struct EnsureFunctionAttrsPattern : public OpRewritePattern<FuncOpTy> {
  using OpRewritePattern<FuncOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOpTy op, PatternRewriter &rewriter) const override {
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    HivmCoreKind coreKind =
        getExpectedFunctionCoreKind(op.getOperation()).value_or(HivmCoreKind::AIC);
    return ensureFunctionAttrs(rewriter, op.getOperation(), coreKind);
  }
};

struct LowerTlaReturnToFuncReturnPattern : public OpRewritePattern<::tla::ReturnOp> {
  using OpRewritePattern<::tla::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::tla::ReturnOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

static bool functionAttrsAreLegal(Operation *op) {
  if (isPrivateSymbol(op))
    return true;
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module)
    return true;
  std::optional<HivmCoreKind> coreKind = getExpectedFunctionCoreKind(op);
  return coreKind && hasExpectedFunctionAttrs(op, *coreKind);
}

static LogicalResult applyHaccHivmC310AttrPatterns(ModuleOp module, MLIRContext *ctx) {
  applyTlaExecUnitHints(module);
  {
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<ModuleOp>(hasC310TargetAttrs);
    target.addDynamicallyLegalOp<::tla::CubeOp, ::tla::MmadOp>([](Operation *op) {
      return moduleCoreAllows(op->getParentOfType<ModuleOp>(), HivmCoreKind::AIC);
    });
    target.addDynamicallyLegalOp<::tla::VectorOp, ::tla::AddOp>([](Operation *op) {
      return moduleCoreAllows(op->getParentOfType<ModuleOp>(), HivmCoreKind::AIV);
    });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(ctx);
    patterns.add<ObserveCoreOpPattern<::tla::CubeOp, HivmCoreKind::AIC>,
                 ObserveCoreOpPattern<::tla::MmadOp, HivmCoreKind::AIC>,
                 ObserveCoreOpPattern<::tla::VectorOp, HivmCoreKind::AIV>,
                 ObserveCoreOpPattern<::tla::AddOp, HivmCoreKind::AIV>>(ctx);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return failure();
  }

  {
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<ModuleOp>(
        [](ModuleOp module) { return getModuleCoreKind(module).has_value(); });
    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp op) { return functionAttrsAreLegal(op); });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(ctx);
    patterns.add<DefaultModuleCoreTypePattern, EnsureFunctionAttrsPattern<func::FuncOp>>(ctx);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return failure();
  }

  return success();
}

static FailureOr<hivm::AddressSpace> mapTlaAddressSpaceToHivm(AddressSpace addressSpace) {
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

static FailureOr<Attribute> mapTlaAddressSpaceToHivmMemspace(MLIRContext *ctx,
                                                             AddressSpace addressSpace) {
  FailureOr<hivm::AddressSpace> hivmSpace = mapTlaAddressSpaceToHivm(addressSpace);
  if (failed(hivmSpace))
    return failure();
  return hivm::AddressSpaceAttr::get(ctx, *hivmSpace);
}

static bool hasZeroStaticCoords(ArrayRef<int64_t> coords) {
  return llvm::all_of(coords, [](int64_t coord) { return coord == 0; });
}

static bool hasDefaultRowMajorStrides(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
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

static FailureOr<MemRefType> bridgeTlaFuncMemrefType(Type tlaMemrefType) {
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

static FailureOr<MemRefType> bridgeTlaFuncTensorType(Type tlaTensorType) {
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

} // namespace
} // namespace tla
