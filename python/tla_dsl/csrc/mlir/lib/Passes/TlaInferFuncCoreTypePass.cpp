#include "PassesCommon.h"
#include "PassesInternal.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace tla {
namespace {

// Aggregate two core kinds: same kind stays; differing non-MIX kinds (or any
// MIX) promote to MIX.
HivmCoreKind promoteCoreKind(std::optional<HivmCoreKind> current, HivmCoreKind observed) {
  if (!current)
    return observed;
  if (*current == observed || *current == HivmCoreKind::MIX)
    return *current;
  return HivmCoreKind::MIX;
}

// A pure-vector entry keeps only the HACC entry attrs in the final IR (no
// func_core_type / mix_mode / parallel_mode), per the pure-vector-entry
// attribute convention.
bool shouldOmitPureVectorEntryCoreAttrs(Operation *op, HivmCoreKind coreKind) {
  if (!op || coreKind != HivmCoreKind::AIV)
    return false;
  if (op->hasAttr("hivm.part_of_mix"))
    return false;
  if (isPrivateSymbol(op))
    return false;
  return true;
}

// Materialize the final HACC/HIVM entry attributes for a device function from
// its core kind (hacc.entry, function_kind, hivm.func_core_type, mix_mode,
// parallel_mode, and the C310 regbase target). Pure-vector entries are stripped
// back to just the entry attrs (plus the target). This is the single place that
// stamps the per-device-function HACC/HIVM entry metadata.
void stampFunctionHaccHivmAttrs(Operation *op, HivmCoreKind coreKind) {
  if (isPrivateSymbol(op))
    return;
  MLIRContext *ctx = op->getContext();
  setC310RegbaseTargetAttr(op, ctx);
  if (shouldOmitPureVectorEntryCoreAttrs(op, coreKind)) {
    setRequiredHaccEntryAttrs(op, ctx);
    op->removeAttr(hivm::TFuncCoreTypeAttr::name);
    op->removeAttr(kMixModeAttrName);
    op->removeAttr(kParallelModeAttrName);
    return;
  }
  StringRef mixMode =
      coreKind == HivmCoreKind::AIV && !op->hasAttr("hivm.part_of_mix") ? "aiv" : "mix";
  op->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY), UnitAttr::get(ctx));
  op->setAttr(hacc::HACCFuncTypeAttr::name,
              hacc::HACCFuncTypeAttr::get(ctx, hacc::HACCFuncType::DEVICE));
  op->setAttr(hivm::TFuncCoreTypeAttr::name,
              hivm::TFuncCoreTypeAttr::get(ctx, toFuncCoreType(coreKind)));
  op->setAttr(kMixModeAttrName, StringAttr::get(ctx, mixMode));
  op->setAttr(kParallelModeAttrName, StringAttr::get(ctx, "simd"));
}

// Infer each device function's core type (AIC/AIV/MIX) from the tla ops it
// contains, then aggregate the module core type. Mirrors the design of
// AscendNPU-IR-Dev's hivm InferFuncCoreType pass, but works on tla ops. The
// per-function decision is ordered:
//
//   1. both a tla.vector and a tla.cube region present  -> MIX;
//   1b. else both tla.mmad and tla.vec.func present      -> error: mixed work
//      without the regions needed to split it;
//   2. else tla.cube / tla.mmad present                 -> AIC;
//   3. else tla.vector / tla.vec.func present           -> AIV;
//   4. else fall back to on-chip scratch placement: any L1 alloc -> AIC,
//      only-UB allocs -> AIV. The fallback avoids misclassifying a split mixed
//      function whose shared allocations were duplicated into it, and only
//      matters when no op pins the core type.
class TlaInferFuncCoreTypePass
    : public PassWrapper<TlaInferFuncCoreTypePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaInferFuncCoreTypePass)

  StringRef getArgument() const override { return "tla-infer-func-core-type"; }
  StringRef getName() const override { return "TlaInferFuncCoreTypePass"; }
  StringRef getDescription() const override {
    return "Infer AIC/AIV/MIX core type for tla device functions and module.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, hacc::HACCDialect, hivm::HIVMDialect, ::tla::TlaDialect>();
  }

  // A function whose core type we should infer: skip declarations, private
  // helpers, and (once HACC-marked) host functions. Runs before HACC marking
  // too, where tla.func ops carry no function-kind attr yet.
  static bool isInferableFunc(Operation *funcOp, Region &body) {
    if (body.empty() || isPrivateSymbol(funcOp))
      return false;
    auto kind = funcOp->getAttrOfType<hacc::HACCFuncTypeAttr>(hacc::HACCFuncTypeAttr::name);
    if (kind && kind.getFunctionKind() != hacc::HACCFuncType::DEVICE)
      return false;
    return true;
  }

  std::optional<HivmCoreKind> inferFuncCoreKind(Operation *funcOp) {
    bool hasVector = false, hasCube = false, hasMmad = false, hasVecFunc = false;
    bool sawUbAlloc = false, sawL1Alloc = false;
    funcOp->walk([&](Operation *op) {
      if (isa<::tla::VectorOp>(op))
        hasVector = true;
      else if (isa<::tla::CubeOp>(op))
        hasCube = true;
      else if (isa<::tla::MmadOp>(op))
        hasMmad = true;
      else if (isa<::tla::VecFuncOp>(op))
        hasVecFunc = true;
      else if (auto alloc = dyn_cast<::tla::AllocPtrOp>(op)) {
        if (auto ptrTy = dyn_cast<::tla::PtrType>(alloc.getResult().getType())) {
          sawUbAlloc |= ptrTy.getAddrspace() == AddressSpace::ub;
          sawL1Alloc |= ptrTy.getAddrspace() == AddressSpace::l1;
        }
      }
      // MIX is terminal: once both regions are seen, no later op can change
      // the decision, so stop walking.
      return (hasVector && hasCube) ? WalkResult::interrupt()
                                    : WalkResult::advance();
    });

    // Region kind decides the core type; with no region, op presence does.
    //
    // Both regions -> mixed.
    if (hasVector && hasCube)
      return HivmCoreKind::MIX;
    // Cube region only -> AIC.
    else if (!hasVector && hasCube)
      return HivmCoreKind::AIC;
    // Vector region only -> AIV.
    else if (hasVector && !hasCube)
      return HivmCoreKind::AIV;
    // No region: decide from the ops instead.
    else if (!hasVector && !hasCube) {
      // Cube and vector ops but no regions to split on: reject.
      if (hasVecFunc && hasMmad) {
        funcOp->emitError()
            << "function has both tla.mmad (cube) and tla.vec.func (vector) work "
               "but lacks the matching tla.vector/tla.cube regions to mark it mixed";
        signalPassFailure();
        return std::nullopt;
      }
      // Cube op only -> AIC.
      else if (!hasVecFunc && hasMmad)
        return HivmCoreKind::AIC;
      // Vector op only -> AIV.
      else if (hasVecFunc && !hasMmad)
        return HivmCoreKind::AIV;
      // No op either: fall back to scratch (L1 -> AIC, UB -> AIV).
      else if (!hasVecFunc && !hasMmad) {
        if (sawL1Alloc)
          return HivmCoreKind::AIC;
        else if (sawUbAlloc)
          return HivmCoreKind::AIV;
      }
    }
    // Fallback for empty kernels
    return HivmCoreKind::AIV;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    std::optional<HivmCoreKind> moduleKind;

    auto classify = [&](Operation *funcOp, Region &body) {
      if (!isInferableFunc(funcOp, body))
        return;
      std::optional<HivmCoreKind> funcKind = inferFuncCoreKind(funcOp);
      if (!funcKind)
        return;
      // A function with both cube and vector work (MIX) is a mixed-split
      // candidate; that is derived downstream from func_core_type == MIX.
      stampFunctionHaccHivmAttrs(funcOp, *funcKind);
      moduleKind = promoteCoreKind(moduleKind, *funcKind);
    };

    for (::tla::FuncOp funcOp : module.getOps<::tla::FuncOp>())
      classify(funcOp, funcOp.getBody());
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>())
      classify(funcOp, funcOp.getBody());

    // Always tag the module core type, defaulting an empty / region-less module
    // to AIV -- the same fallback inferFuncCoreKind applies per function.
    HivmCoreKind resolvedModuleKind = moduleKind.value_or(HivmCoreKind::AIV);
    module->setAttr(hivm::TModuleCoreTypeAttr::name,
                    hivm::TModuleCoreTypeAttr::get(ctx, toModuleCoreType(resolvedModuleKind)));
  }
};

} // namespace

std::unique_ptr<Pass> createTlaInferFuncCoreTypePass() {
  return std::make_unique<TlaInferFuncCoreTypePass>();
}

void registerTlaInferFuncCoreTypePass() { PassRegistration<TlaInferFuncCoreTypePass>(); }

} // namespace tla
