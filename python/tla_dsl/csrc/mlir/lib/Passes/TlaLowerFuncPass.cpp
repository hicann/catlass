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

// Lower the tla.func / tla.return containers to func.func / func.return, copying
// the (non-signature) HACC/HIVM attributes stamped above onto the new func.func.
static LogicalResult lowerTlaFuncContainers(ModuleOp module, MLIRContext *ctx) {
  ConversionTarget target(*ctx);
  target.addLegalDialect<func::FuncDialect, ::tla::TlaDialect>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addIllegalOp<::tla::FuncOp, ::tla::ReturnOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<LowerTlaFuncToFuncPattern<LowerTlaFuncToFuncAttrPolicy::CopyNonSignatureAttrs>,
               LowerTlaReturnToFuncReturnPattern>(ctx);
  return applyPartialConversion(module, target, std::move(patterns));
}

// Lower tla device functions to HACC func.func in a single step: classify each
// device function's core type (AIC/AIV/MIX) from the tla.cube / tla.vector
// regions it contains, stamp the per-function HACC/HIVM entry metadata, lower
// the tla.func containers to func.func, and tag the module core type + C310
// target.
//
// Region placement is mandatory (enforced by the tla op verifiers: tla.mmad and
// cube-path copies live in tla.cube; tla.vec.func, the vector compute ops, and
// GM<->UB copies live in tla.vector), so region presence alone determines the
// core type:
//
//   both tla.cube and tla.vector present -> MIX
//   tla.cube only                        -> AIC
//   tla.vector only, or no region at all  -> AIV
//
// This runs before TlaSplitMixedFuncPass, so every function still has its
// frontend regions intact here; the split fragments get their core type stamped
// by that pass directly and are not re-classified.
class TlaLowerFuncPass : public PassWrapper<TlaLowerFuncPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerFuncPass)

  StringRef getArgument() const override { return "tla-lower-func"; }
  StringRef getName() const override { return "TlaLowerFuncPass"; }
  StringRef getDescription() const override {
    return "Lower tla.func device containers to HACC func.func: classify AIC/AIV/MIX "
           "from tla.cube/tla.vector regions, stamp the per-function HACC/HIVM entry "
           "attributes, and attach the module core type and C310 target attributes.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::DLTIDialect, hacc::HACCDialect, hivm::HIVMDialect, func::FuncDialect,
                    ::tla::TlaDialect>();
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

  HivmCoreKind inferFuncCoreKind(Operation *funcOp) {
    bool hasVector = false, hasCube = false;
    funcOp->walk([&](Operation *op) {
      if (isa<::tla::VectorOp>(op))
        hasVector = true;
      else if (isa<::tla::CubeOp>(op))
        hasCube = true;
      // MIX is terminal: once both regions are seen, no later op can change
      // the decision, so stop walking.
      return (hasVector && hasCube) ? WalkResult::interrupt()
                                    : WalkResult::advance();
    });

    if (hasVector && hasCube)
      return HivmCoreKind::MIX;
    if (hasCube)
      return HivmCoreKind::AIC;
    // tla.vector only, or a region-less (empty / sync-only) function -> AIV.
    return HivmCoreKind::AIV;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // 1. Classify + stamp each device function, aggregating the module core type.
    //    Done on the tla.func containers so the attrs are carried onto func.func
    //    by the CopyNonSignatureAttrs lowering below.
    std::optional<HivmCoreKind> moduleKind;
    auto classify = [&](Operation *funcOp, Region &body) {
      if (!isInferableFunc(funcOp, body))
        return;
      HivmCoreKind funcKind = inferFuncCoreKind(funcOp);
      stampFunctionHaccHivmAttrs(funcOp, funcKind);
      moduleKind = promoteCoreKind(moduleKind, funcKind);
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

    // 2. Lower the tla.func containers to func.func and attach the C310 module
    //    target attributes.
    if (failed(lowerTlaFuncContainers(module, ctx))) {
      signalPassFailure();
      return;
    }
    ensureC310TargetAttrs(module);
  }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerFuncPass() { return std::make_unique<TlaLowerFuncPass>(); }

void registerTlaLowerFuncPass() { PassRegistration<TlaLowerFuncPass>(); }

} // namespace tla
