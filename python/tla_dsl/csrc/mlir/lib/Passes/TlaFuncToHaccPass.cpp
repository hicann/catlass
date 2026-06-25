#include "PassesCommon.h"

namespace tla {
namespace {

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

class TlaFuncToHaccPass : public PassWrapper<TlaFuncToHaccPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaFuncToHaccPass)

  StringRef getArgument() const override { return "tla-func-to-hacc"; }
  StringRef getName() const override { return "TlaFuncToHaccPass"; }
  StringRef getDescription() const override {
    return "Lower tla.func containers to func.func and attach C310 module target "
           "attributes. The per-function HACC/HIVM attributes are stamped earlier "
           "by tla-infer-func-core-type.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::DLTIDialect, hacc::HACCDialect, hivm::HIVMDialect, func::FuncDialect,
                    ::tla::TlaDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(lowerTlaFuncContainers(module, &getContext()))) {
      signalPassFailure();
      return;
    }
    ensureC310TargetAttrs(module);
  }
};

} // namespace

std::unique_ptr<Pass> createTlaFuncToHaccPass() { return std::make_unique<TlaFuncToHaccPass>(); }

void registerTlaFuncToHaccPass() { PassRegistration<TlaFuncToHaccPass>(); }

} // namespace tla
