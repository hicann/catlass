#include "Dialect/Tla/IR/TlaOps.h"
#include "PassesInternal.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace tla {
namespace {

class TlaRejectDebugPrintPass
    : public mlir::PassWrapper<TlaRejectDebugPrintPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaRejectDebugPrintPass)

  llvm::StringRef getArgument() const final { return "tla-reject-debug-print"; }
  llvm::StringRef getDescription() const final {
    return "Reject frontend-only tla.debug_print before backend lowering";
  }

  void runOnOperation() final {
    auto result = getOperation().walk([&](DebugPrintOp op) {
      op.emitError("tla.debug_print backend lowering is not implemented; only "
                   "TLA IR emission is supported");
      return mlir::WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createTlaRejectDebugPrintPass() {
  return std::make_unique<TlaRejectDebugPrintPass>();
}

void registerTlaRejectDebugPrintPass() {
  mlir::PassRegistration<TlaRejectDebugPrintPass>();
}

} // namespace tla
