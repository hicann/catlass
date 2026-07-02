#include "PassesCommon.h"

namespace tla {
namespace {

// CTRL[48] is for saturation control of FP8/Hif8/FP16/BF16 computation in
// CUBE/FIXPIPE/VECTOR/SCALAR /AIPP/WAIPP.
static constexpr unsigned int SaturationControlBit = 48;

// CTRL[60] is the control bit to override saturation behavior for Vector Thread
// Extension Instructions SIMD.VCVTFI, SIMD.VCVTII, SIMD.VCVTFF and SIMT.F2F.
static constexpr unsigned int OverrideSaturationBit = 60;

// CTRL[56] is the mask-control bit; it must be cleared on AIV (vector) kernels.
static constexpr unsigned int MaskControlBit = 56;

class TlaPrologueEpiloguePass
    : public PassWrapper<TlaPrologueEpiloguePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaPrologueEpiloguePass)

  StringRef getArgument() const override { return "tla-prologue-epilogue"; }
  StringRef getName() const override { return "TlaPrologueEpiloguePass"; }
  StringRef getDescription() const override {
    return "Add HIVM kernel prologue and epilogue operations.";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);
    auto pipeAll = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);

    for (func::FuncOp funcOp : getOperation().getOps<func::FuncOp>()) {
      if (funcOp.isDeclaration())
        continue;
      if (!hasRequiredHaccEntryAttrs(funcOp))
        continue;

      Block &entry = funcOp.getBody().front();
      Location loc = funcOp.getLoc();

      builder.setInsertionPointToStart(&entry);
      builder.create<hivm::SetCtrlOp>(loc, false, OverrideSaturationBit);
      builder.create<hivm::SetCtrlOp>(loc, true, SaturationControlBit);
      if (getExpectedFunctionCoreKind(funcOp) == HivmCoreKind::AIV)
        builder.create<hivm::SetCtrlOp>(loc, false, MaskControlBit);

      Operation *terminator = entry.getTerminator();
      Operation *lastBodyOp = terminator ? terminator->getPrevNode() : nullptr;
      if (auto barrier = llvm::dyn_cast_or_null<hivm::PipeBarrierOp>(lastBodyOp)) {
        if (barrier.getPipe().getPipe() == hivm::PIPE::PIPE_ALL)
          continue;
      }

      if (terminator)
        builder.setInsertionPoint(terminator);
      else
        builder.setInsertionPointToEnd(&entry);
      builder.create<hivm::PipeBarrierOp>(loc, pipeAll);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTlaPrologueEpiloguePass() {
  return std::make_unique<TlaPrologueEpiloguePass>();
}

void registerTlaPrologueEpiloguePass() {
  PassRegistration<TlaPrologueEpiloguePass>();
}

} // namespace tla
