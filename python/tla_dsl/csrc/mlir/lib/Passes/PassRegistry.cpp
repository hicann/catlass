#include "PassesCommon.h"
#include "PassesInternal.h"

#include "bishengir/Conversion/ArithToHIVMAVE/ArithToHIVMAVE.h"
#include "bishengir/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.h"
#include "bishengir/Conversion/HIVMAVEToStandard/HIVMAVEToStandard.h"
#include "bishengir/Conversion/HIVMToStandard/HIVMToStandard.h"
#include "bishengir/Conversion/VectorToHIVMAVE/VectorToHIVMAVE.h"

#include "mlir/Dialect/MemRef/Transforms/Passes.h"

namespace tla {
namespace {

// CTRL[48] is for saturation control of FP8/Hif8/FP16/BF16 computation in
// CUBE/FIXPIPE/VECTOR/SCALAR /AIPP/WAIPP.
static constexpr unsigned int SaturationControlBit = 48;
static constexpr unsigned int MaskControlBit = 56; // reserved

// CTRL[60] is the control bit to override saturation behavior for Vector Thread
// Extension Instructions SIMD.VCVTFI, SIMD.VCVTII, SIMD.VCVTFF and SIMT.F2F.
static constexpr unsigned int OverrideSaturationBit = 60;

class AddKernelPrologueEpiloguePass
    : public PassWrapper<AddKernelPrologueEpiloguePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddKernelPrologueEpiloguePass)

  StringRef getArgument() const override { return "add-kernel-prologue-epilogue"; }
  StringRef getName() const override { return "AddKernelPrologueEpiloguePass"; }
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

      Operation *terminator = entry.getTerminator();
      Operation *lastBodyOp = terminator ? terminator->getPrevNode() : nullptr;
      if (auto barrier = llvm::dyn_cast_or_null<hivm::PipeBarrierOp>(lastBodyOp)) {
        if (barrier.getPipe().getPipe() == hivm::PIPE::PIPE_ALL)
          return;
      }

      if (terminator)
        builder.setInsertionPoint(terminator);
      else
        builder.setInsertionPointToEnd(&entry);
      builder.create<hivm::PipeBarrierOp>(loc, pipeAll);
    }
  }
};

class EnsureC310RegbaseTargetAttrPass
    : public PassWrapper<EnsureC310RegbaseTargetAttrPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnsureC310RegbaseTargetAttrPass)

  StringRef getArgument() const override { return "tla-ensure-c310-regbase-target-attr"; }
  StringRef getName() const override { return "EnsureC310RegbaseTargetAttrPass"; }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto targetAttr = hivm_regbaseintrins::SIMT_TargetAttr::get(ctx, "dav-c310");

    getOperation().walk([&](func::FuncOp funcOp) {
      auto functionKind =
          funcOp->getAttrOfType<hacc::HACCFuncTypeAttr>(hacc::HACCFuncTypeAttr::name);
      if (!functionKind || functionKind.getFunctionKind() != hacc::HACCFuncType::DEVICE)
        return;
      funcOp->setAttr(hivm_regbaseintrins::kDavinciTargetAttrName, targetAttr);
    });
  }
};

} // namespace

std::unique_ptr<Pass> createAddKernelPrologueEpiloguePass() {
  return std::make_unique<AddKernelPrologueEpiloguePass>();
}

void registerTlaPasses() {
  registerTlaFuncToHaccPass();
  registerTlaSplitMixedFuncPass();
  registerTlaInferFuncCoreTypePass();
  registerTlaLowerToHivmPass();
  registerConvertTlaToVectorPass();
  registerTlaSyncToHivmPass();
  registerTlaAllocPtrToHivmPointerCastPass();
  registerTlaLowerMutexToStdPass();
  registerTlaLowerToStdPass();
  PassRegistration<AddKernelPrologueEpiloguePass>();
}

void buildTlaPipeline(OpPassManager &pm) {
  // Run before HIVM lowering so Tla function containers become func.func
  // before downstream passes and tools inspect the module. Run again after
  // tla-lower-to-std to repair required HACC/HIVM attrs on any func.func ops
  // introduced or modified by later lowering.
  // Establish AIC/AIV/MIX core types from the tla op structure up front, so the
  // HACC attribute machinery and the mixed-func split consume a single,
  // structure-derived classification instead of bespoke op observation.
  pm.addPass(createTlaInferFuncCoreTypePass());
  pm.addPass(createTlaFuncToHaccPass());
  pm.addPass(createTlaSplitMixedFuncPass());
  pm.addPass(createTlaLowerToHivmPass());
  pm.addPass(createTlaSyncToHivmPass());
  pm.addPass(createTlaAllocPtrToHivmPointerCastPass());
  pm.addPass(createConvertTlaToVectorPass());
  pm.addPass(createTlaLowerMutexToStdPass());
  pm.addPass(createTlaLowerToStdPass());
  pm.addPass(createAddKernelPrologueEpiloguePass());
  pm.addPass(createCSEPass());
  pm.addPass(mlir::createVectorToHIVMAVEConversionPass());
  pm.nest<func::FuncOp>().addPass(mlir::createArithToHIVMAVEConversionPass());
  mlir::ConvertHIVMToStandardOptions hivmToStdOptions;
  pm.addPass(mlir::createConvertHIVMToStandardPass(hivmToStdOptions));
  pm.addPass(mlir::createConvertHIVMAVEToStandardPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createConvertHIVMAVEToAVEIntrinPass());
  pm.addPass(std::make_unique<EnsureC310RegbaseTargetAttrPass>());
  pm.addPass(createConvertSCFToCFPass());
}

} // namespace tla
