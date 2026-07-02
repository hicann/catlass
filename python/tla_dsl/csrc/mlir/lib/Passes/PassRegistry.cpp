#include "PassesCommon.h"
#include "PassesInternal.h"

#include "bishengir/Conversion/ArithToHIVMAVE/ArithToHIVMAVE.h"
#include "bishengir/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.h"
#include "bishengir/Conversion/HIVMAVEToStandard/HIVMAVEToStandard.h"
#include "bishengir/Conversion/HIVMToStandard/HIVMToStandard.h"
#include "bishengir/Conversion/VectorToHIVMAVE/VectorToHIVMAVE.h"

#include "mlir/Dialect/MemRef/Transforms/Passes.h"

namespace tla {

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
  registerTlaPrologueEpiloguePass();
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
  pm.addPass(createTlaPrologueEpiloguePass());
  pm.addPass(createCSEPass());
  pm.addPass(mlir::createVectorToHIVMAVEConversionPass());
  pm.nest<func::FuncOp>().addPass(mlir::createArithToHIVMAVEConversionPass());
  mlir::ConvertHIVMToStandardOptions hivmToStdOptions;
  pm.addPass(mlir::createConvertHIVMToStandardPass(hivmToStdOptions));
  pm.addPass(mlir::createConvertHIVMAVEToStandardPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createConvertHIVMAVEToAVEIntrinPass());
  pm.addPass(createConvertSCFToCFPass());
}

} // namespace tla
