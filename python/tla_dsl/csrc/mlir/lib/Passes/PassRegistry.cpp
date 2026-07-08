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
  registerTlaLowerFuncPass();
  registerTlaSplitMixedFuncPass();
  registerTlaLowerBlockIdxPass();
  registerConvertTlaToVectorPass();
  registerTlaLowerFlagBarrierToHivmPass();
  registerTlaAllocPtrToHivmPointerCastPass();
  registerTlaLowerMutexToStdPass();
  registerTlaLowerToStdPass();
  registerTlaPrologueEpiloguePass();
  registerTlaLowerAVEToRegbaseIntrinsPass();
}

void buildTlaPipeline(OpPassManager &pm) {
  // Lower Tla function containers to func.func before HIVM lowering so
  // downstream passes and tools inspect func.func. This single pass classifies
  // each device function's AIC/AIV/MIX core type from its tla.cube/tla.vector
  // regions, stamps the HACC/HIVM entry metadata, lowers tla.func to func.func,
  // and tags the module core type -- one structure-derived classification the
  // HACC machinery and the mixed-func split both consume.
  pm.addPass(createTlaLowerFuncPass());
  pm.addPass(createTlaSplitMixedFuncPass());
  pm.addPass(createTlaAllocPtrToHivmPointerCastPass());
  pm.addPass(createConvertTlaToVectorPass());
  pm.addPass(createTlaLowerBlockIdxPass());
  pm.addPass(createTlaLowerFlagBarrierToHivmPass());
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
  pm.addPass(createTlaLowerAVEToRegbaseIntrinsPass());
  pm.addPass(createConvertSCFToCFPass());
}

} // namespace tla
