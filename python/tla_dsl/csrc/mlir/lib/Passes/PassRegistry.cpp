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
  registerTlaRejectDebugPrintPass();
  registerTlaLowerFuncPass();
  registerTlaLowerScalarAccessPass();
  registerTlaSplitMixedFuncPass();
  registerTlaLowerTensorDescPass();
  registerTlaLowerBlockIdxPass();
  registerTlaVectorRegionPass();
  registerTlaLowerFlagBarrierToHivmPass();
  registerTlaLowerPtrPass();
  registerTlaLowerMutexToStdPass();
  registerTlaCubeRegionPass();
  registerTlaFinalizeMemrefPass();
  registerTlaPrologueEpiloguePass();
  registerTlaLowerAVEToRegbaseIntrinsPass();
}

void buildTlaPipeline(OpPassManager &pm) {
  pm.addPass(createTlaRejectDebugPrintPass());
  // Lower Tla function containers to func.func before HIVM lowering so
  // downstream passes and tools inspect func.func. This single pass classifies
  // each device function's AIC/AIV/MIX core type from its tla.cube/tla.vector
  // regions, stamps the HACC/HIVM entry metadata, lowers tla.func to func.func,
  // and tags the module core type -- one structure-derived classification the
  // HACC machinery and the mixed-func split both consume.
  pm.addPass(createTlaLowerFuncPass());
  pm.addPass(createTlaLowerPtrPass());
  pm.addPass(createTlaSplitMixedFuncPass());
  // All ScalarSSA (bridged memref + descriptor !tla.tensor) -> memref.load/store
  // before region outlining. GM-kernel-arg only; tile producers are left intact.
  pm.addPass(createTlaLowerScalarAccessPass());
  // Materialize tensor-view producer chains as tla.tensor_desc before region
  // passes consume them. Runs after tla-lower-ptr so descriptor bases use the
  // bridged memref or !tla.ptr produced by the standard pipeline.
  pm.addPass(createTlaLowerTensorDescPass());
  pm.addPass(createTlaVectorRegionPass());
  pm.addPass(createTlaCubeRegionPass());
  pm.addPass(createTlaFinalizeMemrefPass());
  pm.addPass(createTlaLowerBlockIdxPass());
  pm.addPass(createTlaLowerFlagBarrierToHivmPass());
  pm.addPass(createTlaLowerMutexToStdPass());
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
