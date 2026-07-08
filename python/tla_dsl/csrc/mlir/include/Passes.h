#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace tla {

std::unique_ptr<mlir::Pass> createTlaAllocPtrToHivmPointerCastPass();
std::unique_ptr<mlir::Pass> createTlaLowerToStdPass();
std::unique_ptr<mlir::Pass> createTlaLowerFlagBarrierToHivmPass();
std::unique_ptr<mlir::Pass> createTlaLowerFuncPass();
std::unique_ptr<mlir::Pass> createTlaSplitMixedFuncPass();
std::unique_ptr<mlir::Pass> createTlaLowerBlockIdxPass();
std::unique_ptr<mlir::Pass> createConvertTlaToVectorPass();
std::unique_ptr<mlir::Pass> createTlaPrologueEpiloguePass();
std::unique_ptr<mlir::Pass> createTlaLowerAVEToRegbaseIntrinsPass();

void registerTlaPasses();
void buildTlaPipeline(mlir::OpPassManager &pm);

} // namespace tla
