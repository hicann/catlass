#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace tla {

std::unique_ptr<mlir::Pass> createTlaAllocPtrToHivmPointerCastPass();
std::unique_ptr<mlir::Pass> createTlaLowerToStdPass();
std::unique_ptr<mlir::Pass> createTlaSyncToHivmPass();
std::unique_ptr<mlir::Pass> createTlaFuncToHaccPass();
std::unique_ptr<mlir::Pass> createTlaLowerToHivmPass();
std::unique_ptr<mlir::Pass> createSupportTritonPass();
std::unique_ptr<mlir::Pass> createConvertTlaToVectorPass();
std::unique_ptr<mlir::Pass> createAddKernelPrologueEpiloguePass();

void registerTlaPasses();
void buildTlaPipeline(mlir::OpPassManager &pm);

} // namespace tla
