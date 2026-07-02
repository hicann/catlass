#pragma once

#include <functional>
#include <memory>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

namespace tla {

void registerTlaFuncToHaccPass();
void registerTlaSplitMixedFuncPass();
void registerTlaInferFuncCoreTypePass();
void registerTlaLowerToHivmPass();
void registerConvertTlaToVectorPass();
void registerTlaSyncToHivmPass();
void registerTlaAllocPtrToHivmPointerCastPass();
void registerTlaLowerMutexToStdPass();
void registerTlaLowerToStdPass();
void registerTlaPrologueEpiloguePass();

std::unique_ptr<mlir::Pass> createTlaLowerMutexToStdPass();
mlir::LogicalResult lowerTlaMutexToStd(
    mlir::ModuleOp module,
    std::function<mlir::Value(mlir::Operation *, int64_t, unsigned)>
        getOrCreateConstant);

} // namespace tla
