#pragma once

#include <functional>
#include <memory>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

namespace tla {

void registerTlaLowerFuncPass();
void registerTlaSplitMixedFuncPass();
void registerTlaLowerBlockIdxPass();
void registerTlaVectorRegionPass();
void registerTlaLowerFlagBarrierToHivmPass();
void registerTlaLowerPtrPass();
void registerTlaLowerMutexToStdPass();
void registerTlaCubeRegionPass();
void registerTlaFinalizeMemrefPass();
void registerTlaPrologueEpiloguePass();
void registerTlaLowerAVEToRegbaseIntrinsPass();

std::unique_ptr<mlir::Pass> createTlaLowerMutexToStdPass();
std::unique_ptr<mlir::Pass> createTlaLowerAVEToRegbaseIntrinsPass();
mlir::LogicalResult lowerTlaMutexToStd(
    mlir::ModuleOp module,
    std::function<mlir::Value(mlir::Operation *, int64_t, unsigned)>
        getOrCreateConstant);

} // namespace tla
