#pragma once

#include <functional>
#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace tla {

void registerTlaLowerDebugPrintPass();
void registerTlaLowerFuncPass();
void registerTlaLowerScalarAccessPass();
void registerTlaSplitMixedFuncPass();
void registerTlaLowerTensorDescPass();
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
mlir::FailureOr<llvm::StringRef>
getPrintTensorHelperSuffix(mlir::Type elementType, std::string &diagnostic);
mlir::LogicalResult lowerTlaMutexToStd(
    mlir::ModuleOp module,
    std::function<mlir::Value(mlir::Operation *, int64_t, unsigned)>
        getOrCreateConstant);

} // namespace tla
