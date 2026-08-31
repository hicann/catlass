#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"

namespace tla::tools {

void registerTlaCompileDialectsAndTranslations(mlir::DialectRegistry& registry);

void loadTlaCompileDialects(mlir::MLIRContext& context);

void buildTlaCompilePassManagers(mlir::MLIRContext& context, mlir::PassManager& tlaPm);

bool runTlaCompilePipelinesWithManagers(
    mlir::ModuleOp module, llvm::StringRef emitMode, mlir::PassManager& tlaPm, std::string& output, std::string& error);

} // namespace tla::tools
