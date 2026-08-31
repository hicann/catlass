#include "Tools/CompilePipeline.h"

#include "Dialect/Tla/IR/TlaDialect.h"
#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
using namespace mlir;

namespace tla::tools {

void registerTlaCompileDialectsAndTranslations(DialectRegistry& registry)
{
    registry.insert<
        arith::ArithDialect, cf::ControlFlowDialect, DLTIDialect, func::FuncDialect, scf::SCFDialect, LLVM::LLVMDialect,
        memref::MemRefDialect, vector::VectorDialect, ::tla::TlaDialect>();
    registry.insert<
        hacc::HACCDialect, hivm::HIVMDialect, hivmave::AVEDialect, hivm_regbaseintrins::HIVMRegbaseIntrinsDialect>();
    hacc::func_ext::registerHACCDialectExtension(registry);
}

void loadTlaCompileDialects(MLIRContext& context)
{
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<cf::ControlFlowDialect>();
    context.getOrLoadDialect<DLTIDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
    context.getOrLoadDialect<LLVM::LLVMDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<vector::VectorDialect>();
    context.getOrLoadDialect<::tla::TlaDialect>();
    context.getOrLoadDialect<hacc::HACCDialect>();
    context.getOrLoadDialect<hivm::HIVMDialect>();
    context.getOrLoadDialect<hivmave::AVEDialect>();
    context.getOrLoadDialect<hivm_regbaseintrins::HIVMRegbaseIntrinsDialect>();
}

void buildTlaCompilePassManagers(MLIRContext& context, PassManager& tlaPm)
{
    (void)context;
    ::tla::buildTlaPipeline(tlaPm);
}

bool runTlaCompilePipelinesWithManagers(
    ModuleOp module, llvm::StringRef emitMode, PassManager& tlaPm, std::string& output, std::string& error)
{
    if (failed(tlaPm.run(module))) {
        error = "Failed to run Tla pipeline.";
        return false;
    }

    if (emitMode == "mlir") {
        llvm::raw_string_ostream os(output);
        module.print(os);
        os.flush();
        return true;
    }

    error = "Unsupported emit mode; expected 'mlir'.";
    return false;
}

} // namespace tla::tools
