#include "PassesCommon.h"
#include "PassesInternal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"

namespace tla {
namespace {

static bool isVectorCall(::tla::CallExternOp op)
{
    return op->getParentOfType<::tla::VectorOp>() != nullptr;
}

static void updateCallCoreType(func::FuncOp callee, bool isVector)
{
    MLIRContext* ctx = callee.getContext();
    hivm::TFuncCoreType callerCoreType = isVector ? hivm::TFuncCoreType::AIV : hivm::TFuncCoreType::AIC;
    auto calleeCoreType = callee->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
    if (!calleeCoreType) {
        callee->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, callerCoreType));
        return;
    }
    if (calleeCoreType.getFuncCoreType() != callerCoreType) {
        callee->setAttr(
            hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIC_OR_AIV));
    }
}

// Lower tla.call_extern to a private func.func declaration + func.call.
class TlaLowerExternCallPass : public PassWrapper<TlaLowerExternCallPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerExternCallPass)

    StringRef getArgument() const override
    {
        return "tla-lower-extern-call";
    }
    StringRef getName() const override
    {
        return "TlaLowerExternCallPass";
    }
    StringRef getDescription() const override
    {
        return "Lower tla.call_extern to a private func.call declaration";
    }
    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<func::FuncDialect, ::tla::TlaDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        SmallVector<::tla::CallExternOp, 4> calls;
        module.walk([&](::tla::CallExternOp op) { calls.push_back(op); });

        for (::tla::CallExternOp op : calls) {
            StringRef symbol = op.getCallee();
            bool isVector = isVectorCall(op);
            SmallVector<Type, 4> operandTypes(op.getOperandTypes());
            auto functionType = FunctionType::get(module.getContext(), operandTypes, TypeRange{});
            func::FuncOp callee = module.lookupSymbol<func::FuncOp>(symbol);
            if (!callee) {
                OpBuilder moduleBuilder(module.getBodyRegion());
                moduleBuilder.setInsertionPointToStart(module.getBody());
                callee = moduleBuilder.create<func::FuncOp>(op.getLoc(), symbol, functionType);
                callee.setPrivate();
            } else if (!callee.isDeclaration()) {
                op.emitOpError() << "external symbol @" << symbol << " conflicts with a defined function";
                signalPassFailure();
                return;
            } else if (callee.getFunctionType() != functionType) {
                op.emitOpError() << "external symbol @" << symbol << " was called with incompatible function types";
                signalPassFailure();
                return;
            }
            updateCallCoreType(callee, isVector);

            OpBuilder builder(op);
            builder.create<func::CallOp>(op.getLoc(), callee, op.getOperands());
            op.erase();
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerExternCallPass()
{
    return std::make_unique<TlaLowerExternCallPass>();
}

void registerTlaLowerExternCallPass()
{
    PassRegistration<TlaLowerExternCallPass>();
}

} // namespace tla
