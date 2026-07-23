#include "PassesCommon.h"
#include "PassesInternal.h"

namespace tla {
namespace {

constexpr StringLiteral kDebugPrintWorkspaceAttrName = "tla.debug_print.workspace";

static std::string typeToString(Type type)
{
    std::string text;
    llvm::raw_string_ostream os(text);
    type.print(os);
    return os.str();
}

static BlockArgument getOrAppendDebugPrintWorkspaceArg(func::FuncOp funcOp)
{
    MLIRContext* ctx = funcOp.getContext();
    for (BlockArgument arg : funcOp.getArguments()) {
        if (funcOp.getArgAttr(arg.getArgNumber(), kDebugPrintWorkspaceAttrName))
            return arg;
    }

    FunctionType oldType = funcOp.getFunctionType();
    SmallVector<Type, 8> inputs(oldType.getInputs().begin(), oldType.getInputs().end());
    Type workspaceType = IntegerType::get(ctx, 64);
    inputs.push_back(workspaceType);
    funcOp.setType(FunctionType::get(ctx, inputs, oldType.getResults()));

    Block& entry = funcOp.getBody().front();
    unsigned argIndex = entry.getNumArguments();
    BlockArgument workspaceArg = entry.addArgument(workspaceType, funcOp.getLoc());
    funcOp.setArgAttr(argIndex, kDebugPrintWorkspaceAttrName, UnitAttr::get(ctx));
    funcOp.setArgAttr(
        argIndex, hacc::KernelArgTypeAttr::name, hacc::KernelArgTypeAttr::get(ctx, hacc::KernelArgType::kWorkspace));
    return workspaceArg;
}

static void annotatePrintfRuntimeCall(func::FuncOp funcOp)
{
    MLIRContext* ctx = funcOp.getContext();
    funcOp->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE), UnitAttr::get(ctx));
    funcOp->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIC_OR_AIV));
}

static func::FuncOp getOrCreateRuntimeCall(ModuleOp module, StringRef name, ArrayRef<Type> operandTypes)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
        annotatePrintfRuntimeCall(existing);
        return existing;
    }
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());
    auto func = builder.create<func::FuncOp>(module.getLoc(), name, builder.getFunctionType(operandTypes, {}));
    func.setPrivate();
    annotatePrintfRuntimeCall(func);
    return func;
}

static LogicalResult lowerDebugPrint(::tla::DebugPrintOp op, PatternRewriter& rewriter, ModuleOp module)
{
    if (op->getNumResults() != 0 || op->getNumOperands() != 1)
        return op.emitError("tla.debug_print lowering requires exactly one operand and no results");

    Value value = op.getValue();
    Type valueType = value.getType();
    StringRef calleeName;
    auto intType = dyn_cast<IntegerType>(valueType);
    if (intType && intType.getWidth() == 32 && intType.isSignless()) {
        calleeName = "_mlir_ciface_tla_printf_x_i32";
    } else if (valueType.isF32()) {
        calleeName = "_mlir_ciface_tla_printf_v_f32";
    }
    if (calleeName.empty())
        return op.emitError() << "unsupported tla.debug_print operand type " << typeToString(valueType);

    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp)
        return op.emitError("tla.debug_print must be nested inside func.func");
    BlockArgument workspace = getOrAppendDebugPrintWorkspaceArg(funcOp);

    auto callee = getOrCreateRuntimeCall(module, calleeName, {value.getType(), workspace.getType()});
    rewriter.create<func::CallOp>(op.getLoc(), callee, ValueRange{value, workspace});
    rewriter.eraseOp(op);
    return success();
}

class TlaLowerDebugPrintPass : public PassWrapper<TlaLowerDebugPrintPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerDebugPrintPass)

    StringRef getArgument() const override
    {
        return "tla-lower-debug-print";
    }
    StringRef getName() const override
    {
        return "TlaLowerDebugPrintPass";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<func::FuncDialect, hivm::HIVMDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        SmallVector<::tla::DebugPrintOp, 8> ops;
        module.walk([&](::tla::DebugPrintOp op) { ops.push_back(op); });
        for (::tla::DebugPrintOp op : ops) {
            if (!op || !op->getBlock())
                continue;
            PatternRewriter rewriter(op.getContext());
            rewriter.setInsertionPoint(op);
            if (failed(lowerDebugPrint(op, rewriter, module))) {
                signalPassFailure();
                return;
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerDebugPrintPass()
{
    return std::make_unique<TlaLowerDebugPrintPass>();
}

void registerTlaLowerDebugPrintPass()
{
    PassRegistration<TlaLowerDebugPrintPass>();
}

} // namespace tla
