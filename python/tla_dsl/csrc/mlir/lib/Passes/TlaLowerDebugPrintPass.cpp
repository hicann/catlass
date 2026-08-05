#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaTensorToMemref.h"
#include "llvm/ADT/DenseMap.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <cstdint>

namespace tla {

constexpr StringLiteral kDebugPrintWorkspaceAttrName = "tla.debug_print.workspace";
constexpr StringLiteral kDebugPrintFormatCallAttrName = "tla.debug_print.format";
constexpr StringLiteral kPrintTensorWorkspaceAttrName = "tla.print_tensor.workspace";
constexpr StringLiteral kPrintTensorSupportedDtypes =
    "f16, f32, i8, i16, i32, u8, u16, u32";
constexpr StringLiteral kStringFormatCalleeName =
    "_mlir_ciface_tla_printf_format_string";
constexpr StringLiteral kValuesFormatCalleeName =
    "_mlir_ciface_tla_printf_format_values";
constexpr unsigned kFormattedScalarValueSlots = 8;

static std::string typeToString(Type type)
{
    std::string text;
    llvm::raw_string_ostream os(text);
    type.print(os);
    return os.str();
}

static std::string printTensorDiagnosticTypeToken(Type type)
{
    auto integerType = dyn_cast<IntegerType>(type);
    if (integerType && integerType.isUnsigned())
        return "u" + std::to_string(integerType.getWidth());
    return typeToString(type);
}

static FailureOr<StringRef> getDebugPrintScalarTypeToken(Type type)
{
    auto intType = dyn_cast<IntegerType>(type);
    if (intType) {
        unsigned width = intType.getWidth();
        if (width == 8 || width == 16 || width == 32) {
            if (intType.isSignless())
                return width == 8 ? StringRef("i8")
                                  : width == 16 ? StringRef("i16")
                                                : StringRef("i32");
            if (intType.isUnsigned())
                return width == 8 ? StringRef("u8")
                                  : width == 16 ? StringRef("u16")
                                                : StringRef("u32");
        }
    }
    if (type.isF16())
        return StringRef("f16");
    if (type.isF32())
        return StringRef("f32");
    return failure();
}

// Format text becomes an LLVM global, so derive a deterministic, symbol-safe
// suffix from the canonical backend printf payload instead of using user text.
static uint64_t hashDebugPrintCanonicalPayload(StringRef payload)
{
    uint64_t hash = 14695981039346656037ULL;
    for (char c : payload) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

static std::string toLowerHex64(uint64_t value)
{
    constexpr char kHex[] = "0123456789abcdef";
    std::string result(16, '0');
    for (int i = 15; i >= 0; --i) {
        result[i] = kHex[value & 0xf];
        value >>= 4;
    }
    return result;
}

static FailureOr<std::string>
getFormattedDebugPrintCanonicalPayload(::tla::DebugPrintOp op)
{
    auto formatAttr = op->getAttrOfType<StringAttr>("format");
    if (!formatAttr)
        return failure();

    StringRef format = formatAttr.getValue();
    std::string payload;
    payload += "tla.debug_print.format\n";
    payload += "format:";
    payload += std::to_string(format.size());
    payload += ":";
    if (!format.empty())
        payload.append(format.data(), format.size());
    payload += "\ntypes:";
    for (auto item : llvm::enumerate(op.getValues())) {
        FailureOr<StringRef> typeToken =
            getDebugPrintScalarTypeToken(item.value().getType());
        if (failed(typeToken)) {
            op.emitError() << "unsupported tla.debug_print operand type "
                           << typeToString(item.value().getType());
            return failure();
        }
        if (item.index() != 0)
            payload += ",";
        payload.append(typeToken->data(), typeToken->size());
    }
    return payload;
}

static FailureOr<std::string>
getFormattedDebugPrintHelperSuffix(::tla::DebugPrintOp op)
{
    FailureOr<std::string> payload = getFormattedDebugPrintCanonicalPayload(op);
    if (failed(payload))
        return failure();
    return toLowerHex64(hashDebugPrintCanonicalPayload(*payload));
}

static FailureOr<std::string>
getGeneratedDebugPrintFormat(::tla::DebugPrintOp op)
{
    auto formatAttr = op->getAttrOfType<StringAttr>("format");
    if (!formatAttr)
        return failure();

    StringRef format = formatAttr.getValue();
    std::string generated;
    unsigned fieldIndex = 0;
    for (size_t i = 0; i < format.size();) {
        char c = format[i];
        if (c == '{') {
            if (i + 1 >= format.size()) {
                op.emitError("malformed format string");
                return failure();
            }
            if (format[i + 1] == '{') {
                generated.push_back('{');
                i += 2;
                continue;
            }
            if (format[i + 1] == '}') {
                if (fieldIndex >= op.getValues().size()) {
                    op.emitError("format argument count mismatch");
                    return failure();
                }
                FailureOr<StringRef> typeToken =
                    getDebugPrintScalarTypeToken(
                        op.getValues()[fieldIndex].getType());
                if (failed(typeToken)) {
                    op.emitError()
                        << "unsupported tla.debug_print operand type "
                        << typeToString(
                               op.getValues()[fieldIndex].getType());
                    return failure();
                }
                if (typeToken->starts_with("i"))
                    generated += "%d";
                else if (typeToken->starts_with("u"))
                    generated += "%u";
                else if (typeToken->starts_with("f"))
                    generated += "%f";
                else
                    return failure();
                ++fieldIndex;
                i += 2;
                continue;
            }
            op.emitError("unsupported format field");
            return failure();
        }
        if (c == '}') {
            if (i + 1 < format.size() && format[i + 1] == '}') {
                generated.push_back('}');
                i += 2;
                continue;
            }
            op.emitError("malformed format string");
            return failure();
        }
        if (c == '%')
            generated += "%%";
        else
            generated.push_back(c);
        ++i;
    }
    if (fieldIndex != op.getValues().size()) {
        op.emitError("format argument count mismatch");
        return failure();
    }
    return generated;
}

FailureOr<StringRef>
getPrintTensorHelperSuffix(Type elementType, std::string &diagnostic)
{
    StringRef suffix = llvm::StringSwitch<StringRef>(typeToString(elementType))
                           .Case("f16", "f16")
                           .Case("f32", "f32")
                           .Case("i8", "i8")
                           .Case("i16", "i16")
                           .Case("i32", "i32")
                           .Case("ui8", "u8")
                           .Case("ui16", "u16")
                           .Case("ui32", "u32")
                           .Default("");
    if (!suffix.empty())
        return suffix;
    diagnostic = "unsupported dtype " + printTensorDiagnosticTypeToken(elementType) +
                 "; supported dtypes: " + kPrintTensorSupportedDtypes.str();
    return failure();
}

FailureOr<DebugPrintHelperSpec>
getDebugPrintHelperSpec(Type type, DebugPrintPlacement placement,
                        std::string &diagnostic)
{
    if (placement == DebugPrintPlacement::Invalid) {
        diagnostic =
            "unsupported tla.debug_print placement invalid for dtype " +
            printTensorDiagnosticTypeToken(type) + "; expected cube or vector";
        return failure();
    }

    FailureOr<StringRef> suffix = getPrintTensorHelperSuffix(type, diagnostic);
    if (failed(suffix))
        return failure();
    StringRef callee = llvm::StringSwitch<StringRef>(*suffix)
                           .Case("i8", "_mlir_ciface_tla_printf_x_i8")
                           .Case("i16", "_mlir_ciface_tla_printf_x_i16")
                           .Case("i32", "_mlir_ciface_tla_printf_x_i32")
                           .Case("u8", "_mlir_ciface_tla_printf_x_u8")
                           .Case("u16", "_mlir_ciface_tla_printf_x_u16")
                           .Case("u32", "_mlir_ciface_tla_printf_x_u32")
                           .Case("f16", "_mlir_ciface_tla_printf_v_f16")
                           .Case("f32", "_mlir_ciface_tla_printf_v_f32")
                           .Default("");
    if (callee.empty()) {
        diagnostic = "missing tla.debug_print helper for dtype " +
                     printTensorDiagnosticTypeToken(type);
        return failure();
    }
    return DebugPrintHelperSpec{callee};
}

namespace {

static constexpr bool isPrintTensorCallIdRepresentable(uint32_t ordinal)
{
    return ordinal <= 0xffff;
}

static_assert(isPrintTensorCallIdRepresentable(0xffff));
static_assert(!isPrintTensorCallIdRepresentable(0x10000));

static FailureOr<uint16_t>
getPrintTensorCallId(uint32_t ordinal, std::string &diagnostic)
{
    if (isPrintTensorCallIdRepresentable(ordinal))
        return static_cast<uint16_t>(ordinal);
    diagnostic = "call ID exceeds the 16-bit descriptor range";
    return failure();
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

static BlockArgument getOrPrependPrintTensorWorkspaceArg(func::FuncOp funcOp)
{
    MLIRContext* ctx = funcOp.getContext();
    for (BlockArgument arg : funcOp.getArguments()) {
        if (funcOp.getArgAttr(arg.getArgNumber(), kPrintTensorWorkspaceAttrName))
            return arg;
    }

    Type workspaceType = IntegerType::get(ctx, 64);
    funcOp.insertArgument(
        0, workspaceType, DictionaryAttr::get(ctx), funcOp.getLoc());
    BlockArgument workspaceArg = funcOp.getArgument(0);
    funcOp.setArgAttr(0, kPrintTensorWorkspaceAttrName, UnitAttr::get(ctx));
    funcOp.setArgAttr(
        0, hacc::KernelArgTypeAttr::name,
        hacc::KernelArgTypeAttr::get(ctx, hacc::KernelArgType::kWorkspace));
    return workspaceArg;
}

static BlockArgument getOrAppendPrintTensorWorkspaceArg(func::FuncOp funcOp)
{
    MLIRContext* ctx = funcOp.getContext();
    for (BlockArgument arg : funcOp.getArguments()) {
        if (funcOp.getArgAttr(arg.getArgNumber(), kPrintTensorWorkspaceAttrName))
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
    funcOp.setArgAttr(argIndex, kPrintTensorWorkspaceAttrName, UnitAttr::get(ctx));
    funcOp.setArgAttr(
        argIndex, hacc::KernelArgTypeAttr::name,
        hacc::KernelArgTypeAttr::get(ctx, hacc::KernelArgType::kWorkspace));
    return workspaceArg;
}

static bool isMixedSplitFunction(func::FuncOp funcOp)
{
    return funcOp->hasAttr(hivm::TPartOfMixAttr::name);
}

static BlockArgument getOrCreatePrintTensorWorkspaceArg(func::FuncOp funcOp,
                                                        ModuleOp module)
{
    if (!isMixedSplitFunction(funcOp)) {
        return getOrPrependPrintTensorWorkspaceArg(funcOp);
    }

    StringRef name = funcOp.getSymName();
    constexpr StringLiteral aicSuffix = "_mix_aic";
    constexpr StringLiteral aivSuffix = "_mix_aiv";
    StringRef suffix = name.ends_with(aicSuffix) ? aicSuffix : aivSuffix;
    StringRef peerSuffix = suffix == aicSuffix ? aivSuffix : aicSuffix;
    std::string peerName =
        (name.drop_back(suffix.size()) + peerSuffix).str();
    if (auto peer = module.lookupSymbol<func::FuncOp>(peerName))
        (void)getOrAppendPrintTensorWorkspaceArg(peer);
    return getOrAppendPrintTensorWorkspaceArg(funcOp);
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

static LLVM::GlobalOp getOrCreateStringFormatGlobal(ModuleOp module,
                                                    OpBuilder& builder,
                                                    Location loc,
                                                    StringRef suffix,
                                                    StringRef format)
{
    std::string name = "tla_debug_print_format_";
    name.append(suffix.data(), suffix.size());
    if (auto existing = module.lookupSymbol<LLVM::GlobalOp>(name))
        return existing;

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    std::string value(format.data(), format.size());
    value.push_back('\0');
    auto globalType = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    return builder.create<LLVM::GlobalOp>(
        loc, globalType, /*isConstant=*/true, LLVM::Linkage::Internal, name,
        builder.getStringAttr(StringRef(value.data(), value.size())),
        /*alignment=*/0);
}

static Value createStringFormatPointer(OpBuilder& builder, Location loc,
                                       LLVM::GlobalOp global)
{
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
    Value globalPtr =
        builder.create<LLVM::AddressOfOp>(loc, ptrType, global.getSymNameAttr());
    SmallVector<LLVM::GEPArg> indices{0, 0};
    return builder.create<LLVM::GEPOp>(
        loc, ptrType, global.getType(), globalPtr, indices);
}

static SmallVector<Type, 12>
getFormattedValuesHelperTypes(MLIRContext* ctx)
{
    Type i64Type = IntegerType::get(ctx, 64);
    SmallVector<Type, 12> types{
        LLVM::LLVMPointerType::get(ctx), i64Type, i64Type};
    for (unsigned i = 0; i < kFormattedScalarValueSlots; ++i)
        types.push_back(i64Type);
    types.push_back(i64Type);
    return types;
}

static FailureOr<Value> createFormattedScalarSlot(OpBuilder& builder,
                                                 Location loc,
                                                 Value value)
{
    Type i64Type = builder.getI64Type();
    Type valueType = value.getType();
    auto intType = dyn_cast<IntegerType>(valueType);
    if (intType && (intType.getWidth() == 8 || intType.getWidth() == 16 ||
                    intType.getWidth() == 32)) {
        if (intType.isSignless())
            return builder.create<arith::ExtSIOp>(loc, i64Type, value)
                .getResult();
        if (intType.isUnsigned()) {
            Type signlessType = builder.getIntegerType(intType.getWidth());
            Value signless =
                builder
                    .create<UnrealizedConversionCastOp>(loc, signlessType,
                                                        value)
                    .getResult(0);
            return builder.create<arith::ExtUIOp>(loc, i64Type, signless)
                .getResult();
        }
    }
    if (valueType.isF16())
        value = builder.create<arith::ExtFOp>(loc, builder.getF32Type(), value);
    if (value.getType().isF32()) {
        Value bits = builder.create<arith::BitcastOp>(
            loc, builder.getI32Type(), value);
        return builder.create<arith::ExtUIOp>(loc, i64Type, bits).getResult();
    }
    return failure();
}

static FailureOr<SmallVector<Value, 12>> buildFormattedValuesCallOperands(
    OpBuilder& builder, Location loc, StringRef generatedFormat,
    Value formatPtr, ValueRange values, Value workspace)
{
    SmallVector<Value, 12> callOperands;
    callOperands.push_back(formatPtr);
    callOperands.push_back(builder.create<arith::ConstantIntOp>(
        loc, static_cast<int64_t>(generatedFormat.size()), 64));
    callOperands.push_back(builder.create<arith::ConstantIntOp>(
        loc, static_cast<int64_t>(values.size()), 64));

    Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    SmallVector<Value, 8> slots(kFormattedScalarValueSlots, zero);
    for (auto item : llvm::enumerate(values)) {
        FailureOr<Value> slot =
            createFormattedScalarSlot(builder, loc, item.value());
        if (failed(slot))
            return failure();
        slots[item.index()] = *slot;
    }
    callOperands.append(slots.begin(), slots.end());
    callOperands.push_back(workspace);
    return callOperands;
}

static LogicalResult lowerLegacyDebugPrint(::tla::DebugPrintOp op, PatternRewriter& rewriter, ModuleOp module)
{
    if (op->getNumResults() != 0 || op->getNumOperands() != 1)
        return op.emitError("tla.debug_print lowering requires exactly one operand and no results");

    Value value = op.getValues().front();
    Type valueType = value.getType();
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp)
        return op.emitError("tla.debug_print must be nested inside func.func");
    DebugPrintPlacement placement = DebugPrintPlacement::Invalid;
    if (std::optional<hivm::TFuncCoreType> coreType =
            getFunctionCoreType(funcOp)) {
        if (*coreType == hivm::TFuncCoreType::AIC)
            placement = DebugPrintPlacement::Cube;
        else if (*coreType == hivm::TFuncCoreType::AIV)
            placement = DebugPrintPlacement::Vector;
    }
    std::string diagnostic;
    FailureOr<DebugPrintHelperSpec> helper =
        getDebugPrintHelperSpec(valueType, placement, diagnostic);
    if (failed(helper))
        return op.emitError() << diagnostic;

    BlockArgument workspace = getOrAppendDebugPrintWorkspaceArg(funcOp);
    auto callee = getOrCreateRuntimeCall(
        module, helper->callee, {value.getType(), workspace.getType()});
    rewriter.create<func::CallOp>(op.getLoc(), callee, ValueRange{value, workspace});
    rewriter.eraseOp(op);
    return success();
}

static LogicalResult lowerFormattedDebugPrint(::tla::DebugPrintOp op,
                                              PatternRewriter& rewriter,
                                              ModuleOp module)
{
    if (op->getNumResults() != 0)
        return op.emitError("formatted tla.debug_print lowering requires no results");

    auto formatAttr = op->getAttrOfType<StringAttr>("format");
    if (!formatAttr)
        return op.emitError("formatted tla.debug_print lowering requires a format attribute");

    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp)
        return op.emitError("tla.debug_print must be nested inside func.func");

    FailureOr<std::string> helperSuffix = getFormattedDebugPrintHelperSuffix(op);
    if (failed(helperSuffix))
        return failure();
    FailureOr<std::string> generatedFormat = getGeneratedDebugPrintFormat(op);
    if (failed(generatedFormat))
        return failure();

    Type i64Type = rewriter.getI64Type();
    BlockArgument workspace = getOrAppendDebugPrintWorkspaceArg(funcOp);

    if (op.getValues().empty()) {
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        LLVM::GlobalOp global = getOrCreateStringFormatGlobal(
            module, rewriter, op.getLoc(), *helperSuffix, *generatedFormat);
        Value formatPtr = createStringFormatPointer(rewriter, op.getLoc(), global);
        auto callee = getOrCreateRuntimeCall(
            module, kStringFormatCalleeName, {ptrType, i64Type, workspace.getType()});
        SmallVector<Value, 3> callOperands;
        callOperands.push_back(formatPtr);
        callOperands.push_back(
            rewriter.create<arith::ConstantIntOp>(
                op.getLoc(), static_cast<int64_t>(generatedFormat->size()), 64));
        callOperands.push_back(workspace);
        auto call = rewriter.create<func::CallOp>(op.getLoc(), callee, callOperands);
        call->setAttr(kDebugPrintFormatCallAttrName, formatAttr);
        rewriter.eraseOp(op);
        return success();
    }

    LLVM::GlobalOp global = getOrCreateStringFormatGlobal(
        module, rewriter, op.getLoc(), *helperSuffix, *generatedFormat);
    Value formatPtr = createStringFormatPointer(rewriter, op.getLoc(), global);
    auto callee = getOrCreateRuntimeCall(
        module, kValuesFormatCalleeName,
        getFormattedValuesHelperTypes(rewriter.getContext()));
    FailureOr<SmallVector<Value, 12>> callOperands =
        buildFormattedValuesCallOperands(rewriter, op.getLoc(), *generatedFormat,
                                         formatPtr, op.getValues(), workspace);
    if (failed(callOperands))
        return op.emitError("unsupported tla.debug_print operand type");

    auto call = rewriter.create<func::CallOp>(op.getLoc(), callee, *callOperands);
    call->setAttr(kDebugPrintFormatCallAttrName, formatAttr);
    rewriter.eraseOp(op);
    return success();
}

static LogicalResult lowerDebugPrint(::tla::DebugPrintOp op,
                                     PatternRewriter& rewriter,
                                     ModuleOp module)
{
    if (op->getAttrOfType<StringAttr>("format"))
        return lowerFormattedDebugPrint(op, rewriter, module);
    return lowerLegacyDebugPrint(op, rewriter, module);
}

static LogicalResult lowerPrintTensor(::tla::PrintTensorOp op,
                                     PatternRewriter& rewriter,
                                     ModuleOp module,
                                     uint32_t callId)
{
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp)
        return op.emitError("tla.print_tensor must be nested inside func.func");

    auto descOp = op.getValue().getDefiningOp<::tla::TensorDescOp>();
    if (!descOp)
        return op.emitError("could not resolve the materialized tensor descriptor");
    FailureOr<::tla::TensorDescriptor> descOr =
        ::tla::descriptorFromTensorDescOp(descOp);
    if (failed(descOr))
        return op.emitError("could not decode the materialized tensor descriptor");
    const ::tla::TensorDescriptor& desc = *descOr;
    llvm::DenseMap<Value, Value> baseMemrefCache;
    FailureOr<Value> materialized =
        ::tla::getOrMaterializeDescriptorBaseMemref(
            rewriter, op.getLoc(), desc, op, baseMemrefCache);
    if (failed(materialized))
        return op.emitError("could not materialize the tensor base");
    Value tensorPtr = rewriter.create<::mlir::memref::ExtractAlignedPointerAsIndexOp>(
        op.getLoc(), *materialized);
    auto tensorType = op.getValue().getType();
    Type elementType = tensorType.getPtr().getPointee();
    Value elementOffset;
    if (::tla::isLinearLayout(desc.layoutTag)) {
        Value rowElements = rewriter.createOrFold<arith::MulIOp>(
            op.getLoc(), desc.rowOffset, desc.stride0);
        Value colElements = rewriter.createOrFold<arith::MulIOp>(
            op.getLoc(), desc.colOffset, desc.stride1);
        elementOffset = rewriter.createOrFold<arith::AddIOp>(
            op.getLoc(), rowElements, colElements);
    } else {
        if (!::tla::isPackedLayout(desc.layoutTag) ||
            desc.packedShape.size() != 4 || desc.packedStride.size() != 4)
            return op.emitError("could not materialize the tensor physical offset");

        Value rowDivisor = desc.packedShape[0];
        if (desc.layoutTag == ::tla::TensorLayoutTag::zNUnAlign)
            rowDivisor = rewriter.createOrFold<arith::DivSIOp>(
                op.getLoc(), desc.packedStride[3], desc.packedShape[2]);
        Value physical0 = rewriter.createOrFold<arith::RemSIOp>(
            op.getLoc(), desc.rowOffset, rowDivisor);
        Value physical1 = rewriter.createOrFold<arith::DivSIOp>(
            op.getLoc(), desc.rowOffset, rowDivisor);
        Value physical2 = rewriter.createOrFold<arith::RemSIOp>(
            op.getLoc(), desc.colOffset, desc.packedShape[2]);
        Value physical3 = rewriter.createOrFold<arith::DivSIOp>(
            op.getLoc(), desc.colOffset, desc.packedShape[2]);
        Value term0 = rewriter.createOrFold<arith::MulIOp>(
            op.getLoc(), physical0, desc.packedStride[0]);
        Value term1 = rewriter.createOrFold<arith::MulIOp>(
            op.getLoc(), physical1, desc.packedStride[1]);
        Value term2 = rewriter.createOrFold<arith::MulIOp>(
            op.getLoc(), physical2, desc.packedStride[2]);
        Value term3 = rewriter.createOrFold<arith::MulIOp>(
            op.getLoc(), physical3, desc.packedStride[3]);
        Value rowElements = rewriter.createOrFold<arith::AddIOp>(
            op.getLoc(), term0, term1);
        Value colElements = rewriter.createOrFold<arith::AddIOp>(
            op.getLoc(), term2, term3);
        elementOffset = rewriter.createOrFold<arith::AddIOp>(
            op.getLoc(), rowElements, colElements);
    }
    Value elementBytes = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), elementType.getIntOrFloatBitWidth() / 8);
    Value byteOffset = rewriter.create<arith::MulIOp>(
        op.getLoc(), elementOffset, elementBytes);
    tensorPtr = rewriter.create<arith::AddIOp>(
        op.getLoc(), tensorPtr, byteOffset);
    Value tensorI64 = rewriter.create<arith::IndexCastOp>(
        op.getLoc(), rewriter.getI64Type(), tensorPtr);
    Value count = op.getLength();
    Value shape0;
    Value shape1;
    if (op.getShape().size() == 1) {
        shape0 = ::tla::castValueToI64(rewriter, op.getLoc(), desc.shape1);
        shape1 = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
    } else {
        shape0 = ::tla::castValueToI64(rewriter, op.getLoc(), desc.shape0);
        shape1 = ::tla::castValueToI64(rewriter, op.getLoc(), desc.shape1);
    }
    Value shift = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 32, 64);
    Value packedShape1 = rewriter.create<arith::ShLIOp>(
        op.getLoc(), shape1, shift);
    Value packedShape = rewriter.create<arith::OrIOp>(
        op.getLoc(), shape0, packedShape1);
    Value encodedCallId = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), callId, 64);
    std::string diagnostic;
    FailureOr<StringRef> helperSuffix =
        getPrintTensorHelperSuffix(elementType, diagnostic);
    if (failed(helperSuffix))
        return op.emitError(diagnostic);
    std::string calleeName =
        tensorType.getPtr().getAddrspace() == AddressSpace::ub
            ? "_mlir_ciface_tla_print_tensor_ub_"
            : "_mlir_ciface_tla_print_tensor_gm_";
    calleeName.append(helperSuffix->data(), helperSuffix->size());
    BlockArgument workspace =
        getOrCreatePrintTensorWorkspaceArg(funcOp, module);
    auto callee = getOrCreateRuntimeCall(
        module, calleeName,
        {workspace.getType(), tensorI64.getType(), count.getType(),
         packedShape.getType(), encodedCallId.getType()});
    rewriter.create<func::CallOp>(
        op.getLoc(), callee,
        ValueRange{workspace, tensorI64, count, packedShape, encodedCallId});
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
        registry.insert<arith::ArithDialect, func::FuncDialect, hivm::HIVMDialect,
                        LLVM::LLVMDialect>();
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
        SmallVector<::tla::PrintTensorOp, 8> printOps;
        module.walk([&](::tla::PrintTensorOp op) { printOps.push_back(op); });
        llvm::DenseMap<Operation *, uint32_t> nextCallId;
        for (::tla::PrintTensorOp op : printOps) {
            if (!op || !op->getBlock())
                continue;
            auto funcOp = op->getParentOfType<func::FuncOp>();
            if (!funcOp || funcOp.isPrivate()) {
                op.emitError("tla.print_tensor requires a non-private kernel entrypoint");
                signalPassFailure();
                return;
            }
            if (!funcOp.getBody().hasOneBlock()) {
                op.emitError("tla.print_tensor requires a single-block kernel CFG");
                signalPassFailure();
                return;
            }
            if (op->getBlock() != &op->getParentRegion()->front()) {
                op.emitError("tla.print_tensor must be in an entry CFG block");
                signalPassFailure();
                return;
            }
            for (Operation *ancestor = op->getParentOp();
                 ancestor && ancestor != funcOp.getOperation();
                 ancestor = ancestor->getParentOp()) {
                StringRef name = ancestor->getName().getStringRef();
                if (name == "scf.if" || name == "scf.for" ||
                    name == "scf.while") {
                    op.emitError("tla.print_tensor cannot be nested in dynamic SCF");
                    signalPassFailure();
                    return;
                }
                if (ancestor->getNumRegions() != 0) {
                    op.emitError("tla.print_tensor has an unrecognized multi-execution ancestor");
                    signalPassFailure();
                    return;
                }
            }
            uint32_t ordinal = nextCallId[funcOp.getOperation()]++;
            std::string callIdDiagnostic;
            FailureOr<uint16_t> callId =
                getPrintTensorCallId(ordinal, callIdDiagnostic);
            if (failed(callId)) {
                op.emitError(callIdDiagnostic);
                signalPassFailure();
                return;
            }
            PatternRewriter rewriter(op.getContext());
            rewriter.setInsertionPoint(op);
            if (failed(lowerPrintTensor(op, rewriter, module, *callId))) {
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
