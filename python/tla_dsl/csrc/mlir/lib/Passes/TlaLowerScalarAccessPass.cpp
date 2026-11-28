#include "Passes/TlaTensorToMemref.h"
#include "PassesCommon.h"
#include "PassesInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"

namespace tla {
namespace {

// Build the logical scalar-access view described by tla.tensor_desc. This is
// intentionally independent of the ABI memref's physical layout: the
// descriptor carries the logical offset, shape, and strides for both GM and UB
// tensors.
static FailureOr<Value> materializeScalarView(
    OpBuilder& builder, Operation* diagnosticOp, Value tensor, llvm::DenseMap<Value, Value>& baseMemrefCache)
{
    auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
    if (!descOp)
        return diagnosticOp->emitError("scalar access expects a materialized tla.tensor_desc source"), failure();

    FailureOr<TensorDescriptor> descOr = descriptorFromTensorDescOp(descOp);
    FailureOr<ParsedTensorInfo> rawInfo = parseTensorInfo(tensor.getType());
    if (failed(descOr) || failed(rawInfo))
        return failure();
    TensorDescriptor& desc = *descOr;
    if (desc.addrspace != "gm" && desc.addrspace != "ub")
        return diagnosticOp->emitError("scalar access requires a GM or UB tensor descriptor"), failure();
    if (!isLinearLayout(desc.layoutTag))
        return diagnosticOp->emitError("scalar access supports only linear RowMajor/ColumnMajor descriptors"),
               failure();
    size_t logicalRank = rawInfo->shape.size();
    if (logicalRank != 1 && logicalRank != 2)
        return diagnosticOp->emitError("scalar access descriptor must have logical rank 1 or 2"), failure();
    if (!validateTensorDescriptor(diagnosticOp, desc, "malformed descriptor for scalar access"))
        return failure();

    FailureOr<Value> base =
        getOrMaterializeDescriptorBaseMemref(builder, diagnosticOp->getLoc(), desc, diagnosticOp, baseMemrefCache);
    if (failed(base))
        return failure();
    auto baseType = dyn_cast<MemRefType>((*base).getType());
    if (!baseType)
        return diagnosticOp->emitError("scalar descriptor base did not materialize as memref"), failure();

    Location loc = diagnosticOp->getLoc();
    auto metadata = builder.create<mlir::memref::ExtractStridedMetadataOp>(loc, *base);
    Value rowOffset = builder.createOrFold<arith::MulIOp>(loc, desc.coord[0], desc.stride[0]);
    Value colOffset = builder.createOrFold<arith::MulIOp>(loc, desc.coord[1], desc.stride[1]);
    Value logicalOffset = builder.createOrFold<arith::AddIOp>(loc, rowOffset, colOffset);
    Value storageOffset = builder.createOrFold<arith::AddIOp>(loc, metadata.getOffset(), logicalOffset);

    SmallVector<Value, 2> sizes;
    SmallVector<Value, 2> strides;
    if (logicalRank == 1) {
        sizes.push_back(desc.shape[1]);
        strides.push_back(desc.stride[1]);
    } else {
        sizes.append({desc.shape[0], desc.shape[1]});
        strides.append({desc.stride[0], desc.stride[1]});
    }

    SmallVector<int64_t, 2> dynamicShape(logicalRank, ShapedType::kDynamic);
    SmallVector<int64_t, 2> dynamicStrides(logicalRank, ShapedType::kDynamic);
    auto layout = StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic, dynamicStrides);
    auto viewType = MemRefType::get(dynamicShape, baseType.getElementType(), layout, baseType.getMemorySpace());
    return builder
        .create<mlir::memref::ReinterpretCastOp>(loc, viewType, metadata.getBaseBuffer(), storageOffset, sizes, strides)
        .getResult();
}

struct LowerScalarLoad : OpConversionPattern<::tla::ScalarLoadOp> {
    LowerScalarLoad(TypeConverter& converter, MLIRContext* context, llvm::DenseMap<Value, Value>& baseMemrefCache)
        : OpConversionPattern(converter, context), baseMemrefCache(baseMemrefCache)
    {}

    LogicalResult matchAndRewrite(
        ::tla::ScalarLoadOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto rawInfo = parseTensorInfo(adaptor.getSource().getType());
        if (failed(rawInfo) || adaptor.getIndices().size() != rawInfo->shape.size())
            return op.emitOpError("index count must match logical tensor rank");

        FailureOr<Value> view = materializeScalarView(rewriter, op, adaptor.getSource(), baseMemrefCache);
        if (failed(view))
            return failure();
        Type elementType = cast<MemRefType>((*view).getType()).getElementType();
        if (elementType != getTypeConverter()->convertType(op.getResult().getType()))
            return op.emitOpError("unexpected scalar storage type ") << elementType;

        Value loaded = rewriter.create<mlir::memref::LoadOp>(op.getLoc(), *view, adaptor.getIndices());
        rewriter.replaceOp(op, loaded);
        return success();
    }

private:
    llvm::DenseMap<Value, Value>& baseMemrefCache;
};

struct LowerScalarStore : OpConversionPattern<::tla::ScalarStoreOp> {
    LowerScalarStore(TypeConverter& converter, MLIRContext* context, llvm::DenseMap<Value, Value>& baseMemrefCache)
        : OpConversionPattern(converter, context), baseMemrefCache(baseMemrefCache)
    {}

    LogicalResult matchAndRewrite(
        ::tla::ScalarStoreOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto rawInfo = parseTensorInfo(adaptor.getDest().getType());
        if (failed(rawInfo) || adaptor.getIndices().size() != rawInfo->shape.size())
            return op.emitOpError("index count must match logical tensor rank");

        FailureOr<Value> view = materializeScalarView(rewriter, op, adaptor.getDest(), baseMemrefCache);
        if (failed(view))
            return failure();
        rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, adaptor.getValue(), *view, adaptor.getIndices());
        return success();
    }

private:
    llvm::DenseMap<Value, Value>& baseMemrefCache;
};

struct LowerScalarStorageByte : OpConversionPattern<::tla::ScalarStorageByteOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ::tla::ScalarStorageByteOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, adaptor.getSource());
        return success();
    }
};

struct LowerFullScalarOperand : OpConversionPattern<::tla::FullOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ::tla::FullOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        rewriter.modifyOpInPlace(op, [&] { op->setOperands(adaptor.getOperands()); });
        return success();
    }
};

// Placement: after tla-lower-tensor-desc, before vector/cube outlining. The
// same conversion changes scalar accesses and their control-flow carriers.
class TlaLowerScalarAccessPass : public PassWrapper<TlaLowerScalarAccessPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerScalarAccessPass)

    StringRef getArgument() const override
    {
        return "tla-lower-scalar-access";
    }
    StringRef getName() const override
    {
        return "TlaLowerScalarAccessPass";
    }
    StringRef getDescription() const override
    {
        return "Lower descriptor-based scalar accesses and carry encoded scalars as storage bytes.";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect, hivm::HIVMDialect,
            mlir::memref::MemRefDialect, scf::SCFDialect, ::tla::TlaDialect>();
    }

    void runOnOperation() override
    {
        MLIRContext* context = &getContext();
        TypeConverter converter;
        converter.addConversion([context](Type type) { return storageElementType(context, type); });

        llvm::DenseMap<Value, Value> baseMemrefCache;
        RewritePatternSet patterns(context);
        patterns.add<LowerScalarLoad, LowerScalarStore>(converter, context, baseMemrefCache);
        patterns.add<LowerScalarStorageByte, LowerFullScalarOperand>(converter, context);
        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, converter);
        populateCallOpTypeConversionPattern(patterns, converter);
        populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
        populateReturnOpTypeConversionPattern(patterns, converter);

        ConversionTarget target(*context);
        target.addIllegalOp<::tla::ScalarLoadOp, ::tla::ScalarStoreOp, ::tla::ScalarStorageByteOp>();
        scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns, target);
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return converter.isSignatureLegal(op.getFunctionType()) && converter.isLegal(&op.getBody());
        });
        target.addDynamicallyLegalOp<func::CallOp>(
            [&](func::CallOp op) { return converter.isSignatureLegal(op.getCalleeType()); });
        target.markUnknownOpDynamicallyLegal([&](Operation* op) { return converter.isLegal(op); });

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerScalarAccessPass()
{
    return std::make_unique<TlaLowerScalarAccessPass>();
}

void registerTlaLowerScalarAccessPass()
{
    PassRegistration<TlaLowerScalarAccessPass>();
}

} // namespace tla
