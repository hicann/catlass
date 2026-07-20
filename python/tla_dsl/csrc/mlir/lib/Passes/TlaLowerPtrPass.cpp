#include "Passes/TlaTensorToMemref.h"
#include "PassesCommon.h"
#include "PassesInternal.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"

namespace tla {
namespace {

static FailureOr<uint64_t> alignUpCheckedU64(uint64_t value,
                                             uint64_t alignment) {
  if (alignment == 0)
    return failure();
  uint64_t remainder = value % alignment;
  if (remainder == 0)
    return value;
  uint64_t addend = alignment - remainder;
  if (value > std::numeric_limits<uint64_t>::max() - addend)
    return failure();
  return value + addend;
}

struct TlaAllocPtrOffsetState {
  llvm::StringMap<uint64_t> nextOffsetByAddrspace;
  llvm::DenseMap<mlir::Value, uint64_t> offsetByAllocResult;
};

static FailureOr<uint64_t>
assignOrGetAllocPtrOffset(::tla::AllocPtrOp allocOp,
                          TlaAllocPtrOffsetState &state) {
  auto ptrTy = dyn_cast<::tla::PtrType>(allocOp.getResult().getType());
  if (!ptrTy)
    return failure();

  auto cached = state.offsetByAllocResult.find(allocOp.getResult());
  if (cached != state.offsetByAllocResult.end())
    return cached->second;

  int64_t sizeBytes = allocOp.getSizeBytesAttr().getInt();
  if (sizeBytes < 0)
    return failure();
  uint64_t alignment = ptrTy.getAlignment();
  std::string addrspaceKey =
      ::stringifyAddressSpace(ptrTy.getAddrspace()).str();
  FailureOr<uint64_t> start =
      alignUpCheckedU64(state.nextOffsetByAddrspace[addrspaceKey], alignment);
  FailureOr<uint64_t> alignedSize =
      alignUpCheckedU64(static_cast<uint64_t>(sizeBytes), alignment);
  if (failed(start) || failed(alignedSize) ||
      *start > std::numeric_limits<uint64_t>::max() - *alignedSize)
    return failure();

  state.offsetByAllocResult[allocOp.getResult()] = *start;
  state.nextOffsetByAddrspace[addrspaceKey] = *start + *alignedSize;
  return *start;
}

// Resolve tensor_ptr while tensor descriptors still carry source provenance.
// A pointer backed by another !tla.ptr aliases that address. A pointer backed
// by a kernel-argument memref starts at the memref's aligned address. The
// resulting inttoptr is source IR for the uniform ptr conversion below; it is
// not a long-lived lowering bridge.
static FailureOr<Value> resolveTensorBacking(Value tensor) {
  if (isa<MemRefType, ::tla::PtrType>(tensor.getType()))
    return tensor;
  if (auto desc = tensor.getDefiningOp<::tla::TensorDescOp>())
    return resolveTensorBacking(desc.getBase());
  if (auto tileView = tensor.getDefiningOp<::tla::TileViewOp>())
    return resolveTensorBacking(tileView.getSource());
  if (auto makeTensor = tensor.getDefiningOp<::tla::MakeTensorOp>())
    return makeTensor.getPtr();
  if (auto makeTensorLike = tensor.getDefiningOp<::tla::MakeTensorLikeOp>())
    return makeTensorLike.getPtr();
  return failure();
}

static LogicalResult resolveTensorPtrOps(ModuleOp module) {
  SmallVector<::tla::TensorPtrOp, 8> tensorPtrOps;
  module.walk([&](::tla::TensorPtrOp op) { tensorPtrOps.push_back(op); });
  for (::tla::TensorPtrOp op : tensorPtrOps) {
    if (!op || !op->getBlock())
      continue;

    FailureOr<Value> base = resolveTensorBacking(op.getSrc());
    if (failed(base)) {
      op.emitError() << "tensor_ptr source must resolve through tensor_desc, tile_view, or "
                        "make_tensor to a backing !tla.ptr or memref";
      return failure();
    }

    if (isa<::tla::PtrType>((*base).getType())) {
      op.getResult().replaceAllUsesWith(*base);
      op.erase();
      continue;
    }

    OpBuilder builder(op);
    Value addressIndex =
        builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
            op.getLoc(), *base);
    Value addressI64 = builder.create<arith::IndexCastOp>(
        op.getLoc(), builder.getI64Type(), addressIndex);
    Value pointer = builder.create<::tla::IntToPtrOp>(
        op.getLoc(), op.getResult().getType(), addressI64);
    op.getResult().replaceAllUsesWith(pointer);
    op.erase();
  }
  return success();
}

static bool isPointerConsumerBoundary(::tla::IntToPtrOp op) {
  if (op.getResult().use_empty())
    return false;
  return llvm::all_of(op.getResult().getUsers(), [](Operation *user) {
    return isa<::tla::MakeTensorOp, ::tla::MakeTensorLikeOp>(user);
  });
}

static Value materializePtrFromI64(OpBuilder &builder,
                                   ::tla::PtrType resultType,
                                   ValueRange inputs, Location loc) {
  if (inputs.size() != 1 || !inputs.front().getType().isInteger(64))
    return {};
  return builder.create<::tla::IntToPtrOp>(loc, resultType, inputs.front());
}

struct LowerAllocPtrPattern : public OpConversionPattern<::tla::AllocPtrOp> {
  LowerAllocPtrPattern(
      TypeConverter &converter, MLIRContext *context,
      const llvm::DenseMap<Value, uint64_t> &offsetByAllocResult)
      : OpConversionPattern(converter, context),
        offsetByAllocResult(offsetByAllocResult) {}

  LogicalResult
  matchAndRewrite(::tla::AllocPtrOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto offset = offsetByAllocResult.find(op.getResult());
    if (offset == offsetByAllocResult.end() ||
        offset->second >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return rewriter.notifyMatchFailure(
          op, "missing or overflowing static scratch offset");
    auto address = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), static_cast<int64_t>(offset->second), 64);
    address->setAttr(kAllocSizeBytesMetadataAttrName, op.getSizeBytesAttr());
    rewriter.replaceOp(op, address.getResult());
    return success();
  }

private:
  const llvm::DenseMap<Value, uint64_t> &offsetByAllocResult;
};

struct LowerIntToPtrPattern : public OpConversionPattern<::tla::IntToPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tla::IntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value address = castValueToI64(rewriter, op.getLoc(), adaptor.getAddr());
    if (!address.getType().isInteger(64))
      return rewriter.notifyMatchFailure(op, "address is not integer-like");
    rewriter.replaceOp(op, address);
    return success();
  }
};

struct LowerRecastPtrPattern : public OpConversionPattern<::tla::RecastPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tla::RecastPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

struct LowerPtrAddPattern : public OpConversionPattern<::tla::PtrAddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tla::PtrAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = dyn_cast<::tla::PtrType>(op.getPtr().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(op, "source is not !tla.ptr");
    int64_t elementBytes =
        getByteSizeOfFixedWidthScalarType(ptrType.getPointee());
    if (elementBytes <= 0)
      return rewriter.notifyMatchFailure(op, "pointee has no fixed byte width");

    Value address = adaptor.getPtr();
    Value elementOffset =
        castValueToI64(rewriter, op.getLoc(), adaptor.getOffset());
    if (!address.getType().isInteger(64) ||
        !elementOffset.getType().isInteger(64))
      return rewriter.notifyMatchFailure(
          op, "converted address/offset must be i64");

    Value byteOffset = elementOffset;
    if (elementBytes != 1) {
      Value scale =
          rewriter.create<arith::ConstantIntOp>(op.getLoc(), elementBytes, 64);
      byteOffset =
          rewriter.create<arith::MulIOp>(op.getLoc(), elementOffset, scale);
    }
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, address, byteOffset);
    return success();
  }
};

class TlaLowerPtrPass
    : public PassWrapper<TlaLowerPtrPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerPtrPass)

  StringRef getArgument() const override { return "tla-lower-ptr"; }
  StringRef getName() const override { return "TlaLowerPtrPass"; }
  StringRef getDescription() const override {
    return "Lower first-class !tla.ptr SSA values to i64 byte addresses, "
           "including structural SCF/function type conversion, and "
           "rematerialize ptr only at tensor-view consumers.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                    func::FuncDialect, ::tla::TlaDialect, mlir::memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    TlaAllocPtrOffsetState offsets;
    SmallVector<::tla::AllocPtrOp, 8> allocs;
    module.walk([&](::tla::AllocPtrOp op) { allocs.push_back(op); });
    for (::tla::AllocPtrOp op : allocs) {
      if (failed(assignOrGetAllocPtrOffset(op, offsets))) {
        op.emitError() << "failed to assign a static scratch byte offset";
        signalPassFailure();
        return;
      }
    }

    if (failed(resolveTensorPtrOps(module))) {
      signalPassFailure();
      return;
    }

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion(
        [&](::tla::PtrType) -> Type { return IntegerType::get(context, 64); });
    // Structural SCF conversion remaps region arguments separately from
    // operation results. Both paths may still feed a legal tensor consumer,
    // so rematerialize the typed pointer boundary for each kind of use.
    converter.addArgumentMaterialization(materializePtrFromI64);
    converter.addSourceMaterialization(materializePtrFromI64);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    patterns.add<LowerAllocPtrPattern>(converter, context,
                                       offsets.offsetByAllocResult);
    patterns.add<LowerIntToPtrPattern, LowerRecastPtrPattern,
                 LowerPtrAddPattern>(converter, context);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<arith::ArithDialect, mlir::memref::MemRefDialect,
                           ::tla::TlaDialect>();
    target
        .addIllegalOp<::tla::AllocPtrOp, ::tla::RecastPtrOp, ::tla::TensorPtrOp,
                      ::tla::PtrAddOp>();
    target.addDynamicallyLegalOp<::tla::IntToPtrOp>(isPointerConsumerBoundary);
    target.addLegalOp<::tla::MakeTensorOp, ::tla::MakeTensorLikeOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return converter.isSignatureLegal(op.getCalleeType());
    });
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op, converter) ||
             isLegalForReturnOpTypeConversionPattern(op, converter);
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerPtrPass() {
  return std::make_unique<TlaLowerPtrPass>();
}

void registerTlaLowerPtrPass() { PassRegistration<TlaLowerPtrPass>(); }

} // namespace tla
