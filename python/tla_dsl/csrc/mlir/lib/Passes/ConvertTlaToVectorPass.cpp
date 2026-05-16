#include "PassesCommon.h"
#include "PassesInternal.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace tla {
namespace {
struct ParsedTensorInfo {
  SmallVector<int64_t, 2> shape;
  SmallVector<int64_t, 2> coord;
  AddressSpace addressSpace;
  std::string elementType;
  std::string layoutTag;
};

static FailureOr<ParsedTensorInfo> parseTensorInfo(Type tensorType) {
  ParsedTensorInfo info;
  if (auto tensorTy = dyn_cast<::tla::TlaTensorType>(tensorType)) {
    auto layout = tensorTy.getLayout();
    auto ptr = tensorTy.getPtr();
    if (failed(::tla::getTlaIndexTreeLeaves(layout.getShape().getTree(), info.shape)) ||
        failed(::tla::getTlaIndexTreeLeaves(tensorTy.getCoord().getTree(), info.coord))) {
      return failure();
    }
    llvm::raw_string_ostream os(info.elementType);
    os << ptr.getPointee();
    os.flush();
    info.addressSpace = ptr.getAddrspace();
    info.layoutTag = stringifyLayoutTag(layout.getLayoutTag()).str();
    return info;
  }
  return failure();
}

static FailureOr<MemRefType> getBridgedTensorMemrefType(Value tensor) {
  auto bridged = bridgeTlaFuncTensorType(tensor.getType());
  if (failed(bridged))
    return failure();
  return *bridged;
}

static FailureOr<Value> castMemrefToExpected(PatternRewriter &rewriter, Location loc, Value value,
                                             MemRefType expectedType) {
  if (value.getType() == expectedType)
    return value;
  if (!isa<MemRefType>(value.getType()))
    return failure();
  return rewriter.create<mlir::memref::CastOp>(loc, expectedType, value).getResult();
}

static FailureOr<Value> materializeBaseMemref(PatternRewriter &rewriter, Location loc,
                                              Value tensor) {
  if (auto makeRmem = tensor.getDefiningOp<::tla::MakeRmemTensorOp>())
    return materializeBaseMemref(rewriter, loc, makeRmem.getSource());

  if (auto castOp = tensor.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp.getNumOperands() == 1 && isa<MemRefType>(castOp.getOperand(0).getType())) {
      auto expected = getBridgedTensorMemrefType(tensor);
      if (failed(expected))
        return failure();
      return castMemrefToExpected(rewriter, loc, castOp.getOperand(0), *expected);
    }
  }

  if (auto makeTensor = tensor.getDefiningOp<::tla::MakeTensorLikeOp>()) {
    Value ptr = makeTensor.getPtr();
    if (auto bridge = ptr.getDefiningOp<::tla::HivmMemrefAsPtrOp>()) {
      auto expected = getBridgedTensorMemrefType(tensor);
      if (failed(expected))
        return failure();
      return castMemrefToExpected(rewriter, loc, bridge.getMemref(), *expected);
    }
    if (auto ptrCast = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (ptrCast.getNumOperands() == 1 && isa<MemRefType>(ptrCast.getOperand(0).getType())) {
        auto expected = getBridgedTensorMemrefType(tensor);
        if (failed(expected))
          return failure();
        return castMemrefToExpected(rewriter, loc, ptrCast.getOperand(0), *expected);
      }
    }
    return failure();
  }

  if (auto tileView = tensor.getDefiningOp<::tla::TileViewOp>()) {
    Value source = tileView.getSource();
    if (isa<MemRefType>(source.getType())) {
      auto expected = getBridgedTensorMemrefType(tensor);
      if (failed(expected))
        return failure();
      return castMemrefToExpected(rewriter, loc, source, *expected);
    }
    return materializeBaseMemref(rewriter, loc, source);
  }

  if (isa<MemRefType>(tensor.getType()))
    return tensor;

  return failure();
}

static FailureOr<Value> materializeCopySubview1D(PatternRewriter &rewriter, Location loc,
                                                 Value tensor) {
  auto info = parseTensorInfo(tensor.getType());
  if (failed(info))
    return failure();
  if (info->shape.size() != 1 || info->coord.size() != 1 || info->elementType != "f32" ||
      info->layoutTag != "row_major")
    return failure();
  if (info->shape[0] != 64 || info->coord[0] != 0)
    return failure();

  auto baseMemref = materializeBaseMemref(rewriter, loc, tensor);
  if (failed(baseMemref))
    return failure();
  auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
  if (!baseType || baseType.getRank() != 1)
    return failure();

  auto ctx = rewriter.getContext();
  auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, ArrayRef<int64_t>{1});
  auto subviewType = MemRefType::get({ShapedType::kDynamic}, baseType.getElementType(), layout,
                                     baseType.getMemorySpace());
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value size = rewriter.create<arith::ConstantIndexOp>(loc, 64);
  auto dynamicEntries = DenseI64ArrayAttr::get(ctx, ArrayRef<int64_t>{ShapedType::kDynamic});
  auto staticStride = DenseI64ArrayAttr::get(ctx, ArrayRef<int64_t>{1});
  return rewriter
      .create<mlir::memref::SubViewOp>(loc, subviewType, *baseMemref, ValueRange{zero},
                                       ValueRange{size}, ValueRange{}, dynamicEntries,
                                       dynamicEntries, staticStride)
      .getResult();
}

static Value unwrapFragmentSource(Value value) {
  if (auto makeRmem = value.getDefiningOp<::tla::MakeRmemTensorOp>())
    return makeRmem.getSource();
  return value;
}

static std::string buildUniqueVectorHelperName(ModuleOp module, int &nextVectorRegionId) {
  std::string helperName;
  do {
    helperName = "vector_region_" + std::to_string(nextVectorRegionId++);
  } while (module.lookupSymbol<func::FuncOp>(helperName));
  return helperName;
}

static FailureOr<func::FuncOp> buildHelperFunc(ModuleOp module, func::FuncOp parentFunc,
                                               ::tla::VectorOp vectorOp, Value dst, Value lhs,
                                               Value rhs, int &nextVectorRegionId) {
  MLIRContext *ctx = module.getContext();
  OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPointAfter(parentFunc);

  std::string helperName = buildUniqueVectorHelperName(module, nextVectorRegionId);

  auto dstType = getBridgedTensorMemrefType(dst);
  auto lhsType = getBridgedTensorMemrefType(lhs);
  auto rhsType = getBridgedTensorMemrefType(rhs);
  if (failed(dstType) || failed(lhsType) || failed(rhsType))
    return failure();

  auto funcType =
      moduleBuilder.getFunctionType(TypeRange{*lhsType, *rhsType, *dstType}, TypeRange{});
  auto helper = moduleBuilder.create<func::FuncOp>(vectorOp.getLoc(), helperName, funcType);
  helper.setPrivate();
  helper->setAttr(hivm::TFuncCoreTypeAttr::name,
                  hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
  helper->setAttr("hivm.vector_function", UnitAttr::get(ctx));
  helper->setAttr("no_inline", UnitAttr::get(ctx));

  Block *entry = helper.addEntryBlock();
  OpBuilder b = OpBuilder::atBlockBegin(entry);
  Value zero = b.create<arith::ConstantIndexOp>(vectorOp.getLoc(), 0);
  Value zeroF32 = b.create<arith::ConstantOp>(vectorOp.getLoc(), b.getF32FloatAttr(0.0));
  auto vecType = VectorType::get({64}, b.getF32Type());
  auto inBoundsAttr = b.getDenseBoolArrayAttr({true});
  auto lhsVec = b.create<vector::TransferReadOp>(vectorOp.getLoc(), vecType, entry->getArgument(0),
                                                 ValueRange{zero}, zeroF32, inBoundsAttr);
  auto rhsVec = b.create<vector::TransferReadOp>(vectorOp.getLoc(), vecType, entry->getArgument(1),
                                                 ValueRange{zero}, zeroF32, inBoundsAttr);
  auto sum = b.create<arith::AddFOp>(vectorOp.getLoc(), lhsVec, rhsVec);
  b.create<vector::TransferWriteOp>(vectorOp.getLoc(), sum, entry->getArgument(2), ValueRange{zero},
                                    inBoundsAttr);
  b.create<func::ReturnOp>(vectorOp.getLoc());

  return helper;
}

class LowerVectorRegionPattern : public OpRewritePattern<::tla::VectorOp> {
public:
  LowerVectorRegionPattern(MLIRContext *context, ModuleOp module, int &nextVectorRegionId)
      : OpRewritePattern<::tla::VectorOp>(context), module(module),
        nextVectorRegionId(nextVectorRegionId) {}

  LogicalResult matchAndRewrite(::tla::VectorOp vectorOp,
                                PatternRewriter &rewriter) const override {
    auto *body = vectorOp.getBody().empty() ? nullptr : &vectorOp.getBody().front();
    if (!body) {
      return rewriter.notifyMatchFailure(vectorOp, "expected tla.vector body");
    }

    ::tla::VaddOp vadd;
    for (Operation &op : body->without_terminator()) {
      if (auto candidate = dyn_cast<::tla::VaddOp>(op)) {
        if (vadd) {
          return rewriter.notifyMatchFailure(vectorOp,
                                             "expected exactly one tla.vadd in tla.vector body");
        }
        vadd = candidate;
        continue;
      }
      if (!isa<::tla::LoadOp, ::tla::StoreOp>(op)) {
        return rewriter.notifyMatchFailure(
            vectorOp, "expected tla.vector body to contain only tla.load/tla.store/tla.vadd");
      }
    }
    if (!vadd)
      return rewriter.notifyMatchFailure(vectorOp, "expected tla.vector body to contain tla.vadd");

    auto funcOp = vectorOp->getParentOfType<func::FuncOp>();
    if (!funcOp)
      return rewriter.notifyMatchFailure(vectorOp, "expected enclosing func.func");

    Value dst = unwrapFragmentSource(vadd.getDst());
    Value lhs = unwrapFragmentSource(vadd.getLhs());
    Value rhs = unwrapFragmentSource(vadd.getRhs());

    auto helperOr = buildHelperFunc(module, funcOp, vectorOp, dst, lhs, rhs, nextVectorRegionId);
    if (failed(helperOr)) {
      return rewriter.notifyMatchFailure(vectorOp, "failed to build vector helper function");
    }
    auto helper = *helperOr;

    auto lhsBase = materializeBaseMemref(rewriter, vectorOp.getLoc(), lhs);
    auto rhsBase = materializeBaseMemref(rewriter, vectorOp.getLoc(), rhs);
    auto dstBase = materializeBaseMemref(rewriter, vectorOp.getLoc(), dst);
    if (failed(lhsBase) || failed(rhsBase) || failed(dstBase)) {
      return rewriter.notifyMatchFailure(vectorOp,
                                         "failed to materialize UB memrefs for vector helper call");
    }

    auto lhsType = getBridgedTensorMemrefType(lhs);
    auto rhsType = getBridgedTensorMemrefType(rhs);
    auto dstType = getBridgedTensorMemrefType(dst);
    if (failed(lhsType) || failed(rhsType) || failed(dstType)) {
      return rewriter.notifyMatchFailure(vectorOp, "failed to derive helper memref signature");
    }
    auto lhsArg = castMemrefToExpected(rewriter, vectorOp.getLoc(), *lhsBase, *lhsType);
    auto rhsArg = castMemrefToExpected(rewriter, vectorOp.getLoc(), *rhsBase, *rhsType);
    auto dstArg = castMemrefToExpected(rewriter, vectorOp.getLoc(), *dstBase, *dstType);
    if (failed(lhsArg) || failed(rhsArg) || failed(dstArg)) {
      return rewriter.notifyMatchFailure(vectorOp,
                                         "failed to cast helper operands to expected memref types");
    }

    auto call = rewriter.create<func::CallOp>(vectorOp.getLoc(), helper,
                                              ValueRange{*lhsArg, *rhsArg, *dstArg});
    call->setAttr("hivm.vector_function", UnitAttr::get(rewriter.getContext()));
    call->setAttr("no_inline", UnitAttr::get(rewriter.getContext()));
    rewriter.eraseOp(vectorOp);
    return success();
  }

private:
  ModuleOp module;
  int &nextVectorRegionId;
};

class LowerCopyPattern : public OpRewritePattern<::tla::CopyOp> {
public:
  using OpRewritePattern<::tla::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::tla::CopyOp copyOp, PatternRewriter &rewriter) const override {
    auto dstInfo = parseTensorInfo(copyOp.getDst().getType());
    auto srcInfo = parseTensorInfo(copyOp.getSrc().getType());
    if (failed(dstInfo) || failed(srcInfo))
      return failure();

    auto srcSubview = materializeCopySubview1D(rewriter, copyOp.getLoc(), copyOp.getSrc());
    auto dstSubview = materializeCopySubview1D(rewriter, copyOp.getLoc(), copyOp.getDst());
    if (failed(srcSubview) || failed(dstSubview))
      return failure();

    if (srcInfo->addressSpace == AddressSpace::gm && dstInfo->addressSpace == AddressSpace::ub) {
      auto padModeAttr = rewriter.getAttr<hivm::PadModeAttr>(hivm::PadMode::PadValue);
      Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(copyOp.getLoc(), 0);
      Value zeroF32 =
          rewriter.create<arith::ConstantOp>(copyOp.getLoc(), rewriter.getF32FloatAttr(0.0));
      auto load = rewriter.create<hivm::LoadOp>(copyOp.getLoc(), TypeRange{}, *srcSubview,
                                                *dstSubview, padModeAttr, zeroF32, zeroIndex);
      load->removeAttr("init_out_buffer");
      load->removeAttr("may_implicit_transpose_with_last_axis");
      rewriter.eraseOp(copyOp);
      return success();
    }

    if (srcInfo->addressSpace == AddressSpace::ub && dstInfo->addressSpace == AddressSpace::gm) {
      rewriter.create<hivm::StoreOp>(copyOp.getLoc(), TypeRange{}, *srcSubview, *dstSubview);
      rewriter.eraseOp(copyOp);
      return success();
    }

    return failure();
  }
};

template <typename OpTy> class EraseDeadTlaScaffoldingPattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    if (!llvm::all_of(op->getResults(), [](Value value) { return value.use_empty(); }))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

static void populateTlaToVectorPatterns(RewritePatternSet &patterns, ModuleOp module,
                                        int &nextVectorRegionId) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<LowerVectorRegionPattern>(ctx, module, nextVectorRegionId);
  patterns.add<LowerCopyPattern>(ctx);
  patterns.add<EraseDeadTlaScaffoldingPattern<::tla::MakeRmemTensorOp>,
               EraseDeadTlaScaffoldingPattern<::tla::MakeTensorLikeOp>,
               EraseDeadTlaScaffoldingPattern<::tla::TileViewOp>,
               EraseDeadTlaScaffoldingPattern<::tla::MakeShapeOp>,
               EraseDeadTlaScaffoldingPattern<::tla::MakeCoordOp>,
               EraseDeadTlaScaffoldingPattern<::tla::RecastPtrOp>,
               EraseDeadTlaScaffoldingPattern<::tla::HivmMemrefAsPtrOp>>(ctx);
}

class ConvertTlaToVectorPass : public PassWrapper<ConvertTlaToVectorPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTlaToVectorPass)

  StringRef getArgument() const override { return "tla-to-vector"; }
  StringRef getName() const override { return "TlaToVectorPass"; }
  StringRef getDescription() const override {
    return "Outline tla.vector regions and lower fragment ops to vector IR.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect,
                    hivm::HIVMDialect, vector::VectorDialect, ::tla::TlaDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr(kTlaHasVectorRegionAttrName))
      return;

    nextVectorRegionId = 0;

    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.isDeclaration())
        continue;
      RewritePatternSet patterns(&getContext());
      populateTlaToVectorPatterns(patterns, module, nextVectorRegionId);
      if (failed(mlir::applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    module->removeAttr(kTlaHasVectorRegionAttrName);
  }

private:
  int nextVectorRegionId = 0;
};

} // namespace

std::unique_ptr<Pass> createConvertTlaToVectorPass() {
  return std::make_unique<ConvertTlaToVectorPass>();
}

void registerConvertTlaToVectorPass() { PassRegistration<ConvertTlaToVectorPass>(); }

} // namespace tla
