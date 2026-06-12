#include "PassesCommon.h"
#include "PassesInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace tla {
namespace {
struct ParsedTensorInfo {
  SmallVector<int64_t, 2> shape;
  SmallVector<int64_t, 2> originShape;
  SmallVector<int64_t, 2> coord;
  AddressSpace addressSpace;
  Type elementType;
  std::string layoutTag;
};

static FailureOr<ParsedTensorInfo> parseTensorInfo(Type tensorType) {
  ParsedTensorInfo info;
  if (auto tensorTy = dyn_cast<::tla::TlaTensorType>(tensorType)) {
    auto layout = tensorTy.getLayout();
    auto ptr = tensorTy.getPtr();
    if (failed(::tla::getTlaIndexTreeLeaves(layout.getShape().getTree(), info.shape)) ||
        failed(::tla::getTlaIndexTreeLeaves(layout.getOrigin().getTree(), info.originShape)) ||
        failed(::tla::getTlaIndexTreeLeaves(tensorTy.getCoord().getTree(), info.coord))) {
      return failure();
    }
    info.elementType = ptr.getPointee();
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

static FailureOr<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t numElements = 1;
  for (int64_t dim : shape) {
    if (dim <= 0 || dim == ShapedType::kDynamic)
      return failure();
    numElements *= dim;
  }
  return numElements;
}

static FailureOr<int64_t> getElementByteWidth(Type elementType) {
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    int64_t width = intType.getWidth();
    if (width <= 0 || width % 8 != 0)
      return failure();
    return width / 8;
  }
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    int64_t width = floatType.getWidth();
    if (width <= 0 || width % 8 != 0)
      return failure();
    return width / 8;
  }
  return failure();
}

static FailureOr<int64_t> getVectorLanesForMemref(MemRefType type) {
  if (type.getRank() != 1 && type.getRank() != 2)
    return failure();
  auto numElements = getStaticNumElements(type.getShape());
  auto elementBytes = getElementByteWidth(type.getElementType());
  if (failed(numElements) || failed(elementBytes) || *numElements <= 0 || *elementBytes <= 0)
    return failure();
  constexpr int64_t kVectorBytes = 256;
  int64_t lanes = kVectorBytes / *elementBytes;
  if (lanes <= 0 || *numElements > lanes)
    return failure();
  return lanes;
}

static bool isSupportedVectorTile(MemRefType type) {
  return succeeded(getVectorLanesForMemref(type));
}

static FailureOr<MemRefType> getVectorHelperMemrefType(Value tensor) {
  auto bridged = getBridgedTensorMemrefType(tensor);
  if (failed(bridged))
    return failure();
  if (!isSupportedVectorTile(*bridged))
    return failure();
  auto numElements = getStaticNumElements(bridged->getShape());
  if (failed(numElements))
    return failure();
  auto info = parseTensorInfo(tensor.getType());
  if (succeeded(info)) {
    auto originElements = getStaticNumElements(info->originShape);
    if (succeeded(originElements) && *originElements > 0 && *originElements <= *numElements)
      numElements = originElements;
  }
  auto layout =
      StridedLayoutAttr::get(bridged->getContext(), ShapedType::kDynamic, ArrayRef<int64_t>{1});
  return MemRefType::get({*numElements}, bridged->getElementType(), layout,
                         bridged->getMemorySpace());
}

static LogicalResult verifyHelperMemrefType(MemRefType type, VectorType vectorType) {
  if (!type || type.getRank() != 1 || vectorType.getRank() != 1)
    return failure();
  if (type.getElementType() != vectorType.getElementType())
    return failure();
  auto validLanes = getStaticNumElements(type.getShape());
  if (failed(validLanes) || *validLanes <= 0 || *validLanes > vectorType.getDimSize(0))
    return failure();
  auto lanes = getVectorLanesForMemref(type);
  if (failed(lanes) || *lanes != vectorType.getDimSize(0))
    return failure();
  return success();
}

static FailureOr<Value> createZeroValue(OpBuilder &builder, Location loc, Type elementType) {
  if (elementType.isF32())
    return builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0.0)).getResult();
  if (elementType.isF16())
    return builder.create<arith::ConstantOp>(loc, builder.getF16FloatAttr(0.0)).getResult();
  if (isa<BFloat16Type>(elementType))
    return builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(elementType, 0.0))
        .getResult();
  if (auto intType = dyn_cast<IntegerType>(elementType))
    return builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(intType, 0)).getResult();
  return failure();
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
                                              Value tensor);

static FailureOr<Value> materializeSingleCoordIndex(PatternRewriter &rewriter, Location loc,
                                                    Value coord) {
  auto coordType = dyn_cast<::tla::CoordType>(coord.getType());
  if (!coordType)
    return failure();
  SmallVector<int64_t, 1> leaves;
  if (failed(::tla::getTlaIndexTreeLeaves(coordType.getTree(), leaves)) || leaves.size() != 1)
    return failure();
  if (leaves[0] != ShapedType::kDynamic)
    return rewriter.create<arith::ConstantIndexOp>(loc, leaves[0]).getResult();

  auto makeCoord = coord.getDefiningOp<::tla::MakeCoordOp>();
  if (!makeCoord || makeCoord.getDynElems().size() != 1)
    return failure();
  Value dynCoord = *makeCoord.getDynElems().begin();
  if (!dynCoord.getType().isIndex())
    return failure();
  return dynCoord;
}

static FailureOr<Value> materializeTileViewMemref(PatternRewriter &rewriter, Location loc,
                                                  Value tensor, ::tla::TileViewOp tileView,
                                                  bool useVectorHelperType = false) {
  auto info = parseTensorInfo(tensor.getType());
  if (failed(info))
    return failure();
  if (info->shape.size() != 1 || info->coord.size() != 1 || info->layoutTag != "row_major")
    return failure();
  auto numElements = getStaticNumElements(info->shape);
  if (failed(numElements))
    return failure();

  auto expected =
      useVectorHelperType ? getVectorHelperMemrefType(tensor) : getBridgedTensorMemrefType(tensor);
  if (failed(expected) || expected->getRank() != 1)
    return failure();

  auto baseMemref = materializeBaseMemref(rewriter, loc, tileView.getSource());
  if (failed(baseMemref))
    return failure();
  auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
  if (!baseType || baseType.getRank() != 1)
    return failure();

  Value offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  if (!baseType.hasStaticShape() || baseType.getDimSize(0) != info->shape[0]) {
    auto dynamicOffset = materializeSingleCoordIndex(rewriter, loc, tileView.getCoord());
    if (failed(dynamicOffset))
      return failure();
    offset = *dynamicOffset;
  }

  int64_t sizeElements =
      useVectorHelperType && expected->hasStaticShape() ? expected->getDimSize(0) : *numElements;
  Value size = rewriter.create<arith::ConstantIndexOp>(loc, sizeElements);
  Value stride = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  return rewriter
      .create<mlir::memref::ReinterpretCastOp>(loc, *expected, *baseMemref, offset,
                                               ValueRange{size}, ValueRange{stride})
      .getResult();
}

static FailureOr<Value> materializeBaseMemref(PatternRewriter &rewriter, Location loc,
                                              Value tensor) {
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
    auto info = parseTensorInfo(tensor.getType());
    if (succeeded(info) && info->shape.size() == 1 &&
        succeeded(getVectorHelperMemrefType(tensor)))
      return materializeTileViewMemref(rewriter, loc, tensor, tileView,
                                       /*useVectorHelperType=*/true);
    if (isa<MemRefType>(source.getType())) {
      if (succeeded(info) && info->shape.size() == 1)
        return materializeTileViewMemref(rewriter, loc, tensor, tileView);
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
  if (info->shape.size() != 1 || info->coord.size() != 1 || !info->elementType.isF32() ||
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

static std::string buildUniqueVectorHelperName(ModuleOp module, int &nextVectorRegionId) {
  std::string helperName;
  do {
    helperName = "vector_region_" + std::to_string(nextVectorRegionId++);
  } while (module.lookupSymbol<func::FuncOp>(helperName));
  return helperName;
}

static FailureOr<func::FuncOp> buildHelperFunc(ModuleOp module, func::FuncOp parentFunc,
                                               Operation *vectorOp, Value dst, Value lhs,
                                               Value rhs, int &nextVectorRegionId) {
  MLIRContext *ctx = module.getContext();
  OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPointAfter(parentFunc);

  std::string helperName = buildUniqueVectorHelperName(module, nextVectorRegionId);

  auto dstType = getVectorHelperMemrefType(dst);
  auto lhsType = getVectorHelperMemrefType(lhs);
  auto rhsType = getVectorHelperMemrefType(rhs);
  if (failed(dstType) || failed(lhsType) || failed(rhsType))
    return failure();

  auto funcType =
      moduleBuilder.getFunctionType(TypeRange{*lhsType, *rhsType, *dstType}, TypeRange{});
  auto helper = moduleBuilder.create<func::FuncOp>(vectorOp->getLoc(), helperName, funcType);
  helper.setPrivate();
  helper->setAttr(hivm::TFuncCoreTypeAttr::name,
                  hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
  helper->setAttr("hivm.vector_function", UnitAttr::get(ctx));
  helper->setAttr("no_inline", UnitAttr::get(ctx));

  Block *entry = helper.addEntryBlock();
  OpBuilder b = OpBuilder::atBlockBegin(entry);
  auto lhsMemrefType = dyn_cast<MemRefType>(entry->getArgument(0).getType());
  auto rhsMemrefType = dyn_cast<MemRefType>(entry->getArgument(1).getType());
  auto dstMemrefType = dyn_cast<MemRefType>(entry->getArgument(2).getType());
  if (!lhsMemrefType || !rhsMemrefType || !dstMemrefType)
    return failure();
  auto lanes = getVectorLanesForMemref(lhsMemrefType);
  if (failed(lanes))
    return failure();
  auto vecType = VectorType::get({*lanes}, lhsMemrefType.getElementType());
  if (failed(verifyHelperMemrefType(lhsMemrefType, vecType)) ||
      failed(verifyHelperMemrefType(rhsMemrefType, vecType)) ||
      failed(verifyHelperMemrefType(dstMemrefType, vecType)))
    return failure();
  auto zeroValue = createZeroValue(b, vectorOp->getLoc(), lhsMemrefType.getElementType());
  if (failed(zeroValue))
    return failure();
  Value zero = b.create<arith::ConstantIndexOp>(vectorOp->getLoc(), 0);
  auto getInBoundsAttr = [&](MemRefType type) {
    return b.getBoolArrayAttr({type.getDimSize(0) == vecType.getDimSize(0)});
  };
  auto lhsPermutationMap = AffineMap::getMinorIdentityMap(lhsMemrefType.getRank(), vecType.getRank(), ctx);
  auto rhsPermutationMap = AffineMap::getMinorIdentityMap(rhsMemrefType.getRank(), vecType.getRank(), ctx);
  auto dstPermutationMap = AffineMap::getMinorIdentityMap(dstMemrefType.getRank(), vecType.getRank(), ctx);
  auto lhsVec = b.create<vector::TransferReadOp>(
      vectorOp->getLoc(), vecType, entry->getArgument(0), ValueRange{zero},
      AffineMapAttr::get(lhsPermutationMap), *zeroValue, Value(), getInBoundsAttr(lhsMemrefType));
  auto rhsVec = b.create<vector::TransferReadOp>(
      vectorOp->getLoc(), vecType, entry->getArgument(1), ValueRange{zero},
      AffineMapAttr::get(rhsPermutationMap), *zeroValue, Value(), getInBoundsAttr(rhsMemrefType));
  Value sum;
  Type elementType = vecType.getElementType();
  if (isa<IntegerType>(elementType)) {
    sum = b.create<arith::AddIOp>(vectorOp->getLoc(), lhsVec, rhsVec);
  } else if (isa<FloatType>(elementType)) {
    sum = b.create<arith::AddFOp>(vectorOp->getLoc(), lhsVec, rhsVec);
  } else {
    return vectorOp->emitError("unsupported element type for vector add helper: ")
           << elementType;
  }
  b.create<vector::TransferWriteOp>(vectorOp->getLoc(), sum, entry->getArgument(2),
                                    ValueRange{zero}, AffineMapAttr::get(dstPermutationMap),
                                    Value(), getInBoundsAttr(dstMemrefType));
  b.create<func::ReturnOp>(vectorOp->getLoc());

  return helper;
}

template <typename RegionOpT> class LowerVectorRegionPattern : public OpRewritePattern<RegionOpT> {
public:
  LowerVectorRegionPattern(MLIRContext *context, ModuleOp module, int &nextVectorRegionId)
      : OpRewritePattern<RegionOpT>(context), module(module), nextVectorRegionId(nextVectorRegionId) {}

  LogicalResult matchAndRewrite(RegionOpT vectorOp, PatternRewriter &rewriter) const override {
    auto *body = vectorOp.getBody().empty() ? nullptr : &vectorOp.getBody().front();
    if (!body) {
      return rewriter.notifyMatchFailure(vectorOp, "expected tla.vector body");
    }

    ::tla::StoreOp store;
    vectorOp->walk([&](::tla::StoreOp candidate) {
        if (store) {
          return WalkResult::interrupt();
        }
        store = candidate;
        return WalkResult::advance();
    });
    int storeCount = 0;
    vectorOp->walk([&](::tla::StoreOp) { ++storeCount; });
    if (storeCount > 1) {
      return rewriter.notifyMatchFailure(vectorOp,
                                         "expected exactly one tla.store in tla.vector body");
      }
    if (!store)
      return rewriter.notifyMatchFailure(vectorOp, "expected tla.vector body to contain tla.store");

    auto add = store.getSource().getDefiningOp<::tla::AddOp>();
    if (!add)
      return rewriter.notifyMatchFailure(vectorOp, "expected tla.store source to be tla.add");

    auto lhsLoad = add.getLhs().getDefiningOp<::tla::LoadOp>();
    auto rhsLoad = add.getRhs().getDefiningOp<::tla::LoadOp>();
    if (!lhsLoad || !rhsLoad)
      return rewriter.notifyMatchFailure(vectorOp, "expected tla.add operands to be tla.load");

    auto funcOp = vectorOp.getOperation()->template getParentOfType<func::FuncOp>();
    if (!funcOp)
      return rewriter.notifyMatchFailure(vectorOp, "expected enclosing func.func");

    Value dst = store.getDest();
    Value lhs = lhsLoad.getSource();
    Value rhs = rhsLoad.getSource();

    auto helperOr =
        buildHelperFunc(module, funcOp, vectorOp.getOperation(), dst, lhs, rhs, nextVectorRegionId);
    if (failed(helperOr)) {
      return rewriter.notifyMatchFailure(vectorOp, "failed to build vector helper function");
    }
    auto helper = *helperOr;

    rewriter.setInsertionPoint(store);
    auto lhsBase = materializeBaseMemref(rewriter, vectorOp.getLoc(), lhs);
    auto rhsBase = materializeBaseMemref(rewriter, vectorOp.getLoc(), rhs);
    auto dstBase = materializeBaseMemref(rewriter, vectorOp.getLoc(), dst);
    if (failed(lhsBase) || failed(rhsBase) || failed(dstBase)) {
      return rewriter.notifyMatchFailure(vectorOp,
                                         "failed to materialize UB memrefs for vector helper call");
    }

    auto lhsType = getVectorHelperMemrefType(lhs);
    auto rhsType = getVectorHelperMemrefType(rhs);
    auto dstType = getVectorHelperMemrefType(dst);
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
    rewriter.eraseOp(store);
    rewriter.eraseOp(add);
    rewriter.eraseOp(lhsLoad);
    rewriter.eraseOp(rhsLoad);
    rewriter.inlineBlockBefore(body, vectorOp->getBlock(), vectorOp->getIterator());
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
  patterns.add<LowerVectorRegionPattern<::tla::VectorOp>, LowerVectorRegionPattern<::tla::VecFuncOp>>(
      ctx, module, nextVectorRegionId);
  patterns.add<LowerCopyPattern>(ctx);
  patterns.add<EraseDeadTlaScaffoldingPattern<::tla::MakeTensorLikeOp>,
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
