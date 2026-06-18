#include "PassesCommon.h"
#include "PassesInternal.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
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
  SmallVector<int64_t, 2> strides;
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
        failed(::tla::getTlaIndexTreeLeaves(tensorTy.getCoord().getTree(), info.coord)) ||
        failed(::tla::getTlaIndexTreeLeaves(layout.getStride().getTree(), info.strides))) {
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

static FailureOr<MemRefType>
getVectorHelperMemrefType(Value tensor,
                          DenseMap<Value, Value> *handoffMemrefByTensor = nullptr) {
  FailureOr<MemRefType> bridged = failure();
  if (handoffMemrefByTensor) {
    auto handoff = handoffMemrefByTensor->find(tensor);
    if (handoff != handoffMemrefByTensor->end()) {
      auto handoffType = dyn_cast<MemRefType>(handoff->second.getType());
      if (handoffType)
        bridged = handoffType;
    }
  }
  if (failed(bridged))
    bridged = getBridgedTensorMemrefType(tensor);
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
  auto sourceType = dyn_cast<MemRefType>(value.getType());
  if (!sourceType)
    return failure();

  auto hasSameStaticElementStorage = [](MemRefType sourceType, MemRefType expectedType) {
    return sourceType.hasStaticShape() && expectedType.hasStaticShape() &&
           sourceType.getElementType() == expectedType.getElementType() &&
           sourceType.getMemorySpace() == expectedType.getMemorySpace() &&
           sourceType.getNumElements() == expectedType.getNumElements();
  };

  if (sourceType.getRank() == 1 && expectedType.getRank() == 2 &&
      hasSameStaticElementStorage(sourceType, expectedType)) {
    SmallVector<ReassociationIndices> reassociation{{0, 1}};
    Value expanded =
        rewriter.create<mlir::memref::ExpandShapeOp>(loc, expectedType, value, reassociation)
            .getResult();
    return expanded;
  }

  if (sourceType.getRank() == 2 && expectedType.getRank() == 1 &&
      hasSameStaticElementStorage(sourceType, expectedType)) {
    SmallVector<ReassociationIndices> reassociation{{0, 1}};
    auto collapsedLayout =
        StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic, ArrayRef<int64_t>{1});
    auto collapsedType = MemRefType::get({sourceType.getNumElements()}, expectedType.getElementType(),
                                         collapsedLayout, expectedType.getMemorySpace());
    Value collapsed =
        rewriter.create<mlir::memref::CollapseShapeOp>(loc, collapsedType, value, reassociation)
            .getResult();
    if (collapsed.getType() == expectedType)
      return collapsed;
    return rewriter.create<mlir::memref::CastOp>(loc, expectedType, collapsed).getResult();
  }

  return rewriter.create<mlir::memref::CastOp>(loc, expectedType, value).getResult();
}

static FailureOr<Value>
materializeBaseMemref(PatternRewriter &rewriter, Location loc, Value tensor,
                      DenseMap<Value, Value> *handoffMemrefByTensor = nullptr);
static FailureOr<Value> materializeCopySubviewRank2(
    PatternRewriter &rewriter, Location loc, Value tensor,
    DenseMap<Value, Value> *handoffMemrefByTensor = nullptr,
    ArrayRef<int64_t> concreteShape = {});

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

static FailureOr<SmallVector<Value, 2>> materializeRank2CoordIndices(PatternRewriter &rewriter,
                                                                     Location loc, Value coord) {
  auto coordType = dyn_cast<::tla::CoordType>(coord.getType());
  if (!coordType)
    return failure();
  SmallVector<int64_t, 2> leaves;
  if (failed(::tla::getTlaIndexTreeLeaves(coordType.getTree(), leaves)) || leaves.size() != 2)
    return failure();

  SmallVector<Value, 2> values;
  auto dynIt = coord.getDefiningOp<::tla::MakeCoordOp>();
  auto dynElems = dynIt ? dynIt.getDynElems() : ValueRange{};
  unsigned dynIndex = 0;
  for (int64_t leaf : leaves) {
    if (leaf != ShapedType::kDynamic) {
      values.push_back(rewriter.create<arith::ConstantIndexOp>(loc, leaf));
      continue;
    }
    if (!dynIt || dynIndex >= dynElems.size())
      return failure();
    Value dynCoord = dynElems[dynIndex++];
    if (!dynCoord.getType().isIndex())
      return failure();
    values.push_back(dynCoord);
  }
  return values;
}

static FailureOr<Value> materializeTileViewMemref(PatternRewriter &rewriter, Location loc,
                                                  Value tensor, ::tla::TileViewOp tileView,
                                                  bool useVectorHelperType = false,
                                                  DenseMap<Value, Value> *handoffMemrefByTensor =
                                                      nullptr) {
  auto info = parseTensorInfo(tensor.getType());
  if (failed(info))
    return failure();
  if ((info->shape.size() != 1 && info->shape.size() != 2) ||
      info->coord.size() != info->shape.size() || info->strides.size() != info->shape.size() ||
      info->layoutTag != "row_major")
    return failure();
  auto numElements = getStaticNumElements(info->shape);
  if (failed(numElements))
    return failure();

  auto expected =
      useVectorHelperType ? getVectorHelperMemrefType(tensor) : getBridgedTensorMemrefType(tensor);
  if (failed(expected))
    return failure();

  auto baseMemref = materializeBaseMemref(rewriter, loc, tileView.getSource(),
                                          handoffMemrefByTensor);
  if (failed(baseMemref))
    return failure();
  auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
  if (!baseType)
    return failure();

  if (baseType.getRank() == 2) {
    auto sourceInfo = parseTensorInfo(tileView.getSource().getType());
    if (failed(sourceInfo) || sourceInfo->shape.size() != 2)
      return failure();
    auto coordPair = materializeRank2CoordIndices(rewriter, loc, tileView.getCoord());
    if (failed(coordPair))
      return failure();

    auto subviewLayout = StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                                                ArrayRef<int64_t>{sourceInfo->shape[1], 1});
    auto subviewType = MemRefType::get(info->shape, baseType.getElementType(), subviewLayout,
                                       baseType.getMemorySpace());
    SmallVector<int64_t> staticOffsets{ShapedType::kDynamic, ShapedType::kDynamic};
    SmallVector<int64_t> staticSizes{info->shape.begin(), info->shape.end()};
    SmallVector<int64_t> staticStrides{1, 1};
    Value subview = rewriter
                        .create<mlir::memref::SubViewOp>(
                            loc, subviewType, *baseMemref, ValueRange{(*coordPair)[0], (*coordPair)[1]},
                            ValueRange{}, ValueRange{}, staticOffsets, staticSizes, staticStrides)
                        .getResult();
    if (expected->getRank() == 2)
      return castMemrefToExpected(rewriter, loc, subview, *expected);

    SmallVector<ReassociationIndices> reassociation{{0, 1}};
    auto flatLayout =
        StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic, ArrayRef<int64_t>{1});
    auto flatType = MemRefType::get({*numElements}, baseType.getElementType(), flatLayout,
                                    baseType.getMemorySpace());
    Value flattened =
        rewriter.create<mlir::memref::CollapseShapeOp>(loc, flatType, subview, reassociation)
            .getResult();
    return castMemrefToExpected(rewriter, loc, flattened, *expected);
  }

  if (baseType.getRank() != 1)
    return failure();

  Value offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  if (!baseType.hasStaticShape() || baseType.getDimSize(0) != info->shape[0]) {
    if (info->shape.size() == 1) {
      auto dynamicOffset = materializeSingleCoordIndex(rewriter, loc, tileView.getCoord());
      if (failed(dynamicOffset))
        return failure();
      offset = *dynamicOffset;
    } else {
      if (llvm::any_of(info->strides,
                       [](int64_t stride) { return stride <= 0 || stride == ShapedType::kDynamic; }))
        return failure();
      auto coordPair = materializeRank2CoordIndices(rewriter, loc, tileView.getCoord());
      if (failed(coordPair))
        return failure();
      Value rowStride = rewriter.create<arith::ConstantIndexOp>(loc, info->strides[0]);
      Value colStride = rewriter.create<arith::ConstantIndexOp>(loc, info->strides[1]);
      Value rowOffset = rewriter.create<arith::MulIOp>(loc, (*coordPair)[0], rowStride);
      Value colOffset = rewriter.create<arith::MulIOp>(loc, (*coordPair)[1], colStride);
      offset = rewriter.create<arith::AddIOp>(loc, rowOffset, colOffset);
    }
  }

  if (expected->getRank() == 2) {
    SmallVector<Value> sizes;
    SmallVector<Value> strides;
    for (int64_t dim : info->shape)
      sizes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
    for (int64_t stride : info->strides)
      strides.push_back(rewriter.create<arith::ConstantIndexOp>(loc, stride));
    return rewriter
        .create<mlir::memref::ReinterpretCastOp>(loc, *expected, *baseMemref, offset, sizes,
                                                 strides)
        .getResult();
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

static FailureOr<Value> getMakeTensorLikeFlatMemref(Value tensor) {
  auto makeTensor = tensor.getDefiningOp<::tla::MakeTensorLikeOp>();
  if (!makeTensor)
    return failure();
  Value ptr = makeTensor.getPtr();
  if (auto bridge = ptr.getDefiningOp<::tla::HivmMemrefAsPtrOp>())
    return bridge.getMemref();
  if (auto ptrCast = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (ptrCast.getNumOperands() == 1 && isa<MemRefType>(ptrCast.getOperand(0).getType()))
      return ptrCast.getOperand(0);
  }
  return failure();
}

static FailureOr<Value>
materializeBaseMemref(PatternRewriter &rewriter, Location loc, Value tensor,
                      DenseMap<Value, Value> *handoffMemrefByTensor) {
  if (handoffMemrefByTensor) {
    auto handoff = handoffMemrefByTensor->find(tensor);
    if (handoff != handoffMemrefByTensor->end()) {
      auto expected = getBridgedTensorMemrefType(tensor);
      if (failed(expected))
        return failure();
      return castMemrefToExpected(rewriter, loc, handoff->second, *expected);
    }
  }

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
    if (succeeded(info) && (info->shape.size() == 1 || info->shape.size() == 2) &&
        succeeded(getVectorHelperMemrefType(tensor)))
      return materializeTileViewMemref(rewriter, loc, tensor, tileView,
                                       /*useVectorHelperType=*/true, handoffMemrefByTensor);
    if (isa<MemRefType>(source.getType())) {
      if (succeeded(info) && (info->shape.size() == 1 || info->shape.size() == 2))
        return materializeTileViewMemref(rewriter, loc, tensor, tileView,
                                         /*useVectorHelperType=*/false,
                                         handoffMemrefByTensor);
      auto expected = getBridgedTensorMemrefType(tensor);
      if (failed(expected))
        return failure();
      return castMemrefToExpected(rewriter, loc, source, *expected);
    }
    return materializeBaseMemref(rewriter, loc, source, handoffMemrefByTensor);
  }

  if (isa<MemRefType>(tensor.getType()))
    return tensor;

  return failure();
}

static FailureOr<Value> materializeVectorHelperBaseMemref(PatternRewriter &rewriter, Location loc,
                                                          Value tensor,
                                                          DenseMap<Value, Value>
                                                              *handoffMemrefByTensor) {
  auto expected = getVectorHelperMemrefType(tensor, handoffMemrefByTensor);
  if (failed(expected))
    return failure();

  if (handoffMemrefByTensor) {
    auto handoff = handoffMemrefByTensor->find(tensor);
    if (handoff != handoffMemrefByTensor->end()) {
      if (tensor.getDefiningOp<::tla::MakeTensorLikeOp>()) {
        if (auto handoffType = dyn_cast<MemRefType>(handoff->second.getType())) {
          if (handoffType.getRank() == 2 && handoffType.hasStaticShape()) {
            auto localSubview = materializeCopySubviewRank2(
                rewriter, loc, tensor, /*handoffMemrefByTensor=*/nullptr, handoffType.getShape());
            if (succeeded(localSubview))
              return castMemrefToExpected(rewriter, loc, *localSubview, *expected);
          }
        }
      }
      return castMemrefToExpected(rewriter, loc, handoff->second, *expected);
    }
  }

  if (auto flatMemref = getMakeTensorLikeFlatMemref(tensor); succeeded(flatMemref))
    return castMemrefToExpected(rewriter, loc, *flatMemref, *expected);

  auto base = materializeBaseMemref(rewriter, loc, tensor, handoffMemrefByTensor);
  if (failed(base))
    return failure();
  return castMemrefToExpected(rewriter, loc, *base, *expected);
}

static FailureOr<Value> materializeCopySubview1D(PatternRewriter &rewriter, Location loc,
                                                 Value tensor,
                                                 DenseMap<Value, Value> *handoffMemrefByTensor =
                                                     nullptr) {
  auto info = parseTensorInfo(tensor.getType());
  if (failed(info))
    return failure();
  if (info->shape.size() != 1 || info->coord.size() != 1 || !info->elementType.isF32() ||
      info->layoutTag != "row_major")
    return failure();
  if (info->shape[0] != 64 || info->coord[0] != 0)
    return failure();

  auto baseMemref = materializeBaseMemref(rewriter, loc, tensor, handoffMemrefByTensor);
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

static FailureOr<Value> materializeCopySubviewRank2(
    PatternRewriter &rewriter, Location loc, Value tensor,
    DenseMap<Value, Value> *handoffMemrefByTensor, ArrayRef<int64_t> concreteShape) {
  auto info = parseTensorInfo(tensor.getType());
  if (failed(info))
    return failure();
  if (info->shape.size() != 2 || info->coord.size() != 2 || info->strides.size() != 2 ||
      info->layoutTag != "row_major")
    return failure();
  SmallVector<int64_t, 2> shape(info->shape.begin(), info->shape.end());
  if (!concreteShape.empty()) {
    if (concreteShape.size() != 2)
      return failure();
    shape.assign(concreteShape.begin(), concreteShape.end());
  }
  if (shape[0] <= 0 || shape[0] == ShapedType::kDynamic ||
      shape[1] <= 0 || shape[1] == ShapedType::kDynamic)
    return failure();

  MLIRContext *ctx = rewriter.getContext();
  Value rows = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
  Value cols = rewriter.create<arith::ConstantIndexOp>(loc, shape[1]);
  SmallVector<int64_t> staticOffsets{ShapedType::kDynamic, ShapedType::kDynamic};
  SmallVector<int64_t> staticSizes{shape.begin(), shape.end()};
  SmallVector<int64_t> staticStrides{1, 1};

  if (succeeded(getMakeTensorLikeFlatMemref(tensor))) {
    auto flatMemref = *getMakeTensorLikeFlatMemref(tensor);
    auto flatType = dyn_cast<MemRefType>(flatMemref.getType());
    if (!flatType || flatType.getRank() != 1)
      return failure();
    auto expandedType = MemRefType::get({shape[0], shape[1]},
                                        flatType.getElementType(), MemRefLayoutAttrInterface{},
                                        flatType.getMemorySpace());
    auto expanded = rewriter
                        .create<mlir::memref::ExpandShapeOp>(
                            loc, expandedType, flatMemref,
                            ArrayRef<ReassociationIndices>{ReassociationIndices{0, 1}})
                        .getResult();
    auto subviewLayout =
        StridedLayoutAttr::get(ctx, ShapedType::kDynamic, ArrayRef<int64_t>{shape[1], 1});
    auto subviewType =
        MemRefType::get(shape, flatType.getElementType(), subviewLayout,
                        flatType.getMemorySpace());
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto subview = rewriter
        .create<mlir::memref::SubViewOp>(loc, subviewType, expanded, ValueRange{zero, zero},
                                         ValueRange{}, ValueRange{}, staticOffsets, staticSizes,
                                         staticStrides)
        .getResult();
    if (handoffMemrefByTensor)
      (*handoffMemrefByTensor)[tensor] = subview;
    return subview;
  }

  auto tileView = tensor.getDefiningOp<::tla::TileViewOp>();
  if (!tileView)
    return failure();
  Value source = tileView.getSource();
  auto baseMemref = materializeBaseMemref(rewriter, loc, source, handoffMemrefByTensor);
  if (failed(baseMemref))
    return failure();
  auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
  if (!baseType || baseType.getRank() != 2)
    return failure();

  auto coordPair = materializeRank2CoordIndices(rewriter, loc, tileView.getCoord());
  if (failed(coordPair))
    return failure();

  int64_t rowStride = baseType.hasStaticShape() ? baseType.getDimSize(1) : info->strides[0];
  if (rowStride <= 0 || rowStride == ShapedType::kDynamic)
    rowStride = ShapedType::kDynamic;
  auto subviewLayout =
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic, ArrayRef<int64_t>{rowStride, 1});
  auto subviewType = MemRefType::get(shape, baseType.getElementType(), subviewLayout,
                                     baseType.getMemorySpace());
  (void)rows;
  (void)cols;
  return rewriter
      .create<mlir::memref::SubViewOp>(loc, subviewType, *baseMemref, ValueRange{(*coordPair)[0], (*coordPair)[1]},
                                       ValueRange{}, ValueRange{}, staticOffsets, staticSizes,
                                       staticStrides)
      .getResult();
}

static FailureOr<Value> materializeCopySubview(PatternRewriter &rewriter, Location loc,
                                               Value tensor,
                                               DenseMap<Value, Value> *handoffMemrefByTensor =
                                                   nullptr,
                                               ArrayRef<int64_t> concreteShape = {}) {
  if (auto subview = materializeCopySubview1D(rewriter, loc, tensor, handoffMemrefByTensor);
      succeeded(subview))
    return subview;
  return materializeCopySubviewRank2(rewriter, loc, tensor, handoffMemrefByTensor,
                                     concreteShape);
}

static std::string buildUniqueVectorHelperName(ModuleOp module, int &nextVectorRegionId) {
  std::string helperName;
  do {
    helperName = "vector_region_" + std::to_string(nextVectorRegionId++);
  } while (module.lookupSymbol<func::FuncOp>(helperName));
  return helperName;
}

enum class VectorBinaryKind { Add, Sub, Mul, Div };

struct VectorBinaryInfo {
  VectorBinaryKind kind;
  StringRef name;
};

static std::optional<VectorBinaryInfo> getVectorBinaryInfo(Operation *op) {
  if (!op)
    return std::nullopt;
  if (isa<::tla::AddOp>(op))
    return VectorBinaryInfo{VectorBinaryKind::Add, "add"};
  if (isa<::tla::SubOp>(op))
    return VectorBinaryInfo{VectorBinaryKind::Sub, "sub"};
  if (isa<::tla::MulOp>(op))
    return VectorBinaryInfo{VectorBinaryKind::Mul, "mul"};
  if (isa<::tla::DivOp>(op))
    return VectorBinaryInfo{VectorBinaryKind::Div, "div"};
  return std::nullopt;
}

// Build the AVE vector op for a tla binary op. The mask is the all-true
// predicate; pass-thru is omitted. For div the signedness is carried as the
// TypeFn cast attribute (cast_unsigned for unsigned integer element types,
// cast_signed otherwise).
static Value createVectorBinaryResult(OpBuilder &b, Location loc, VectorBinaryKind kind,
                                      Type elementType, VectorType vecType, Value lhs,
                                      Value rhs, Value mask) {
  switch (kind) {
  case VectorBinaryKind::Add:
    return b.create<hivmave::VFAddOp>(loc, vecType, lhs, rhs, mask, /*pass_thru=*/nullptr);
  case VectorBinaryKind::Sub:
    return b.create<hivmave::VFSubOp>(loc, vecType, lhs, rhs, mask, /*pass_thru=*/nullptr);
  case VectorBinaryKind::Mul:
    return b.create<hivmave::VFMulOp>(loc, vecType, lhs, rhs, mask, /*pass_thru=*/nullptr);
  case VectorBinaryKind::Div: {
    auto cast = hivm::TypeFn::cast_signed;
    if (auto intType = dyn_cast<IntegerType>(elementType))
      if (intType.getSignedness() == IntegerType::Unsigned)
        cast = hivm::TypeFn::cast_unsigned;
    return b.create<hivmave::VFDivOp>(loc, vecType, lhs, rhs, mask,
                                      hivm::TypeFnAttr::get(b.getContext(), cast),
                                      /*pass_thru=*/nullptr);
  }
  }
  return nullptr;
}

static FailureOr<func::FuncOp> buildHelperFunc(ModuleOp module, func::FuncOp parentFunc,
                                               Operation *vectorOp, VectorBinaryKind binaryKind,
                                               StringRef binaryName, Value dst, Value lhs,
                                               Value rhs, int &nextVectorRegionId,
                                               DenseMap<Value, Value> &handoffMemrefByTensor,
                                               llvm::StringMap<func::FuncOp> &helperCache) {
  MLIRContext *ctx = module.getContext();
  OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPointAfter(parentFunc);

  auto dstType = getVectorHelperMemrefType(dst, &handoffMemrefByTensor);
  auto lhsType = getVectorHelperMemrefType(lhs, &handoffMemrefByTensor);
  auto rhsType = getVectorHelperMemrefType(rhs, &handoffMemrefByTensor);
  if (failed(dstType) || failed(lhsType) || failed(rhsType))
    return failure();

  auto funcType =
      moduleBuilder.getFunctionType(TypeRange{*lhsType, *rhsType, *dstType}, TypeRange{});

  std::string cacheKey;
  llvm::raw_string_ostream cacheKeyStream(cacheKey);
  cacheKeyStream << binaryName << ":";
  funcType.print(cacheKeyStream);
  cacheKeyStream.flush();
  auto cached = helperCache.find(cacheKey);
  if (cached != helperCache.end())
    return cached->getValue();

  std::string helperName = buildUniqueVectorHelperName(module, nextVectorRegionId);
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
  Value lhsVec = b.create<hivmave::VFLoadOp>(
      vectorOp->getLoc(), vecType, entry->getArgument(0), ValueRange{zero})
                     .getRes();
  Value rhsVec = b.create<hivmave::VFLoadOp>(
      vectorOp->getLoc(), vecType, entry->getArgument(1), ValueRange{zero})
                     .getRes();
  Type elementType = vecType.getElementType();
  if (!isa<IntegerType>(elementType) && !isa<FloatType>(elementType)) {
    return vectorOp->emitError("unsupported element type for vector binary helper: ")
           << elementType;
  }
  auto maskVecType = VectorType::get({*lanes}, b.getI1Type());
  Value opMask =
      b.create<hivmave::VFPgeOp>(vectorOp->getLoc(), maskVecType, hivmave::PgePattern::ALL);
  Value result = createVectorBinaryResult(b, vectorOp->getLoc(), binaryKind, elementType, vecType,
                                          lhsVec, rhsVec, opMask);
  if (!result)
    return vectorOp->emitError("unsupported binary op for element type: ") << elementType;
  Value storeMask =
      b.create<hivmave::VFPgeOp>(vectorOp->getLoc(), maskVecType, hivmave::PgePattern::ALL);
  b.create<hivmave::VFMaskedStoreOp>(vectorOp->getLoc(), entry->getArgument(2),
                                     ValueRange{zero}, storeMask, result);
  b.create<func::ReturnOp>(vectorOp->getLoc());

  helperCache[cacheKey] = helper;
  return helper;
}

class LowerVecFuncRegionPattern : public OpRewritePattern<::tla::VecFuncOp> {
public:
  LowerVecFuncRegionPattern(MLIRContext *context, ModuleOp module, int &nextVectorRegionId,
                            DenseMap<Value, Value> &handoffMemrefByTensor,
                            llvm::StringMap<func::FuncOp> &helperCache)
      : OpRewritePattern<::tla::VecFuncOp>(context, /*benefit=*/2), module(module),
        nextVectorRegionId(nextVectorRegionId), handoffMemrefByTensor(handoffMemrefByTensor),
        helperCache(helperCache) {}

  LogicalResult matchAndRewrite(::tla::VecFuncOp vecFuncOp,
                                PatternRewriter &rewriter) const override {
    auto *body = vecFuncOp.getBody().empty() ? nullptr : &vecFuncOp.getBody().front();
    if (!body) {
      return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.vec.func body");
    }

    ::tla::StoreOp store;
    vecFuncOp->walk([&](::tla::StoreOp candidate) {
        if (store) {
          return WalkResult::interrupt();
        }
        store = candidate;
        return WalkResult::advance();
    });
    int storeCount = 0;
    vecFuncOp->walk([&](::tla::StoreOp) { ++storeCount; });
    if (storeCount > 1) {
      return rewriter.notifyMatchFailure(vecFuncOp,
                                         "expected exactly one tla.store in tla.vec.func body");
    }
    if (!store)
      return rewriter.notifyMatchFailure(vecFuncOp,
                                         "expected tla.vec.func body to contain tla.store");

    Operation *binaryOp = store.getSource().getDefiningOp();
    auto binaryInfo = getVectorBinaryInfo(binaryOp);
    if (!binaryInfo || binaryOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(
          vecFuncOp, "expected tla.store source to be a supported tla binary op");

    auto lhsLoad = binaryOp->getOperand(0).getDefiningOp<::tla::LoadOp>();
    auto rhsLoad = binaryOp->getOperand(1).getDefiningOp<::tla::LoadOp>();
    if (!lhsLoad || !rhsLoad)
      return rewriter.notifyMatchFailure(vecFuncOp,
                                         "expected tla binary op operands to be tla.load");

    auto funcOp = vecFuncOp->getParentOfType<func::FuncOp>();
    if (!funcOp)
      return rewriter.notifyMatchFailure(vecFuncOp, "expected enclosing func.func");

    Value dst = store.getDest();
    Value lhs = lhsLoad.getSource();
    Value rhs = rhsLoad.getSource();

    auto helperOr =
        buildHelperFunc(module, funcOp, vecFuncOp.getOperation(), binaryInfo->kind,
                        binaryInfo->name, dst, lhs, rhs, nextVectorRegionId,
                        handoffMemrefByTensor, helperCache);
    if (failed(helperOr)) {
      return rewriter.notifyMatchFailure(vecFuncOp, "failed to build vector helper function");
    }
    auto helper = *helperOr;

    rewriter.setInsertionPoint(store);
    auto lhsBase =
        materializeVectorHelperBaseMemref(rewriter, vecFuncOp.getLoc(), lhs,
                                          &handoffMemrefByTensor);
    auto rhsBase =
        materializeVectorHelperBaseMemref(rewriter, vecFuncOp.getLoc(), rhs,
                                          &handoffMemrefByTensor);
    auto dstBase =
        materializeVectorHelperBaseMemref(rewriter, vecFuncOp.getLoc(), dst,
                                          &handoffMemrefByTensor);
    if (failed(lhsBase) || failed(rhsBase) || failed(dstBase)) {
      return rewriter.notifyMatchFailure(vecFuncOp,
                                         "failed to materialize UB memrefs for vector helper call");
    }

    auto lhsType = getVectorHelperMemrefType(lhs, &handoffMemrefByTensor);
    auto rhsType = getVectorHelperMemrefType(rhs, &handoffMemrefByTensor);
    auto dstType = getVectorHelperMemrefType(dst, &handoffMemrefByTensor);
    if (failed(lhsType) || failed(rhsType) || failed(dstType)) {
      return rewriter.notifyMatchFailure(vecFuncOp, "failed to derive helper memref signature");
    }
    auto lhsArg = castMemrefToExpected(rewriter, vecFuncOp.getLoc(), *lhsBase, *lhsType);
    auto rhsArg = castMemrefToExpected(rewriter, vecFuncOp.getLoc(), *rhsBase, *rhsType);
    auto dstArg = castMemrefToExpected(rewriter, vecFuncOp.getLoc(), *dstBase, *dstType);
    if (failed(lhsArg) || failed(rhsArg) || failed(dstArg)) {
      return rewriter.notifyMatchFailure(vecFuncOp,
                                         "failed to cast helper operands to expected memref types");
    }

    auto call = rewriter.create<func::CallOp>(vecFuncOp.getLoc(), helper,
                                              ValueRange{*lhsArg, *rhsArg, *dstArg});
    call->setAttr("hivm.vector_function", UnitAttr::get(rewriter.getContext()));
    call->setAttr("no_inline", UnitAttr::get(rewriter.getContext()));
    rewriter.eraseOp(store);
    rewriter.eraseOp(binaryOp);
    rewriter.eraseOp(lhsLoad);
    rewriter.eraseOp(rhsLoad);
    rewriter.inlineBlockBefore(body, vecFuncOp->getBlock(), vecFuncOp->getIterator());
    rewriter.eraseOp(vecFuncOp);
    return success();
  }

private:
  ModuleOp module;
  int &nextVectorRegionId;
  DenseMap<Value, Value> &handoffMemrefByTensor;
  llvm::StringMap<func::FuncOp> &helperCache;
};

class LowerCopyPattern : public OpRewritePattern<::tla::CopyOp> {
public:
  explicit LowerCopyPattern(MLIRContext *context, DenseMap<Value, Value> &handoffMemrefByTensor)
      : OpRewritePattern<::tla::CopyOp>(context, /*benefit=*/3),
        handoffMemrefByTensor(handoffMemrefByTensor) {}

  LogicalResult matchAndRewrite(::tla::CopyOp copyOp, PatternRewriter &rewriter) const override {
    auto dstInfo = parseTensorInfo(copyOp.getDst().getType());
    auto srcInfo = parseTensorInfo(copyOp.getSrc().getType());
    if (failed(dstInfo) || failed(srcInfo))
      return failure();

    bool isGmToUb =
        srcInfo->addressSpace == AddressSpace::gm && dstInfo->addressSpace == AddressSpace::ub;
    bool isUbToGm =
        srcInfo->addressSpace == AddressSpace::ub && dstInfo->addressSpace == AddressSpace::gm;
    if (!isGmToUb && !isUbToGm)
      return failure();

    ArrayRef<int64_t> srcShapeHint = {};
    ArrayRef<int64_t> dstShapeHint = {};
    if (isGmToUb)
      dstShapeHint = srcInfo->shape;
    if (isUbToGm)
      srcShapeHint = dstInfo->shape;
    auto srcSubview =
        materializeCopySubview(rewriter, copyOp.getLoc(), copyOp.getSrc(),
                               &handoffMemrefByTensor, srcShapeHint);
    auto dstSubview =
        materializeCopySubview(rewriter, copyOp.getLoc(), copyOp.getDst(),
                               &handoffMemrefByTensor, dstShapeHint);
    if (failed(srcSubview) || failed(dstSubview))
      return failure();

    if (isGmToUb) {
      auto padModeAttr = rewriter.getAttr<hivm::PadModeAttr>(hivm::PadMode::PadValue);
      Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(copyOp.getLoc(), 0);
      auto dstMemrefType = dyn_cast<MemRefType>((*dstSubview).getType());
      if (!dstMemrefType)
        return failure();
      auto zeroValue = createZeroValue(rewriter, copyOp.getLoc(), dstMemrefType.getElementType());
      if (failed(zeroValue))
        return failure();
      auto load = rewriter.create<hivm::LoadOp>(copyOp.getLoc(), TypeRange{}, *srcSubview,
                                                *dstSubview, padModeAttr, *zeroValue, zeroIndex);
      load->removeAttr("init_out_buffer");
      load->removeAttr("may_implicit_transpose_with_last_axis");
      handoffMemrefByTensor[copyOp.getDst()] = *dstSubview;
      rewriter.eraseOp(copyOp);
      return success();
    }

    if (isUbToGm) {
      handoffMemrefByTensor[copyOp.getSrc()] = *srcSubview;
      if (auto dstSubviewOp = (*dstSubview).getDefiningOp<mlir::memref::SubViewOp>())
        dstSubviewOp->setAttr("to_be_bubbled_slice", UnitAttr::get(rewriter.getContext()));
      auto store =
          rewriter.create<hivm::StoreOp>(copyOp.getLoc(), TypeRange{}, *srcSubview, *dstSubview);
      if (srcInfo->shape.size() == 2)
        store->setAttr("tiled_op", UnitAttr::get(rewriter.getContext()));
      rewriter.eraseOp(copyOp);
      return success();
    }

    return failure();
  }

private:
  DenseMap<Value, Value> &handoffMemrefByTensor;
};

class InlineVectorRegionWrapperPattern : public OpRewritePattern<::tla::VectorOp> {
public:
  explicit InlineVectorRegionWrapperPattern(MLIRContext *context)
      : OpRewritePattern<::tla::VectorOp>(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(::tla::VectorOp vectorOp,
                                PatternRewriter &rewriter) const override {
    auto role = vectorOp->getAttrOfType<StringAttr>("tla.vector_role");
    if (!role || role.getValue() != "region")
      return failure();
    auto *body = vectorOp.getBody().empty() ? nullptr : &vectorOp.getBody().front();
    if (!body)
      return failure();
    rewriter.inlineBlockBefore(body, vectorOp->getBlock(), vectorOp->getIterator());
    rewriter.eraseOp(vectorOp);
    return success();
  }
};

static void inlineVectorRegionWrappers(func::FuncOp funcOp) {
  SmallVector<::tla::VectorOp, 4> wrappers;
  funcOp.walk([&](::tla::VectorOp vectorOp) {
    auto role = vectorOp->getAttrOfType<StringAttr>("tla.vector_role");
    if (role && role.getValue() == "region")
      wrappers.push_back(vectorOp);
  });

  IRRewriter rewriter(funcOp.getContext());
  for (::tla::VectorOp vectorOp : wrappers) {
    if (!vectorOp || vectorOp.getBody().empty())
      continue;
    Block *body = &vectorOp.getBody().front();
    rewriter.inlineBlockBefore(body, vectorOp->getBlock(), vectorOp->getIterator());
    rewriter.eraseOp(vectorOp);
  }
}

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
                                        int &nextVectorRegionId,
                                        DenseMap<Value, Value> &handoffMemrefByTensor,
                                        llvm::StringMap<func::FuncOp> &helperCache) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InlineVectorRegionWrapperPattern>(ctx);
  patterns.add<LowerVecFuncRegionPattern>(ctx, module, nextVectorRegionId,
                                          handoffMemrefByTensor, helperCache);
  patterns.add<LowerCopyPattern>(ctx, handoffMemrefByTensor);
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
                    hivm::HIVMDialect, hivmave::AVEDialect, vector::VectorDialect,
                    ::tla::TlaDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr(kTlaHasVectorRegionAttrName))
      return;

    nextVectorRegionId = 0;
    llvm::StringMap<func::FuncOp> helperCache;

    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.isDeclaration())
        continue;
      inlineVectorRegionWrappers(funcOp);
      DenseMap<Value, Value> handoffMemrefByTensor;
      RewritePatternSet patterns(&getContext());
      populateTlaToVectorPatterns(patterns, module, nextVectorRegionId, handoffMemrefByTensor,
                                  helperCache);
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
