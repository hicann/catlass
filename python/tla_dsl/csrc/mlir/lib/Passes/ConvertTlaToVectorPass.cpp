#include "PassesCommon.h"
#include "PassesInternal.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
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

static FailureOr<int64_t> getVectorLaneCount(Type elementType) {
  auto elementBytes = getElementByteWidth(elementType);
  if (failed(elementBytes) || *elementBytes <= 0)
    return failure();
  constexpr int64_t kVectorBytes = 256;
  return kVectorBytes / *elementBytes;
}

static FailureOr<int64_t> getVectorLanesForMemref(MemRefType type) {
  if (type.getRank() != 1 && type.getRank() != 2)
    return failure();
  auto numElements = getStaticNumElements(type.getShape());
  auto lanesOr = getVectorLaneCount(type.getElementType());
  if (failed(numElements) || failed(lanesOr) || *numElements <= 0)
    return failure();
  int64_t lanes = *lanesOr;
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

// The full UB tensor that a tile_view chunk views into. tla.load/tla.store
// operate on per-iteration chunk tile_views; the helper argument is the whole
// tensor those chunks come from.
static Value getFullTensorOf(Value tile) {
  while (auto tileView = tile.getDefiningOp<::tla::TileViewOp>())
    tile = tileView.getSource();
  return tile;
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
  Value ptr;
  if (auto makeTensorLike = tensor.getDefiningOp<::tla::MakeTensorLikeOp>())
    ptr = makeTensorLike.getPtr();
  else if (auto makeTensor = tensor.getDefiningOp<::tla::MakeTensorOp>())
    ptr = makeTensor.getPtr();
  else
    return failure();
  if (auto bridge = ptr.getDefiningOp<::tla::HivmMemrefAsPtrOp>())
    return bridge.getMemref();
  if (auto ptrCast = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (ptrCast.getNumOperands() == 1 && isa<MemRefType>(ptrCast.getOperand(0).getType()))
      return ptrCast.getOperand(0);
  }
  return failure();
}

static FailureOr<MemRefType> getVectorHelperArgMemrefType(Value operand) {
  if (auto flat = getMakeTensorLikeFlatMemref(operand); succeeded(flat)) {
    if (auto flatType = dyn_cast<MemRefType>((*flat).getType());
        flatType && flatType.getRank() == 1 && flatType.hasStaticShape())
      return flatType;
  }
  auto bridged = getBridgedTensorMemrefType(operand);
  if (failed(bridged) || bridged->getRank() != 1)
    return failure();
  return *bridged;
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

  if (auto makeTensor = tensor.getDefiningOp<::tla::MakeTensorOp>()) {
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

static FailureOr<Value> materializeCopySubview1D(PatternRewriter &rewriter, Location loc,
                                                 Value tensor,
                                                 DenseMap<Value, Value> *handoffMemrefByTensor =
                                                     nullptr) {
  auto info = parseTensorInfo(tensor.getType());
  if (failed(info))
    return failure();
  if (info->shape.size() != 1 || info->coord.size() != 1 || info->layoutTag != "row_major")
    return failure();
  if (info->shape[0] == ShapedType::kDynamic || info->coord[0] == ShapedType::kDynamic)
    return failure();

  auto baseMemref = materializeBaseMemref(rewriter, loc, tensor, handoffMemrefByTensor);
  if (failed(baseMemref))
    return failure();
  auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
  if (!baseType || baseType.getRank() != 1)
    return failure();

  auto ctx = rewriter.getContext();
  int64_t subviewOffset = baseType.getDimSize(0) == info->shape[0] ? 0 : info->coord[0];
  if (subviewOffset == 0 && baseType.hasStaticShape() && baseType.getDimSize(0) == info->shape[0])
    return *baseMemref;

  auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, ArrayRef<int64_t>{1});
  auto subviewType =
      MemRefType::get({info->shape[0]}, baseType.getElementType(), layout, baseType.getMemorySpace());
  Value offset = rewriter.create<arith::ConstantIndexOp>(loc, subviewOffset);
  Value size = rewriter.create<arith::ConstantIndexOp>(loc, info->shape[0]);
  Value stride = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  return rewriter
      .create<mlir::memref::ReinterpretCastOp>(loc, subviewType, *baseMemref, offset,
                                               ValueRange{size}, ValueRange{stride})
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

enum class VectorBinaryKind { Add, Sub, Mul, Div, Max, Min };
enum class VectorRhsKind { Vector, Scalar };

// The lhs/rhs/mask operands of a tla binary op.
struct TlaBinaryOperands {
  Value lhs;
  Value rhs;
  Value mask;
};

static TlaBinaryOperands getTlaBinaryOperands(Operation *op) {
  TlaBinaryOperands r{};
  if (auto o = dyn_cast<::tla::AddOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::SubOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::MulOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::DivOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::MaxOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::MinOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::AddsOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::SubsOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::MulsOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::MaxsOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::MinsOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  } else if (auto o = dyn_cast<::tla::DivsOp>(op)) {
    r.lhs = o.getLhs(); r.rhs = o.getRhs(); r.mask = o.getMask();
  }
  return r;
}

struct VectorOpInfo {
  VectorBinaryKind kind;
  VectorRhsKind rhsKind;
  StringRef mnemonic;
  TlaBinaryOperands operands;
};

static std::optional<VectorOpInfo> getVectorBinaryInfo(Operation *op) {
  if (!op)
    return std::nullopt;
  if (isa<::tla::AddOp>(op))
    return VectorOpInfo{VectorBinaryKind::Add, VectorRhsKind::Vector, "add",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::SubOp>(op))
    return VectorOpInfo{VectorBinaryKind::Sub, VectorRhsKind::Vector, "sub",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::MulOp>(op))
    return VectorOpInfo{VectorBinaryKind::Mul, VectorRhsKind::Vector, "mul",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::DivOp>(op))
    return VectorOpInfo{VectorBinaryKind::Div, VectorRhsKind::Vector, "div",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::MaxOp>(op))
    return VectorOpInfo{VectorBinaryKind::Max, VectorRhsKind::Vector, "max",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::MinOp>(op))
    return VectorOpInfo{VectorBinaryKind::Min, VectorRhsKind::Vector, "min",
                        getTlaBinaryOperands(op)};
  return std::nullopt;
}

static std::optional<VectorOpInfo> getVectorScalarBinaryInfo(Operation *op) {
  if (!op)
    return std::nullopt;
  if (isa<::tla::AddsOp>(op))
    return VectorOpInfo{VectorBinaryKind::Add, VectorRhsKind::Scalar, "adds",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::SubsOp>(op))
    return VectorOpInfo{VectorBinaryKind::Sub, VectorRhsKind::Scalar, "subs",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::MulsOp>(op))
    return VectorOpInfo{VectorBinaryKind::Mul, VectorRhsKind::Scalar, "muls",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::MaxsOp>(op))
    return VectorOpInfo{VectorBinaryKind::Max, VectorRhsKind::Scalar, "maxs",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::MinsOp>(op))
    return VectorOpInfo{VectorBinaryKind::Min, VectorRhsKind::Scalar, "mins",
                        getTlaBinaryOperands(op)};
  if (isa<::tla::DivsOp>(op))
    return VectorOpInfo{VectorBinaryKind::Div, VectorRhsKind::Scalar, "divs",
                        getTlaBinaryOperands(op)};
  return std::nullopt;
}

static std::optional<VectorOpInfo> getAnyVectorOperationInfo(Operation *op) {
  if (auto info = getVectorBinaryInfo(op))
    return info;
  if (auto info = getVectorScalarBinaryInfo(op))
    return info;
  return std::nullopt;
}

// The predicate-register width (b8/b16/b32) matching the element type.
static hivmave::MaskWidth maskWidthForElement(Type elementType) {
  unsigned bits = elementType.getIntOrFloatBitWidth();
  if (bits <= 8)
    return hivmave::MaskWidth::B8;
  if (bits <= 16)
    return hivmave::MaskWidth::B16;
  return hivmave::MaskWidth::B32;
}

// True for the tla ops that produce a vector compute result inside a vec.func
// region: the element-wise binary ops and the where/select op.
static bool isVectorComputeOp(Operation *op) {
  return getAnyVectorOperationInfo(op).has_value() || isa_and_nonnull<::tla::WhereOp>(op);
}

// Build the AVE vector op for a tla binary op. The mask predicates active lanes.
// For div the signedness is carried as the TypeFn cast attribute.
static Value createVectorBinaryResult(OpBuilder &b, Location loc, VectorBinaryKind kind,
                                      Type elementType, VectorType vecType, Value lhs,
                                      Value rhs, Value mask) {
  switch (kind) {
  case VectorBinaryKind::Add:
    return b.create<hivmave::VFAddOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
  case VectorBinaryKind::Sub:
    return b.create<hivmave::VFSubOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
  case VectorBinaryKind::Mul:
    return b.create<hivmave::VFMulOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
  case VectorBinaryKind::Div: {
    auto cast = hivm::TypeFn::cast_signed;
    if (auto intType = dyn_cast<IntegerType>(elementType))
      if (intType.getSignedness() == IntegerType::Unsigned)
        cast = hivm::TypeFn::cast_unsigned;
    return b.create<hivmave::VFDivOp>(loc, vecType, lhs, rhs, mask,
                                      hivm::TypeFnAttr::get(b.getContext(), cast), Value())
        .getResult();
  }
  case VectorBinaryKind::Max:
    return b.create<hivmave::VFMaxOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
  case VectorBinaryKind::Min:
    return b.create<hivmave::VFMinOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
  }
  return nullptr;
}

// Shared state while re-creating a tla.vec.func body inside the helper function.
struct VecLowerCtx {
  int64_t lanes;
  Type elementType;
  VectorType vecType;
  VectorType maskVecType;
};

// Return the value already mapped into the helper, or clone an arith.constant
// on demand (loop bounds / index math constants are pulled in lazily this way).
static Value lookupOrCloneScalarValue(OpBuilder &b, Value value,
                                      DenseMap<Value, Value> &valueMap) {
  if (Value mapped = valueMap.lookup(value))
    return mapped;
  Operation *def = value.getDefiningOp();
  if (!def || def->getNumResults() != 1 || !isa<arith::ConstantOp>(def))
    return nullptr;
  Operation *cloned = b.clone(*def);
  valueMap[value] = cloned->getResult(0);
  return cloned->getResult(0);
}

static FailureOr<Value> castScalarForVectorElement(Value scalar, Type elementType) {
  if (scalar.getType() == elementType)
    return scalar;
  return failure();
}

static FailureOr<Value> materializeVectorScalarValue(OpBuilder &b, TlaBinaryOperands operands,
                                                     DenseMap<Value, Value> &valueMap,
                                                     VecLowerCtx &ctx) {
  Value scalar = lookupOrCloneScalarValue(b, operands.rhs, valueMap);
  if (!scalar)
    return failure();
  auto castScalar = castScalarForVectorElement(scalar, ctx.elementType);
  if (failed(castScalar))
    return failure();
  return *castScalar;
}

static FailureOr<Value> createVectorScalarBinaryResult(OpBuilder &b, Location loc,
                                                       VectorOpInfo info,
                                                       VecLowerCtx &ctx, Value lhs,
                                                       Value scalar, Value mask) {
  if (info.kind == VectorBinaryKind::Add || info.kind == VectorBinaryKind::Mul ||
      info.kind == VectorBinaryKind::Max || info.kind == VectorBinaryKind::Min) {
    if (info.kind == VectorBinaryKind::Add)
      return b.create<hivmave::VFAddsOp>(loc, ctx.vecType, lhs, scalar, mask, Value())
          .getResult();
    if (info.kind == VectorBinaryKind::Mul)
      return b.create<hivmave::VFMulsOp>(loc, ctx.vecType, lhs, scalar, mask, Value())
          .getResult();
    if (info.kind == VectorBinaryKind::Max)
      return b.create<hivmave::VFMaxsOp>(loc, ctx.vecType, lhs, scalar, mask, Value())
          .getResult();
    return b.create<hivmave::VFMinsOp>(loc, ctx.vecType, lhs, scalar, mask, Value())
        .getResult();
  }

  Value rhs = b.create<vector::BroadcastOp>(loc, ctx.vecType, scalar).getResult();
  return createVectorBinaryResult(b, loc, info.kind, ctx.elementType, ctx.vecType, lhs, rhs,
                                  mask);
}

// Per-iteration element offset of a tile_view chunk, expressed against the
// helper's (cloned) index arithmetic.
static FailureOr<Value> materializeCoordOffsetInHelper(OpBuilder &b, Location loc, Value coord,
                                                       DenseMap<Value, Value> &valueMap) {
  auto coordType = dyn_cast<::tla::CoordType>(coord.getType());
  if (!coordType)
    return failure();
  SmallVector<int64_t, 1> leaves;
  if (failed(::tla::getTlaIndexTreeLeaves(coordType.getTree(), leaves)) || leaves.size() != 1)
    return failure();
  if (leaves[0] != ShapedType::kDynamic)
    return b.create<arith::ConstantIndexOp>(loc, leaves[0]).getResult();

  auto makeCoord = coord.getDefiningOp<::tla::MakeCoordOp>();
  if (!makeCoord || makeCoord.getDynElems().size() != 1)
    return failure();
  Value mapped = lookupOrCloneScalarValue(b, *makeCoord.getDynElems().begin(), valueMap);
  if (!mapped || !mapped.getType().isIndex())
    return failure();
  return mapped;
}

// Lower a tla.tile_view inside the helper to a 256-byte (lanes-wide) tile of the
// mapped full-size helper argument, at the chunk's per-iteration element offset.
static FailureOr<Value> lowerTileViewInHelper(OpBuilder &b, Location loc,
                                              ::tla::TileViewOp tileView,
                                              DenseMap<Value, Value> &valueMap, int64_t lanes) {
  Value source = valueMap.lookup(getFullTensorOf(tileView.getSource()));
  if (!source)
    source = valueMap.lookup(tileView.getSource());
  if (!source)
    return tileView.emitError("failed to map tla.tile_view source in vector helper"), failure();
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  if (!sourceType || sourceType.getRank() != 1)
    return tileView.emitError("expected rank-1 memref source for vector tile_view"), failure();
  auto offset = materializeCoordOffsetInHelper(b, loc, tileView.getCoord(), valueMap);
  if (failed(offset))
    return tileView.emitError("failed to materialize tile_view coordinate"), failure();
  auto layout =
      StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, ArrayRef<int64_t>{1});
  auto tileType =
      MemRefType::get({lanes}, sourceType.getElementType(), layout, sourceType.getMemorySpace());
  Value size = b.create<arith::ConstantIndexOp>(loc, lanes);
  Value stride = b.create<arith::ConstantIndexOp>(loc, 1);
  return b
      .create<mlir::memref::ReinterpretCastOp>(loc, tileType, source, *offset, ValueRange{size},
                                               ValueRange{stride})
      .getResult();
}

static LogicalResult lowerNestedVectorBlock(Block *sourceBlock, OpBuilder &b,
                                            DenseMap<Value, Value> &valueMap, VecLowerCtx &ctx);

// Re-create one vec.func body op inside the helper: tla ops become AVE vector
// ops; scf control flow and index arithmetic are carried verbatim.
static LogicalResult lowerNestedVectorOp(Operation &op, OpBuilder &b,
                                         DenseMap<Value, Value> &valueMap, VecLowerCtx &ctx) {
  Location loc = op.getLoc();

  // make_shape / make_coord feed only tile_view offsets (recomputed below); map
  // them to themselves so lookups succeed.
  if (isa<::tla::MakeShapeOp, ::tla::MakeCoordOp>(op)) {
    valueMap[op.getResult(0)] = op.getResult(0);
    return success();
  }

  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    valueMap[constant.getResult()] = b.clone(op)->getResult(0);
    return success();
  }

  if (auto tileView = dyn_cast<::tla::TileViewOp>(op)) {
    auto tile = lowerTileViewInHelper(b, loc, tileView, valueMap, ctx.lanes);
    if (failed(tile))
      return failure();
    valueMap[tileView.getResult()] = *tile;
    return success();
  }

  if (auto loadOp = dyn_cast<::tla::LoadOp>(op)) {
    Value source = valueMap.lookup(loadOp.getSource());
    if (!source)
      return failure();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    valueMap[loadOp.getResult()] =
        b.create<hivmave::VFLoadOp>(loc, ctx.vecType, source, ValueRange{zero}).getRes();
    return success();
  }

  if (auto fullOp = dyn_cast<::tla::FullOp>(op)) {
    Value source = lookupOrCloneScalarValue(b, fullOp.getValue(), valueMap);
    if (!source)
      return failure();
    if (source.getType() != ctx.elementType)
      return fullOp.emitError("tla.full scalar type ")
                 << source.getType() << " does not match vector element type "
                 << ctx.elementType,
             failure();
    valueMap[fullOp.getResult()] =
        b.create<hivmave::VFBroadcastScalarOp>(loc, ctx.vecType, source).getRes();
    return success();
  }

  if (auto info = getVectorBinaryInfo(&op)) {
    if (op.getNumResults() != 1)
      return failure();
    TlaBinaryOperands operands = info->operands;
    Value lhs = valueMap.lookup(operands.lhs);
    if (!lhs)
      return failure();
    Value rhs = valueMap.lookup(operands.rhs);
    if (!rhs)
      return failure();
    Value mask;
    if (operands.mask) {
      mask = valueMap.lookup(operands.mask);
      if (!mask)
        return failure();
    } else {
      mask = b.create<hivmave::VFPgeOp>(loc, ctx.maskVecType, hivmave::PgePattern::ALL);
    }
    Value result = createVectorBinaryResult(b, loc, info->kind, ctx.elementType, ctx.vecType,
                                            lhs, rhs, mask);
    if (!result)
      return failure();
    valueMap[op.getResult(0)] = result;
    return success();
  }

  if (auto info = getVectorScalarBinaryInfo(&op)) {
    if (op.getNumResults() != 1)
      return failure();
    TlaBinaryOperands operands = info->operands;
    Value lhs = valueMap.lookup(operands.lhs);
    if (!lhs)
      return failure();
    auto scalarOr = materializeVectorScalarValue(b, operands, valueMap, ctx);
    if (failed(scalarOr))
      return failure();
    Value mask;
    if (operands.mask) {
      mask = valueMap.lookup(operands.mask);
      if (!mask)
        return failure();
    } else {
      mask = b.create<hivmave::VFPgeOp>(loc, ctx.maskVecType, hivmave::PgePattern::ALL);
    }
    auto result = createVectorScalarBinaryResult(b, loc, *info, ctx, lhs, *scalarOr, mask);
    if (failed(result))
      return failure();
    valueMap[op.getResult(0)] = *result;
    return success();
  }

  // tla.where: per-lane select. The mask predicates which lanes take `x`; the
  // remaining lanes take `y`. Lowers to ave.hir.vsel(mask, x, y).
  if (auto whereOp = dyn_cast<::tla::WhereOp>(op)) {
    Value mask = valueMap.lookup(whereOp.getMask());
    Value x = valueMap.lookup(whereOp.getX());
    Value y = valueMap.lookup(whereOp.getY());
    if (!mask || !x || !y)
      return failure();
    valueMap[whereOp.getResult()] =
        b.create<hivmave::VFSelectOp>(loc, ctx.vecType, mask, x, y);
    return success();
  }

  // tla.create_mask: build the predicate vector for the enclosing region from a
  // fixed pattern -> ave.hir.pge<PATTERN>. The dtype attr fixes the lane count
  // (256 bytes / element size) and must match the vector region width.
  if (auto maskOp = dyn_cast<::tla::CreateMaskOp>(op)) {
    auto pattern = hivmave::symbolizePgePattern(maskOp.getPattern());
    if (!pattern)
      return maskOp.emitError("unknown tla.create_mask pattern: ") << maskOp.getPattern(),
             failure();
    auto laneCountOr = getVectorLaneCount(maskOp.getDtype());
    if (failed(laneCountOr))
      return maskOp.emitError("unsupported tla.create_mask dtype: ") << maskOp.getDtype(),
             failure();
    int64_t laneCount = *laneCountOr;
    if (laneCount != ctx.lanes)
      return maskOp.emitError("tla.create_mask dtype implies ")
                 << laneCount << " lanes, but the vector region is " << ctx.lanes
                 << " lanes wide",
             failure();
    valueMap[maskOp.getResult()] =
        b.create<hivmave::VFPgeOp>(loc, ctx.maskVecType, *pattern);
    return success();
  }

  // tla.update_mask: tail predicate + remaining count. Lowers to ave.hir.plt,
  // whose mask result drives masked stores and whose second result
  // (true_shape - lanes) is threaded back as the loop-carried tail counter.
  // The dtype attr fixes the lane count (256 bytes / element size) and must
  // match the enclosing vector region width.
  if (auto updateMaskOp = dyn_cast<::tla::UpdateMaskOp>(op)) {
    Value trueShape = valueMap.lookup(updateMaskOp.getTrueShape());
    if (!trueShape)
      return failure();
    auto laneCountOr = getVectorLaneCount(updateMaskOp.getDtype());
    if (failed(laneCountOr))
      return updateMaskOp.emitError("unsupported tla.update_mask dtype: ")
             << updateMaskOp.getDtype(), failure();
    int64_t laneCount = *laneCountOr;
    if (laneCount != ctx.lanes)
      return updateMaskOp.emitError("tla.update_mask dtype implies ")
                 << laneCount << " lanes, but the vector region is " << ctx.lanes
                 << " lanes wide",
             failure();
    auto plt = b.create<hivmave::VFPltOp>(loc, ctx.maskVecType, b.getIndexType(),
                                          trueShape);
    valueMap[updateMaskOp.getMask()] = plt.getRes();
    // new_true_shape = true_shape - lanes, which is exactly what plt computes.
    // We materialize it with index arithmetic rather than consuming plt's second
    // result: that result is i32 in hardware but typed index, so carrying it
    // through the loop would leave an unfoldable i32<->index unrealized cast.
    Value lanesValue = b.create<arith::ConstantIndexOp>(loc, ctx.lanes);
    valueMap[updateMaskOp.getNewTrueShape()] =
        b.create<arith::SubIOp>(loc, trueShape, lanesValue);
    return success();
  }

  if (auto storeOp = dyn_cast<::tla::StoreOp>(op)) {
    Value dest = valueMap.lookup(storeOp.getDest());
    Value source = valueMap.lookup(storeOp.getSource());
    if (!dest || !source)
      return failure();
    auto destType = dyn_cast<MemRefType>(dest.getType());
    auto sourceVectorType = dyn_cast<VectorType>(source.getType());
    if (!destType || destType.getRank() != 1 || !sourceVectorType)
      return failure();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value mask;
    if (storeOp.getMask()) {
      mask = valueMap.lookup(storeOp.getMask());
      if (!mask)
        return failure();
    } else {
      mask = b.create<hivmave::VFPgeOp>(loc, ctx.maskVecType, hivmave::PgePattern::ALL);
    }
    auto permutationMap =
        AffineMap::getMinorIdentityMap(destType.getRank(), sourceVectorType.getRank(),
                                       b.getContext());
    auto inBounds = b.getBoolArrayAttr({destType.getDimSize(0) == ctx.lanes});
    b.create<vector::TransferWriteOp>(loc, source, dest, ValueRange{zero},
                                      AffineMapAttr::get(permutationMap), mask, inBounds);
    return success();
  }

  // scf.for: rebuild the loop, including loop-carried iter_args, and lower its
  // body. Init args and the scf.yield operands are index/scalar SSA threaded
  // through the helper (e.g. the tail counter produced by tla.update_mask).
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    Value lb = lookupOrCloneScalarValue(b, forOp.getLowerBound(), valueMap);
    Value ub = lookupOrCloneScalarValue(b, forOp.getUpperBound(), valueMap);
    Value step = lookupOrCloneScalarValue(b, forOp.getStep(), valueMap);
    if (!lb || !ub || !step)
      return failure();
    // Loop-carried `index` values (e.g. the tla.update_mask tail counter) are
    // carried across the loop as i64 instead: after scf->cf lowering the
    // downstream index->iN conversion only rewrites the induction variable, so
    // an index iter_arg would leave dangling index<->iN unrealized casts on the
    // carried value that ReconcileUnrealizedCasts cannot fold across the cf
    // block boundary. Casting at the boundaries with arith.index_cast keeps the
    // carried value a plain integer that lowers cleanly.
    Type i64Ty = b.getIntegerType(64);
    auto regionIterArgs = forOp.getRegionIterArgs();
    SmallVector<bool> wasIndex(regionIterArgs.size(), false);
    SmallVector<Value> initArgs;
    for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs())) {
      Value mapped = lookupOrCloneScalarValue(b, init, valueMap);
      if (!mapped)
        return failure();
      if (isa<IndexType>(mapped.getType())) {
        wasIndex[idx] = true;
        mapped = b.create<arith::IndexCastOp>(loc, i64Ty, mapped);
      }
      initArgs.push_back(mapped);
    }
    LogicalResult bodyStatus = success();
    auto newFor = b.create<scf::ForOp>(
        loc, lb, ub, step, initArgs,
        [&](OpBuilder &nb, Location nloc, Value iv, ValueRange iterArgs) {
          DenseMap<Value, Value> nestedMap = valueMap;
          nestedMap[forOp.getInductionVar()] = iv;
          for (size_t i = 0; i < regionIterArgs.size(); ++i) {
            Value newArg = iterArgs[i];
            if (wasIndex[i])
              newArg = nb.create<arith::IndexCastOp>(nloc, nb.getIndexType(), newArg);
            nestedMap[regionIterArgs[i]] = newArg;
          }
          if (failed(lowerNestedVectorBlock(forOp.getBody(), nb, nestedMap, ctx))) {
            bodyStatus = failure();
            nb.create<scf::YieldOp>(nloc, iterArgs);
            return;
          }
          auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
          SmallVector<Value> yielded;
          for (auto [i, v] : llvm::enumerate(oldYield.getOperands())) {
            Value mapped = lookupOrCloneScalarValue(nb, v, nestedMap);
            if (!mapped) {
              bodyStatus = failure();
              break;
            }
            if (wasIndex[i] && isa<IndexType>(mapped.getType()))
              mapped = nb.create<arith::IndexCastOp>(nloc, i64Ty, mapped);
            yielded.push_back(mapped);
          }
          if (failed(bodyStatus)) {
            nb.create<scf::YieldOp>(nloc, iterArgs);
            return;
          }
          nb.create<scf::YieldOp>(nloc, yielded);
        });
    if (failed(bodyStatus))
      return failure();
    for (auto [i, oldRes] : llvm::enumerate(forOp.getResults())) {
      Value newRes = newFor.getResult(i);
      if (wasIndex[i] && !oldRes.use_empty())
        newRes = b.create<arith::IndexCastOp>(loc, b.getIndexType(), newRes);
      valueMap[oldRes] = newRes;
    }
    return success();
  }

  // scf.if: rebuild as a result-less conditional (carried results must be
  // unused) and lower both regions. The condition is already in the value map.
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    for (Value result : ifOp.getResults())
      if (!result.use_empty())
        return failure();
    Value cond = lookupOrCloneScalarValue(b, ifOp.getCondition(), valueMap);
    if (!cond)
      return failure();
    bool hasElse = !ifOp.getElseRegion().empty();
    auto newIf = b.create<scf::IfOp>(loc, TypeRange{}, cond, hasElse);
    DenseMap<Value, Value> thenMap = valueMap;
    OpBuilder tb(newIf.thenBlock()->getTerminator());
    if (failed(lowerNestedVectorBlock(ifOp.thenBlock(), tb, thenMap, ctx)))
      return failure();
    if (hasElse) {
      DenseMap<Value, Value> elseMap = valueMap;
      OpBuilder eb(newIf.elseBlock()->getTerminator());
      if (failed(lowerNestedVectorBlock(ifOp.elseBlock(), eb, elseMap, ctx)))
        return failure();
    }
    return success();
  }

  // Index/scalar arithmetic (arith.*) feeding offsets/conditions: clone with
  // mapped operands.
  if (op.getDialect()->getNamespace() == arith::ArithDialect::getDialectNamespace()) {
    IRMapping mapper;
    for (Value operand : op.getOperands()) {
      Value mapped = lookupOrCloneScalarValue(b, operand, valueMap);
      if (!mapped)
        return failure();
      mapper.map(operand, mapped);
    }
    Operation *cloned = b.clone(op, mapper);
    for (auto [oldResult, newResult] : llvm::zip(op.getResults(), cloned->getResults()))
      valueMap[oldResult] = newResult;
    return success();
  }

  if (op.hasTrait<OpTrait::IsTerminator>())
    return success();

  return failure();
}

static LogicalResult lowerNestedVectorBlock(Block *sourceBlock, OpBuilder &b,
                                            DenseMap<Value, Value> &valueMap, VecLowerCtx &ctx) {
  for (Operation &op : sourceBlock->getOperations()) {
    // Terminators are reproduced by the enclosing op (scf.for/scf.if) or by
    // buildHelperFunc's func.return.
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    if (failed(lowerNestedVectorOp(op, b, valueMap, ctx)))
      return failure();
  }
  return success();
}

// Collect, in body order, the unique full UB tensors that tla.load/tla.store
// chunks reference. These become the helper's arguments.
static void collectVectorHelperOperands(Block *block, SmallVectorImpl<Value> &operands) {
  for (Operation &op : block->getOperations()) {
    if (auto loadOp = dyn_cast<::tla::LoadOp>(op)) {
      Value root = getFullTensorOf(loadOp.getSource());
      if (!llvm::is_contained(operands, root))
        operands.push_back(root);
      continue;
    }
    if (auto storeOp = dyn_cast<::tla::StoreOp>(op)) {
      Value root = getFullTensorOf(storeOp.getDest());
      if (!llvm::is_contained(operands, root))
        operands.push_back(root);
      continue;
    }
    for (Region &region : op.getRegions())
      for (Block &nested : region)
        collectVectorHelperOperands(&nested, operands);
  }
}

// Collect unique scalar values used inside the region but defined outside it
// (e.g. a sub_block_idx/block_idx computed at the top of the kernel, or a
// vector-scalar RHS constant). Passing them into the helper avoids cloning float
// constants into vector helpers where vector.broadcast can fold to illegal
// vector arith.constant ops before the HIVMAVE conversion pipeline.
// They are passed in as trailing scalar arguments rather than recomputed inside
// the outlined vector function.
static void collectVectorHelperScalarOperands(::tla::VecFuncOp vecFuncOp,
                                              SmallVectorImpl<Value> &scalars) {
  vecFuncOp.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Type operandType = operand.getType();
      if (!operandType.isIntOrIndex() && !isa<FloatType>(operandType))
        continue;
      Region *defRegion = operand.getParentRegion();
      if (defRegion && !vecFuncOp.getBody().isAncestor(defRegion) &&
          !llvm::is_contained(scalars, operand))
        scalars.push_back(operand);
    }
  });
}

// Build a vector_region helper for a tla.vec.func body. The helper receives one
// full-size UB memref per referenced tensor; the for/if control flow is carried
// inside the helper, where each tla.load/store is lowered to an AVE
// vload/masked-store over a 256-byte tile carved from the full memref at the
// per-iteration offset.
static FailureOr<func::FuncOp> buildHelperFunc(ModuleOp module, func::FuncOp parentFunc,
                                               ::tla::VecFuncOp vecFuncOp,
                                               ArrayRef<Value> helperOperands,
                                               ArrayRef<Value> scalarOperands,
                                               int &nextVectorRegionId,
                                               DenseMap<Value, Value> &handoffMemrefByTensor) {
  MLIRContext *ctx = module.getContext();
  Operation *vectorOp = vecFuncOp.getOperation();
  OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPointAfter(parentFunc);

  Block *body = vecFuncOp.getBody().empty() ? nullptr : &vecFuncOp.getBody().front();
  if (!body || helperOperands.empty())
    return failure();

  SmallVector<Type> functionInputs;
  functionInputs.reserve(helperOperands.size());
  for (Value operand : helperOperands) {
    auto operandType = getVectorHelperArgMemrefType(operand);
    if (failed(operandType))
      return failure();
    functionInputs.push_back(*operandType);
  }
  // Trailing scalar args: scalars captured from outside the region.
  for (Value scalar : scalarOperands)
    functionInputs.push_back(scalar.getType());
  auto funcType = moduleBuilder.getFunctionType(functionInputs, TypeRange{});

  // The per-iteration vector tile is one 256-byte register's worth of elements.
  // A single VecLowerCtx (lanes/vecType/mask) is built from this element type and
  // reused for every op, so for now all operand tiles are expected to share one
  // element type; validate that each tile operand is a supported int/float type
  // (the trailing scalar args are index/int and are handled separately). This runs
  // before the helper is created so a validation failure leaks no partial IR.
  Type elementType = cast<MemRefType>(functionInputs.front()).getElementType();
  for (size_t i = 0; i < helperOperands.size(); ++i) {
    Type tileElementType = cast<MemRefType>(functionInputs[i]).getElementType();
    if (!isa<IntegerType>(tileElementType) && !isa<FloatType>(tileElementType))
      return vectorOp->emitError("unsupported element type for vector binary helper: ")
             << tileElementType;
  }
  auto lanesOr = getVectorLaneCount(elementType);
  if (failed(lanesOr))
    return failure();
  int64_t lanes = *lanesOr;
  if (lanes <= 0)
    return failure();

  std::string helperName = buildUniqueVectorHelperName(module, nextVectorRegionId);
  auto helper = moduleBuilder.create<func::FuncOp>(vectorOp->getLoc(), helperName, funcType);
  helper.setPrivate();
  helper->setAttr(hivm::TFuncCoreTypeAttr::name,
                  hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
  helper->setAttr("hivm.vector_function", UnitAttr::get(ctx));
  helper->setAttr("no_inline", UnitAttr::get(ctx));

  Block *entry = helper.addEntryBlock();
  OpBuilder b = OpBuilder::atBlockBegin(entry);

  VecLowerCtx lowerCtx{lanes, elementType, VectorType::get({lanes}, elementType),
                       VectorType::get({lanes}, b.getI1Type())};
  DenseMap<Value, Value> valueMap;
  for (auto [i, operand] : llvm::enumerate(helperOperands))
    valueMap[operand] = entry->getArgument(i);
  // Captured scalars map to their trailing block arguments.
  for (auto [j, scalar] : llvm::enumerate(scalarOperands))
    valueMap[scalar] = entry->getArgument(helperOperands.size() + j);
  if (failed(lowerNestedVectorBlock(body, b, valueMap, lowerCtx))) {
    // Discard the partially-built helper so an unsupported construct fails
    // cleanly (the vec.func is left intact) instead of leaking malformed IR.
    helper.erase();
    return failure();
  }
  b.create<func::ReturnOp>(vectorOp->getLoc());
  return helper;
}

class LowerVecFuncRegionPattern : public OpRewritePattern<::tla::VecFuncOp> {
public:
  LowerVecFuncRegionPattern(MLIRContext *context, ModuleOp module, int &nextVectorRegionId,
                            DenseMap<Value, Value> &handoffMemrefByTensor)
      : OpRewritePattern<::tla::VecFuncOp>(context, /*benefit=*/2), module(module),
        nextVectorRegionId(nextVectorRegionId), handoffMemrefByTensor(handoffMemrefByTensor) {}

  LogicalResult matchAndRewrite(::tla::VecFuncOp vecFuncOp,
                                PatternRewriter &rewriter) const override {
    auto *body = vecFuncOp.getBody().empty() ? nullptr : &vecFuncOp.getBody().front();
    if (!body)
      return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.vec.func body");

    // Collect the load / binary compute / store ops (used for arg dedup and
    // graph validation); the helper builder walks the region itself to carry
    // the control flow structure.
    SmallVector<::tla::LoadOp, 4> loads;
    SmallVector<::tla::FullOp, 4> fulls;
    SmallVector<Operation *, 4> computeOps;
    SmallVector<::tla::StoreOp, 2> stores;
    vecFuncOp->walk([&](Operation *op) {
      if (auto load = dyn_cast<::tla::LoadOp>(op)) {
        loads.push_back(load);
      } else if (auto full = dyn_cast<::tla::FullOp>(op)) {
        fulls.push_back(full);
      } else if (auto store = dyn_cast<::tla::StoreOp>(op)) {
        stores.push_back(store);
      } else if (isVectorComputeOp(op)) {
        computeOps.push_back(op);
      }
      return WalkResult::advance();
    });
    if (stores.empty())
      return rewriter.notifyMatchFailure(
          vecFuncOp, "expected tla.vec.func body with a tla.store");

    // Validate the graph: every compute operand and store source must come from
    // a tla.load result or a prior compute result inside this region.
    DenseSet<Value> producedValues;
    for (::tla::LoadOp load : loads)
      producedValues.insert(load.getResult());
    for (::tla::FullOp full : fulls)
      producedValues.insert(full.getResult());
    for (Operation *computeOp : computeOps) {
      if (computeOp->getNumResults() != 1)
        return rewriter.notifyMatchFailure(vecFuncOp, "unexpected tla compute op shape");
      if (auto info = getAnyVectorOperationInfo(computeOp)) {
        // lhs/rhs must be produced inside the region; vector-scalar rhs is a
        // scalar value captured or cloned into the helper.
        // The optional mask comes from tla.create_mask and is validated separately.
        TlaBinaryOperands ops = info->operands;
        if (!ops.lhs || !ops.rhs || !producedValues.contains(ops.lhs))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected binary op operand from tla.load or prior compute op");
        if (info->rhsKind == VectorRhsKind::Vector && !producedValues.contains(ops.rhs))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected binary op rhs from tla.load or prior compute op");
      } else if (auto whereOp = dyn_cast<::tla::WhereOp>(computeOp)) {
        if (!producedValues.contains(whereOp.getX()) ||
            !producedValues.contains(whereOp.getY()))
          return rewriter.notifyMatchFailure(
              vecFuncOp, "expected tla.where operand from tla.load or prior compute op");
      } else {
        return rewriter.notifyMatchFailure(vecFuncOp, "unexpected tla compute op");
      }
      producedValues.insert(computeOp->getResult(0));
    }
    for (::tla::StoreOp store : stores)
      if (!producedValues.contains(store.getSource()))
        return rewriter.notifyMatchFailure(
            vecFuncOp, "expected tla.store source from tla.load or compute op");

    auto funcOp = vecFuncOp->getParentOfType<func::FuncOp>();
    if (!funcOp)
      return rewriter.notifyMatchFailure(vecFuncOp, "expected enclosing func.func");

    // The helper takes one full-size UB memref per referenced tensor, in body
    // order. Compute that operand list once and use it for both the helper
    // signature and the call.
    SmallVector<Value> helperOperands;
    collectVectorHelperOperands(body, helperOperands);
    if (helperOperands.empty())
      return rewriter.notifyMatchFailure(vecFuncOp, "expected vector region tensor operands");
    // Scalars captured from outside the region (e.g. a sub_block_idx computed at
    // the top of the kernel) are passed as trailing scalar arguments.
    SmallVector<Value> scalarOperands;
    collectVectorHelperScalarOperands(vecFuncOp, scalarOperands);

    auto helperOr = buildHelperFunc(module, funcOp, vecFuncOp, helperOperands, scalarOperands,
                                    nextVectorRegionId, handoffMemrefByTensor);
    if (failed(helperOr))
      return rewriter.notifyMatchFailure(vecFuncOp, "failed to build vector helper function");
    auto helper = *helperOr;

    // The for/if control flow now lives inside the helper, so this is a single
    // call (passing the full UB memrefs) that replaces the whole vec.func region.
    rewriter.setInsertionPoint(vecFuncOp);
    SmallVector<Value, 8> callOperands;
    callOperands.reserve(helperOperands.size());
    for (Value tensor : helperOperands) {
      auto type = getVectorHelperArgMemrefType(tensor);
      if (failed(type))
        return rewriter.notifyMatchFailure(
            vecFuncOp, "failed to type UB memref for vector helper call");
      FailureOr<Value> base = getMakeTensorLikeFlatMemref(tensor);
      if (failed(base))
        base = materializeBaseMemref(rewriter, vecFuncOp.getLoc(), tensor,
                                     /*handoffMemrefByTensor=*/nullptr);
      if (failed(base))
        return rewriter.notifyMatchFailure(
            vecFuncOp, "failed to materialize UB memref for vector helper call");
      auto arg = castMemrefToExpected(rewriter, vecFuncOp.getLoc(), *base, *type);
      if (failed(arg))
        return rewriter.notifyMatchFailure(vecFuncOp,
                                           "failed to cast helper operand to expected memref type");
      callOperands.push_back(*arg);
    }
    // Captured scalars are defined in the parent (before this region), so they
    // dominate the call — pass them directly as trailing call operands.
    for (Value scalar : scalarOperands)
      callOperands.push_back(scalar);

    auto call = rewriter.create<func::CallOp>(vecFuncOp.getLoc(), helper, callOperands);
    call->setAttr("hivm.vector_function", UnitAttr::get(rewriter.getContext()));
    call->setAttr("no_inline", UnitAttr::get(rewriter.getContext()));
    rewriter.eraseOp(vecFuncOp);
    return success();
  }

private:
  ModuleOp module;
  int &nextVectorRegionId;
  DenseMap<Value, Value> &handoffMemrefByTensor;
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
    // Every tla.vector op is a frontend-authored wrapper region; inline it.
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
  funcOp.walk([&](::tla::VectorOp vectorOp) { wrappers.push_back(vectorOp); });

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
                                        DenseMap<Value, Value> &handoffMemrefByTensor) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InlineVectorRegionWrapperPattern>(ctx);
  patterns.add<LowerVecFuncRegionPattern>(ctx, module, nextVectorRegionId,
                                          handoffMemrefByTensor);
  patterns.add<LowerCopyPattern>(ctx, handoffMemrefByTensor);
  patterns.add<EraseDeadTlaScaffoldingPattern<::tla::MakeTensorLikeOp>,
               EraseDeadTlaScaffoldingPattern<::tla::MakeTensorOp>,
               EraseDeadTlaScaffoldingPattern<::tla::TileViewOp>,
               EraseDeadTlaScaffoldingPattern<::tla::MakeShapeOp>,
               EraseDeadTlaScaffoldingPattern<::tla::MakeCoordOp>,
               EraseDeadTlaScaffoldingPattern<::tla::RecastPtrOp>,
               EraseDeadTlaScaffoldingPattern<::tla::HivmMemrefAsPtrOp>>(ctx);
}

// Per-core identity queries (block_idx / block_dim / sub_block_idx) must be
// computed outside a tla.vec.func and passed in; emitting them inside the vector
// region produces an op the vector backend cannot codegen. tla-lower-to-hivm runs
// before this pass, so an in-region query already appears as its lowered hivm form
// (block_idx -> hivm.get_block_idx, block_dim -> hivm.get_block_num,
// sub_block_idx -> hivm.get_sub_block_idx); match both spellings.
static bool isIllegalVecFuncArchOp(Operation *op, StringRef &dslName) {
  if (isa<::tla::BlockIdxOp, hivm::GetBlockIdxOp>(op)) {
    dslName = "tla.arch.block_idx";
    return true;
  }
  if (isa<::tla::BlockDimOp, hivm::GetBlockNumOp>(op)) {
    dslName = "tla.arch.block_dim";
    return true;
  }
  if (isa<::tla::SubBlockIdxOp, hivm::GetSubBlockIdxOp>(op)) {
    dslName = "tla.arch.sub_block_idx";
    return true;
  }
  return false;
}

// Fail compilation if any per-core identity query is used inside a tla.vec.func.
static LogicalResult checkNoArchOpsInVecFunc(func::FuncOp funcOp) {
  LogicalResult result = success();
  funcOp.walk([&](::tla::VecFuncOp vecFuncOp) {
    vecFuncOp.getBody().walk([&](Operation *op) {
      StringRef dslName;
      if (isIllegalVecFuncArchOp(op, dslName)) {
        op->emitOpError() << "'" << dslName
                          << "' is not allowed inside a tla.vec.func region; compute it "
                             "outside the region and pass the value in";
        result = failure();
      }
    });
  });
  return result;
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

    nextVectorRegionId = 0;

    // Snapshot the functions up front: lowering a vec.func appends a new
    // vector_region helper to the module, and that helper must not be fed back
    // through the lowering/folding driver (it already holds lowered AVE ops and
    // the carried scf control flow).
    SmallVector<func::FuncOp, 4> funcOps(module.getOps<func::FuncOp>());
    for (func::FuncOp funcOp : funcOps) {
      if (funcOp.isDeclaration())
        continue;
      // Skip the generated vector_region helpers: they already hold lowered AVE
      // ops and the carried scf control flow, and must not be re-driven.
      if (funcOp->hasAttr(kHivmVectorFunctionAttrName))
        continue;
      // Only AIV (and not-yet-split MIX) functions hold vector work. Their core
      // kind is the func_core_type set by the infer pass, falling back to the
      // module core type for pure-vector entries (whose func_core_type is
      // intentionally stripped by the HACC attr convention).
      std::optional<HivmCoreKind> coreKind = getExpectedFunctionCoreKind(funcOp.getOperation());
      if (coreKind != HivmCoreKind::AIV && coreKind != HivmCoreKind::MIX)
        continue;
      if (failed(checkNoArchOpsInVecFunc(funcOp))) {
        signalPassFailure();
        return;
      }
      inlineVectorRegionWrappers(funcOp);
      DenseMap<Value, Value> handoffMemrefByTensor;
      RewritePatternSet patterns(&getContext());
      populateTlaToVectorPatterns(patterns, module, nextVectorRegionId, handoffMemrefByTensor);
      if (failed(mlir::applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
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
