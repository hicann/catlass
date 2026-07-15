#include "Passes/TlaTensorToMemref.h"

#include "PassesCommon.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h" // ReassociationIndices

#include "llvm/ADT/DenseSet.h"

#include <array>

namespace tla {

mlir::FailureOr<mlir::MemRefType> bridgeTlaTensorType(mlir::Type tlaTensorType) {
  return bridgeTlaFuncTensorType(tlaTensorType);
}

bool isPackedLayout(TensorLayoutTag layoutTag) {
  return layoutTag == TensorLayoutTag::zN || layoutTag == TensorLayoutTag::zZ ||
         layoutTag == TensorLayoutTag::nZ || layoutTag == TensorLayoutTag::L0C;
}

bool isLinearLayout(TensorLayoutTag layoutTag) {
  return layoutTag == TensorLayoutTag::RowMajor || layoutTag == TensorLayoutTag::ColumnMajor;
}

llvm::StringRef stringifyTensorLayoutTag(TensorLayoutTag layoutTag) {
  switch (layoutTag) {
  case TensorLayoutTag::Unknown:
    return "unknown";
  case TensorLayoutTag::RowMajor:
    return "row_major";
  case TensorLayoutTag::ColumnMajor:
    return "column_major";
  case TensorLayoutTag::zN:
    return "zN";
  case TensorLayoutTag::zZ:
    return "zZ";
  case TensorLayoutTag::nZ:
    return "nZ";
  case TensorLayoutTag::L0C:
    return "L0Clayout";
  }
  return "unknown";
}

mlir::FailureOr<TensorLayoutTag> convertTlaLayoutTag(::LayoutTag layoutTag) {
  switch (layoutTag) {
  case LayoutTag::row_major:
    return TensorLayoutTag::RowMajor;
  case LayoutTag::column_major:
    return TensorLayoutTag::ColumnMajor;
  case LayoutTag::zN:
    return TensorLayoutTag::zN;
  case LayoutTag::nZ:
    return TensorLayoutTag::nZ;
  case LayoutTag::zZ:
    return TensorLayoutTag::zZ;
  case LayoutTag::L0Clayout:
    return TensorLayoutTag::L0C;
  }
  return failure();
}

mlir::FailureOr<TensorLayoutTag> parseTensorLayoutTagAttr(llvm::StringRef layouttag) {
  auto layoutTag = symbolizeLayoutTag(layouttag);
  if (!layoutTag)
    return failure();
  return convertTlaLayoutTag(*layoutTag);
}

mlir::FailureOr<TensorLayoutTag> getExplicitTensorLayoutTagAttr(mlir::Operation *op) {
  auto layoutTagAttr = op->getAttrOfType<StringAttr>("layouttag");
  if (!layoutTagAttr)
    return failure();
  return parseTensorLayoutTagAttr(layoutTagAttr.getValue());
}

mlir::FailureOr<TileTypeInfo> decodeTileTypeInfo(mlir::Type tileType) {
  TileTypeInfo info;
  auto tensorTy = llvm::dyn_cast<::tla::TlaTensorType>(tileType);
  if (!tensorTy)
    return failure();
  auto layout = tensorTy.getLayout();
  auto ptr = tensorTy.getPtr();
  if (!layout.getOrigin() || !ptr.getPointee())
    return failure();

  if (failed(::tla::getTlaIndexTreeLeaves(layout.getShape().getTree(), info.shapeDims)) ||
      failed(::tla::getTlaIndexTreeLeaves(layout.getStride().getTree(), info.strideDims)) ||
      failed(::tla::getTlaIndexTreeLeaves(tensorTy.getCoord().getTree(), info.coordDims)) ||
      failed(::tla::getTlaIndexTreeLeaves(layout.getOrigin().getTree(), info.originShapeDims)))
    return failure();

  std::string elemBuf;
  llvm::raw_string_ostream elemOs(elemBuf);
  elemOs << ptr.getPointee();
  elemOs.flush();
  info.elementType = std::move(elemBuf);
  info.mlirElementType = ptr.getPointee();
  info.tlaAddressSpace = ptr.getAddrspace();
  info.addressSpace = stringifyAddressSpace(ptr.getAddrspace()).str();
  auto layoutTag = convertTlaLayoutTag(layout.getLayoutTag());
  if (info.elementType.empty() || info.addressSpace.empty() || failed(layoutTag))
    return failure();
  info.layoutTag = *layoutTag;

  info.rank = static_cast<int64_t>(info.coordDims.size());
  if (isLinearLayout(info.layoutTag)) {
    if (info.rank == 1 && info.shapeDims.size() == 1 && info.strideDims.size() == 1 &&
        info.originShapeDims.size() == 1) {
      int64_t extent = info.shapeDims[0];
      int64_t stride = info.strideDims[0];
      int64_t origin = info.originShapeDims[0];
      int64_t coord = info.coordDims[0];
      info.shapeDims = {1, extent};
      info.strideDims = {extent == ShapedType::kDynamic ? ShapedType::kDynamic : stride * extent,
                         stride};
      info.originShapeDims = {1, origin};
      info.coordDims = {0, coord};
      info.rank = 2;
    }
    if (info.rank != 2 || info.shapeDims.size() != 2 || info.strideDims.size() != 2 ||
        info.originShapeDims.size() != 2)
      return failure();
  } else if (isPackedLayout(info.layoutTag)) {
    if (info.rank != 2 || info.originShapeDims.size() != 2)
      return failure();
    if (info.shapeDims.size() != 4 || info.strideDims.size() != 4)
      return failure();
  } else {
    return failure();
  }
  return info;
}

mlir::FailureOr<ParsedTensorInfo> parseTensorInfo(mlir::Type tensorType) {
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

mlir::Value castValueToI64(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value) {
  Type type = value.getType();
  if (type.isInteger(64))
    return value;
  if (type.isIndex())
    return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), value);
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() < 64)
      return builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), value);
    if (intType.getWidth() > 64)
      return builder.create<arith::TruncIOp>(loc, builder.getI64Type(), value);
  }
  return value;
}

mlir::FailureOr<mlir::Value> castMemrefToType(mlir::OpBuilder &builder, mlir::Location loc,
                                              mlir::Value value, mlir::MemRefType memrefType) {
  if (value.getType() == memrefType)
    return value;
  if (!isa<MemRefType>(value.getType()))
    return failure();
  return builder.create<mlir::memref::CastOp>(loc, memrefType, value).getResult();
}

mlir::MemRefType getDynamicStridedMemrefType(mlir::MemRefType memrefType) {
  SmallVector<int64_t, 4> dynamicShape(memrefType.getRank(), ShapedType::kDynamic);
  SmallVector<int64_t, 4> dynamicStrides(memrefType.getRank(), ShapedType::kDynamic);
  auto layout =
      StridedLayoutAttr::get(memrefType.getContext(), ShapedType::kDynamic, dynamicStrides);
  return MemRefType::get(dynamicShape, memrefType.getElementType(), layout,
                         memrefType.getMemorySpace());
}

bool validateTensorDescriptorV1(mlir::Operation *op, const TensorDescriptor &desc,
                                llvm::StringRef errorMessage, bool requireShapeOperands) {
  if (!desc.bridgedBaseMemrefType || desc.rank != 2 || desc.addrspace.empty() ||
      desc.elementType.empty() || !desc.rowOffset.getType().isIndex() ||
      !desc.colOffset.getType().isIndex() || !desc.stride0.getType().isIndex() ||
      !desc.stride1.getType().isIndex() || !desc.originShape0.getType().isIndex() ||
      !desc.originShape1.getType().isIndex() || !desc.absCoord0.getType().isIndex() ||
      !desc.absCoord1.getType().isIndex()) {
    op->emitError() << errorMessage;
    return false;
  }
  if (requireShapeOperands &&
      (!desc.shape0.getType().isIndex() || !desc.shape1.getType().isIndex())) {
    op->emitError() << errorMessage;
    return false;
  }
  if (isPackedLayout(desc.layoutTag)) {
    if (desc.packedShape.size() != 4 || desc.packedStride.size() != 4) {
      op->emitError() << errorMessage;
      return false;
    }
    for (Value value : desc.packedShape) {
      if (!value.getType().isIndex()) {
        op->emitError() << errorMessage;
        return false;
      }
    }
    for (Value value : desc.packedStride) {
      if (!value.getType().isIndex()) {
        op->emitError() << errorMessage;
        return false;
      }
    }
  }
  return true;
}

mlir::Value makeIndexConstant(mlir::OpBuilder &builder, mlir::Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

mlir::FailureOr<mlir::Value> makeZeroValue(mlir::OpBuilder &builder, mlir::Location loc,
                                           mlir::Type type) {
  if (isa<FloatType>(type))
    return builder.create<arith::ConstantOp>(loc, type, builder.getFloatAttr(type, 0.0))
        .getResult();
  if (isa<IntegerType>(type))
    return builder.create<arith::ConstantOp>(loc, type, builder.getIntegerAttr(type, 0))
        .getResult();
  return failure();
}

mlir::FailureOr<mlir::Value> makeStaticTensorInfoIndex(mlir::OpBuilder &builder, mlir::Operation *op,
                                                       int64_t value, llvm::StringRef fieldName) {
  if (value == ShapedType::kDynamic) {
    op->emitError() << "dynamic tensor metadata leaf in " << fieldName
                    << " is not yet supported in LowerToStdPass descriptor extraction";
    return failure();
  }
  return makeIndexConstant(builder, op->getLoc(), value);
}

mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
materializeStaticTensorInfoIndices(mlir::OpBuilder &builder, mlir::Operation *op,
                                   llvm::ArrayRef<int64_t> values, llvm::StringRef fieldName) {
  SmallVector<Value, 4> materialized;
  materialized.reserve(values.size());
  for (int64_t value : values) {
    FailureOr<Value> indexValue = makeStaticTensorInfoIndex(builder, op, value, fieldName);
    if (failed(indexValue))
      return failure();
    materialized.push_back(*indexValue);
  }
  return materialized;
}

mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
materializeTensorInfoIndicesWithDynamicValues(mlir::OpBuilder &builder, mlir::Operation *op,
                                              llvm::ArrayRef<int64_t> values,
                                              llvm::ArrayRef<mlir::Value> dynamicValues,
                                              llvm::StringRef fieldName) {
  SmallVector<Value, 4> materialized;
  materialized.reserve(values.size());
  size_t dynamicIndex = 0;
  for (int64_t value : values) {
    if (value == ShapedType::kDynamic) {
      if (dynamicIndex >= dynamicValues.size()) {
        op->emitError() << "dynamic tensor metadata leaf in " << fieldName
                        << " is missing SSA dynamic value in descriptor extraction";
        return failure();
      }
      Value dynamicValue = dynamicValues[dynamicIndex++];
      if (!dynamicValue || !dynamicValue.getType().isIndex()) {
        op->emitError() << "dynamic tensor metadata leaf in " << fieldName
                        << " requires index-typed SSA dynamic value";
        return failure();
      }
      materialized.push_back(dynamicValue);
      continue;
    }
    FailureOr<Value> indexValue = makeStaticTensorInfoIndex(builder, op, value, fieldName);
    if (failed(indexValue))
      return failure();
    materialized.push_back(*indexValue);
  }
  return materialized;
}

mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
packRank2DynamicMetadataLeaves(mlir::Operation *op, llvm::ArrayRef<int64_t> leafDims,
                               mlir::Value axis0, mlir::Value axis1, llvm::StringRef fieldName) {
  SmallVector<Value, 4> dynamicVals;
  if (leafDims.size() != 2) {
    op->emitError() << fieldName << " expects two index leaves for rank-2 tla.tile_view metadata";
    return failure();
  }
  for (size_t i = 0; i < 2; ++i) {
    if (leafDims[i] == ShapedType::kDynamic)
      dynamicVals.push_back(i == 0 ? axis0 : axis1);
  }
  return dynamicVals;
}

mlir::FailureOr<TensorDescriptor>
buildTensorDescriptorFromTensorInfo(mlir::OpBuilder &builder, mlir::Operation *op, mlir::Value base,
                                    mlir::Type bridgedBaseMemrefType, const TileTypeInfo &info,
                                    llvm::ArrayRef<mlir::Value> coordDynamicValues,
                                    llvm::ArrayRef<mlir::Value> originShapeDynamicValues) {
  FailureOr<SmallVector<Value, 4>> coord =
      coordDynamicValues.empty()
          ? materializeStaticTensorInfoIndices(builder, op, info.coordDims, "coord")
          : materializeTensorInfoIndicesWithDynamicValues(builder, op, info.coordDims,
                                                          coordDynamicValues, "coord");
  FailureOr<SmallVector<Value, 4>> originShape =
      originShapeDynamicValues.empty()
          ? materializeStaticTensorInfoIndices(builder, op, info.originShapeDims, "origin_shape")
          : materializeTensorInfoIndicesWithDynamicValues(
                builder, op, info.originShapeDims, originShapeDynamicValues, "origin_shape");
  if (failed(coord) || failed(originShape))
    return failure();

  Value stride0;
  Value stride1;
  Value shape0;
  Value shape1;
  SmallVector<Value, 4> packedShape;
  SmallVector<Value, 4> packedStride;

  if (isLinearLayout(info.layoutTag)) {
    FailureOr<SmallVector<Value, 4>> shape =
        materializeStaticTensorInfoIndices(builder, op, info.shapeDims, "shape");
    FailureOr<SmallVector<Value, 4>> stride =
        materializeStaticTensorInfoIndices(builder, op, info.strideDims, "stride");
    if (failed(shape) || failed(stride))
      return failure();
    shape0 = (*shape)[0];
    shape1 = (*shape)[1];
    stride0 = (*stride)[0];
    stride1 = (*stride)[1];
  } else {
    FailureOr<SmallVector<Value, 4>> shape =
        materializeStaticTensorInfoIndices(builder, op, info.shapeDims, "packed shape");
    FailureOr<SmallVector<Value, 4>> stride =
        materializeStaticTensorInfoIndices(builder, op, info.strideDims, "packed stride");
    if (failed(shape) || failed(stride))
      return failure();
    packedShape = std::move(*shape);
    packedStride = std::move(*stride);
    shape0 = (*originShape)[0];
    shape1 = (*originShape)[1];
    stride0 = packedStride[0];
    stride1 = packedStride[1];
  }

  return TensorDescriptor{base,
                          bridgedBaseMemrefType,
                          (*coord)[0],
                          (*coord)[1],
                          stride0,
                          stride1,
                          shape0,
                          shape1,
                          (*originShape)[0],
                          (*originShape)[1],
                          (*coord)[0],
                          (*coord)[1],
                          info.layoutTag,
                          info.addressSpace,
                          info.elementType,
                          info.rank,
                          std::move(packedShape),
                          std::move(packedStride)};
}

mlir::Value IndexConstantCache::get(mlir::Operation *anchor, int64_t value, unsigned bits) {
  Key key{value, bits};
  Block *scopeBlock = nullptr;
  if (auto tlaFunc = anchor->getParentOfType<::tla::FuncOp>()) {
    scopeBlock = &tlaFunc.getBody().front();
  } else if (auto func = anchor->getParentOfType<mlir::func::FuncOp>()) {
    scopeBlock = &func.getBody().front();
  } else if (auto module = anchor->getParentOfType<ModuleOp>()) {
    scopeBlock = &module.getBodyRegion().front();
  } else {
    scopeBlock = anchor->getBlock();
  }
  auto &cache = byScope[scopeBlock];
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  OpBuilder builder(scopeBlock, scopeBlock->begin());
  Value constant;
  if (bits == 0) {
    constant = builder.create<arith::ConstantIndexOp>(anchor->getLoc(), value);
  } else {
    auto intType = builder.getIntegerType(bits);
    auto intAttr = builder.getIntegerAttr(intType, value);
    constant = builder.create<arith::ConstantOp>(anchor->getLoc(), intType, intAttr);
  }
  cache[key] = constant;
  return constant;
}

mlir::FailureOr<TensorDescriptor> buildTileViewResultDescriptorFromParent(
    mlir::Operation *op, mlir::Value base, mlir::MemRefType bridgedBaseType,
    const TileTypeInfo &info, const TensorDescriptor &parent, mlir::Value row, mlir::Value col,
    mlir::Value sh0, mlir::Value sh1, ConstantFactory getConstant) {
  Location loc = op->getLoc();
  OpBuilder b(op);

  if (!isPackedLayout(info.layoutTag) && !isLinearLayout(info.layoutTag)) {
    op->emitError() << "tile_view: unsupported layout tag for descriptor lowering";
    return failure();
  }

  Value abs0 = b.create<arith::AddIOp>(loc, parent.absCoord0, row);
  Value abs1 = b.create<arith::AddIOp>(loc, parent.absCoord1, col);
  Value rest0 = b.create<arith::SubIOp>(loc, parent.originShape0, row);
  Value rest1 = b.create<arith::SubIOp>(loc, parent.originShape1, col);
  Value origin0 = b.create<arith::MinSIOp>(loc, sh0, rest0);
  Value origin1 = b.create<arith::MinSIOp>(loc, sh1, rest1);

  Value stride0;
  Value stride1;
  Value shape0;
  Value shape1;
  SmallVector<Value, 4> packedShape;
  SmallVector<Value, 4> packedStride;

  if (isLinearLayout(info.layoutTag)) {
    auto materializeRowMajorStride = [&](int64_t dim, Value parentStride) -> FailureOr<Value> {
      if (dim == ShapedType::kDynamic) {
        if (!parentStride || !parentStride.getType().isIndex()) {
          op->emitError() << "tile_view: dynamic stride requires parent tile descriptor "
                             "stride (index SSA)";
          return failure();
        }
        return parentStride;
      }
      return getConstant(op, dim, 0);
    };
    FailureOr<Value> st0 = materializeRowMajorStride(info.strideDims[0], parent.stride0);
    FailureOr<Value> st1 = materializeRowMajorStride(info.strideDims[1], parent.stride1);
    if (failed(st0) || failed(st1))
      return failure();
    stride0 = *st0;
    stride1 = *st1;
    shape0 = info.shapeDims[0] == ShapedType::kDynamic ? sh0 : getConstant(op, info.shapeDims[0], 0);
    shape1 = info.shapeDims[1] == ShapedType::kDynamic ? sh1 : getConstant(op, info.shapeDims[1], 0);
  } else {
    auto ceilDivIndexByPositiveConst = [&](Value numerator, int64_t divisor) -> FailureOr<Value> {
      if (divisor <= 0) {
        op->emitError() << "tile_view: packed shape dynamic leaf requires positive divisor, got "
                        << divisor;
        return failure();
      }
      Value divisorV = getConstant(op, divisor, 0);
      Value one = getConstant(op, 1, 0);
      Value adjusted =
          b.create<arith::AddIOp>(loc, numerator, b.create<arith::SubIOp>(loc, divisorV, one));
      return b.create<arith::DivSIOp>(loc, adjusted, divisorV).getResult();
    };
    auto materializePackedShapeLeaf = [&](size_t idx) -> FailureOr<Value> {
      int64_t leaf = info.shapeDims[idx];
      if (leaf != ShapedType::kDynamic)
        return getConstant(op, leaf, 0);
      if (info.shapeDims.size() < 4) {
        op->emitError() << "tile_view: packed shape must have 4 leaves";
        return failure();
      }
      if (idx == 1) {
        if (info.shapeDims[0] == ShapedType::kDynamic) {
          op->emitError()
              << "tile_view: dynamic packed shape leaf index 1 requires static leaf index 0";
          return failure();
        }
        return ceilDivIndexByPositiveConst(sh0, info.shapeDims[0]);
      }
      if (idx == 3) {
        if (info.shapeDims[2] == ShapedType::kDynamic) {
          op->emitError()
              << "tile_view: dynamic packed shape leaf index 3 requires static leaf index 2";
          return failure();
        }
        return ceilDivIndexByPositiveConst(sh1, info.shapeDims[2]);
      }
      op->emitError() << "tile_view: dynamic packed shape leaf at index " << idx
                      << " is unsupported; only indices 1 and 3 may be dynamic";
      return failure();
    };
    auto materializePackedStrideLeaf = [&](size_t idx) -> FailureOr<Value> {
      int64_t leaf = info.strideDims[idx];
      if (leaf != ShapedType::kDynamic)
        return getConstant(op, leaf, 0);
      if (idx < parent.packedStride.size() && parent.packedStride[idx] &&
          parent.packedStride[idx].getType().isIndex())
        return parent.packedStride[idx];
      op->emitError() << "tile_view: dynamic packed stride leaf index " << idx
                      << " requires parent packed stride SSA";
      return failure();
    };
    packedShape.reserve(info.shapeDims.size());
    packedStride.reserve(info.strideDims.size());
    for (size_t i = 0; i < info.shapeDims.size(); ++i) {
      FailureOr<Value> leaf = materializePackedShapeLeaf(i);
      if (failed(leaf))
        return failure();
      packedShape.push_back(*leaf);
    }
    for (size_t i = 0; i < info.strideDims.size(); ++i) {
      FailureOr<Value> leaf = materializePackedStrideLeaf(i);
      if (failed(leaf))
        return failure();
      packedStride.push_back(*leaf);
    }
    shape0 = origin0;
    shape1 = origin1;
    stride0 = packedStride[0];
    stride1 = packedStride[1];
  }

  return TensorDescriptor{base,
                          bridgedBaseType,
                          abs0,
                          abs1,
                          stride0,
                          stride1,
                          shape0,
                          shape1,
                          origin0,
                          origin1,
                          abs0,
                          abs1,
                          info.layoutTag,
                          info.addressSpace,
                          info.elementType,
                          info.rank,
                          std::move(packedShape),
                          std::move(packedStride)};
}

bool hasOnlyStagedResultUsers(mlir::Operation *op, llvm::ArrayRef<mlir::Operation *> stagedErase) {
  if (!op || op->getNumResults() == 0)
    return false;
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (!llvm::is_contained(stagedErase, user))
        return false;
    }
  }
  return true;
}

void stageDeadTileProducers(mlir::ModuleOp module,
                            llvm::SmallVectorImpl<mlir::Operation *> &toErase) {
  bool progress = true;
  while (progress) {
    progress = false;
    SmallVector<Operation *, 8> newlyDead;
    module.walk([&](Operation *op) {
      if (!llvm::isa<::tla::TileViewOp, ::tla::MakeTensorLikeOp, ::tla::MakeTensorOp>(op) ||
          llvm::is_contained(toErase, op))
        return;
      if (hasOnlyStagedResultUsers(op, toErase))
        newlyDead.push_back(op);
    });
    for (Operation *op : newlyDead) {
      pushStagedErase(toErase, op);
      progress = true;
    }
  }
}

void pushStagedErase(llvm::SmallVectorImpl<mlir::Operation *> *toErase, mlir::Operation *op) {
  if (toErase && op && !llvm::is_contained(*toErase, op))
    toErase->push_back(op);
}

void pushStagedErase(llvm::SmallVectorImpl<mlir::Operation *> &toErase, mlir::Operation *op) {
  if (!op || llvm::is_contained(toErase, op))
    return;
  toErase.push_back(op);
}

mlir::Value createFallbackBaseMemrefCast(mlir::OpBuilder &builder, mlir::Location loc,
                                         const TensorDescriptor &desc) {
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{desc.bridgedBaseMemrefType},
                                          ValueRange{desc.base})
      .getResult(0);
}

// For a GM linear-layout tensor with a static origin_shape, fill `originDims` and the
// contiguous strides implied by that origin (row-major: trailing product; column-major:
// leading product). Returns true if filled; false for non-GM, packed layout, or a dynamic
// origin. Uses the raw (rank-preserving) parse so the view mirrors the declared origin.
static bool tryGmOriginLayout(mlir::Type tensorTy, llvm::SmallVectorImpl<int64_t> &originDims,
                              llvm::SmallVectorImpl<int64_t> &contigStrides) {
  auto info = parseTensorInfo(tensorTy);
  if (failed(info) || info->originShape.empty())
    return false;
  if (info->addressSpace != ::AddressSpace::gm)
    return false;
  if (info->layoutTag != "row_major" && info->layoutTag != "column_major")
    return false;
  if (llvm::any_of(info->originShape, [](int64_t d) { return d == ShapedType::kDynamic; }))
    return false;
  unsigned rank = info->originShape.size();
  originDims.assign(info->originShape.begin(), info->originShape.end());
  contigStrides.assign(rank, 1);
  if (info->layoutTag == "row_major") {
    int64_t acc = 1;
    for (int i = rank - 1; i >= 0; --i) {
      contigStrides[i] = acc;
      acc *= originDims[i];
    }
  } else {
    int64_t acc = 1;
    for (unsigned i = 0; i < rank; ++i) {
      contigStrides[i] = acc;
      acc *= originDims[i];
    }
  }
  return true;
}

// Allocation capacity is optional provenance, not part of pointer identity.
// Preserve it only across joins whose alternatives prove the same capacity;
// otherwise tensor consumers build a view from their own shape/layout.
static FailureOr<int64_t>
inferStaticAllocationSizeBytes(
    Value address, llvm::DenseSet<Value> visiting,
    llvm::DenseMap<Value, int64_t> assumedCapacities =
        llvm::DenseMap<Value, int64_t>()) {
  if (auto assumed = assumedCapacities.find(address);
      assumed != assumedCapacities.end())
    return assumed->second;
  if (!visiting.insert(address).second)
    return failure();

  if (auto blockArg = dyn_cast<BlockArgument>(address)) {
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      unsigned argNumber = blockArg.getArgNumber();
      if (blockArg.getOwner() == forOp.getBody() && argNumber > 0 &&
          argNumber - 1 < forOp.getInitArgs().size()) {
        unsigned iterArgNumber = argNumber - 1;
        auto initSize = inferStaticAllocationSizeBytes(
            forOp.getInitArgs()[iterArgNumber], visiting, assumedCapacities);
        auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        if (failed(initSize) || !yield ||
            iterArgNumber >= yield.getNumOperands())
          return failure();

        // Prove the init capacity is a loop invariant. The current block
        // argument may occur recursively in the backedge expression, so use
        // the candidate capacity while checking that every yielded source
        // preserves it.
        assumedCapacities[blockArg] = *initSize;
        auto yieldSize = inferStaticAllocationSizeBytes(
            yield.getOperand(iterArgNumber), std::move(visiting),
            std::move(assumedCapacities));
        if (succeeded(yieldSize) && *yieldSize == *initSize)
          return *initSize;
      }
    }
    return failure();
  }

  Operation *def = address.getDefiningOp();
  if (!def)
    return failure();
  if (auto size =
          def->getAttrOfType<IntegerAttr>(kAllocSizeBytesMetadataAttrName))
    return size.getInt();
  if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
    if (cast.getNumOperands() == 1)
      return inferStaticAllocationSizeBytes(cast.getOperand(0),
                                            std::move(visiting),
                                            std::move(assumedCapacities));
  }

  // The Python frontend represents pointer-valued joins with structured SCF.
  // Do not claim provenance support for arith.select, which tla-lower-ptr does
  // not convert.
  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    unsigned resultNumber = cast<OpResult>(address).getResultNumber();
    scf::YieldOp thenYield = ifOp.thenYield();
    scf::YieldOp elseYield = ifOp.elseYield();
    if (!thenYield || !elseYield || resultNumber >= thenYield.getNumOperands() ||
        resultNumber >= elseYield.getNumOperands())
      return failure();
    auto thenSize = inferStaticAllocationSizeBytes(
        thenYield.getOperand(resultNumber), visiting, assumedCapacities);
    auto elseSize = inferStaticAllocationSizeBytes(
        elseYield.getOperand(resultNumber), std::move(visiting),
        std::move(assumedCapacities));
    if (succeeded(thenSize) && succeeded(elseSize) &&
        *thenSize == *elseSize)
      return *thenSize;
    return failure();
  }
  if (auto forOp = dyn_cast<scf::ForOp>(def)) {
    unsigned resultNumber = cast<OpResult>(address).getResultNumber();
    if (resultNumber >= forOp.getInitArgs().size())
      return failure();
    auto initSize = inferStaticAllocationSizeBytes(
        forOp.getInitArgs()[resultNumber], visiting, assumedCapacities);
    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (failed(initSize) || !yield || resultNumber >= yield.getNumOperands())
      return failure();

    assumedCapacities[forOp.getRegionIterArg(resultNumber)] = *initSize;
    auto yieldSize = inferStaticAllocationSizeBytes(
        yield.getOperand(resultNumber), std::move(visiting),
        std::move(assumedCapacities));
    if (succeeded(yieldSize) && *initSize == *yieldSize)
      return *initSize;
  }
  return failure();
}

static FailureOr<int64_t> getStaticAllocationElementCount(Value ptr) {
  auto ptrType = dyn_cast<::tla::PtrType>(ptr.getType());
  auto intToPtr = ptr.getDefiningOp<::tla::IntToPtrOp>();
  if (!ptrType || !intToPtr)
    return failure();
  auto sizeBytes = inferStaticAllocationSizeBytes(intToPtr.getAddr(), {});
  int64_t elementBytes = getByteSizeOfFixedWidthScalarType(ptrType.getPointee());
  if (failed(sizeBytes) || *sizeBytes < 0 || elementBytes <= 0 ||
      *sizeBytes % elementBytes != 0)
    return failure();
  return *sizeBytes / elementBytes;
}

mlir::FailureOr<mlir::Value>
materializePtrValueAsMemref(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value ptrValue, mlir::MemRefType memrefType,
                            mlir::Operation *diagnosticOp,
                            mlir::ValueRange dynamicSizes) {
  auto intToPtr = ptrValue.getDefiningOp<::tla::IntToPtrOp>();
  if (!intToPtr) {
    diagnosticOp->emitError()
        << "pointer memref materialization expects the `tla.inttoptr` "
           "boundary produced by `tla-lower-ptr`; got: "
        << ptrValue;
    return failure();
  }

  unsigned expectedDynamicSizes = llvm::count_if(
      memrefType.getShape(),
      [](int64_t dim) { return dim == ShapedType::kDynamic; });
  if (dynamicSizes.size() != expectedDynamicSizes) {
    diagnosticOp->emitError()
        << "pointer_cast materialization expected " << expectedDynamicSizes
        << " dynamic sizes, got " << dynamicSizes.size();
    return failure();
  }
  Value address = castValueToI64(builder, loc, intToPtr.getAddr());
  if (!address.getType().isInteger(64)) {
    diagnosticOp->emitError()
        << "tla.inttoptr address must lower to i64, got " << address.getType();
    return failure();
  }
  return builder
      .create<hivm::PointerCastOp>(loc, memrefType, address, dynamicSizes)
      .getResult();
}

mlir::FailureOr<mlir::Value> materializeDescriptorBaseMemref(mlir::OpBuilder &builder,
                                                             mlir::Location loc,
                                                             const TensorDescriptor &desc,
                                                             AllocatorOffsetState *allocatorState,
                                                             mlir::Operation *diagnosticOp) {
  auto memrefType = dyn_cast<MemRefType>(desc.bridgedBaseMemrefType);
  if (!memrefType)
    return failure();

  if (isa<MemRefType>(desc.base.getType()))
    return castMemrefToType(builder, loc, desc.base, memrefType);

  if (isa<::tla::PtrType>(desc.base.getType())) {
    if (auto allocationElements = getStaticAllocationElementCount(desc.base);
        succeeded(allocationElements)) {
      auto allocationType = MemRefType::get(
          {*allocationElements}, memrefType.getElementType(), AffineMap(),
          memrefType.getMemorySpace());
      return materializePtrValueAsMemref(builder, loc, desc.base,
                                         allocationType, diagnosticOp);
    }

    SmallVector<Value, 2> dynamicSizes;
    for (auto [index, dim] : llvm::enumerate(memrefType.getShape())) {
      if (dim != ShapedType::kDynamic)
        continue;
      if (memrefType.getRank() == 1) {
        dynamicSizes.push_back(builder.create<arith::MulIOp>(
            loc, desc.originShape0, desc.originShape1));
      } else if (index == 0) {
        dynamicSizes.push_back(desc.shape0);
      } else if (index == 1) {
        dynamicSizes.push_back(desc.shape1);
      } else {
        diagnosticOp->emitError()
            << "cannot derive dynamic pointer_cast size for memref dimension "
            << index;
        return failure();
      }
    }
    return materializePtrValueAsMemref(builder, loc, desc.base, memrefType,
                                       diagnosticOp, dynamicSizes);
  }

  if (allocatorState) {
    auto castOp = dyn_cast_or_null<UnrealizedConversionCastOp>(
        desc.base.getDefiningOp());
    if (castOp && castOp->getNumOperands() == 1 &&
        isa<MemRefType>(castOp->getOperand(0).getType())) {
      FailureOr<Value> cast = castMemrefToType(
          builder, loc, castOp->getOperand(0), memrefType);
      if (failed(cast))
        return failure();
      pushStagedErase(allocatorState->toErase, castOp.getOperation());
      return *cast;
    }
  }
  return createFallbackBaseMemrefCast(builder, loc, desc);
}

mlir::FailureOr<mlir::Value>
materializeTensorOperandAsMemref(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value tensor,
                                 mlir::Type tensorType,
                                 llvm::SmallVectorImpl<mlir::Operation *> &toErase) {
  if (auto castOp = tensor.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp->getNumOperands() == 1 && llvm::isa<MemRefType>(castOp->getOperand(0).getType())) {
      toErase.push_back(castOp.getOperation());
      return castOp->getOperand(0);
    }
  }
  auto memrefType = bridgeTlaTensorType(tensorType);
  if (failed(memrefType))
    return failure();
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{*memrefType}, ValueRange{tensor})
      .getResult(0);
}

mlir::FailureOr<mlir::Value>
getOrMaterializeDescriptorBaseMemref(mlir::OpBuilder &builder, mlir::Location loc,
                                     const TensorDescriptor &desc,
                                     AllocatorOffsetState *allocatorState,
                                     mlir::Operation *diagnosticOp,
                                     llvm::DenseMap<mlir::Value, mlir::Value> &baseMemrefCache) {
  auto memrefType = dyn_cast<MemRefType>(desc.bridgedBaseMemrefType);
  if (!memrefType)
    return failure();

  // A proven alloc capacity describes one kernel-lifetime allocation object,
  // so its descriptor is shape-independent and safe to cache by pointer SSA.
  // Otherwise an inttoptr descriptor is a consumer-local view whose dynamic
  // sizes may be defined inside the current region/loop; materialize it at the
  // caller's insertion point and do not cache it by address alone.
  bool isIntToPtr = static_cast<bool>(
      desc.base.getDefiningOp<::tla::IntToPtrOp>());
  bool hasStaticAllocation =
      succeeded(getStaticAllocationElementCount(desc.base));
  if (isIntToPtr && !hasStaticAllocation)
    return materializeDescriptorBaseMemref(builder, loc, desc, allocatorState,
                                           diagnosticOp);

  auto it = baseMemrefCache.find(desc.base);
  if (it != baseMemrefCache.end()) {
    if (isIntToPtr)
      return it->second;
    // Return the cached base consistently with the first materialization below,
    // which returns it un-cast. A `memref.cast` to `memrefType` is only valid when
    // the ranks match; otherwise hand back the cached view directly.
    if (auto cachedType = dyn_cast<MemRefType>(it->second.getType());
        cachedType && cachedType.getRank() == memrefType.getRank())
      return castMemrefToType(builder, loc, it->second, memrefType);
    return it->second;
  }

  // Anchor the first materialization at a point that dominates every use of
  // desc.base (right after its def, or the entry of its block for a block
  // argument) so the cached memref can be reused SSA-safely by any later
  // consumer. Fall back to the caller's insertion point if neither applies.
  OpBuilder::InsertionGuard guard(builder);
  if (Operation *def = desc.base.getDefiningOp())
    builder.setInsertionPointAfter(def);
  else if (auto blockArg = dyn_cast<BlockArgument>(desc.base))
    builder.setInsertionPointToStart(blockArg.getOwner());

  FailureOr<Value> materialized =
      materializeDescriptorBaseMemref(builder, loc, desc, allocatorState, diagnosticOp);
  if (failed(materialized))
    return failure();
  baseMemrefCache[desc.base] = *materialized;
  return *materialized;
}


// ---------------------------------------------------------------------------
// Vector tile memref materialization (from TlaVectorRegionPass).
// Vector-specific policy: 256-byte lane memrefs, rank-1<->2 reshape, and the
// per-copy handoff cache. Layered on the shared decode/bridge above.
// ---------------------------------------------------------------------------

FailureOr<MemRefType> getBridgedTensorMemrefType(Value tensor) {
  return ::tla::bridgeTlaTensorType(tensor.getType());
}

FailureOr<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t numElements = 1;
  for (int64_t dim : shape) {
    if (dim <= 0 || dim == ShapedType::kDynamic)
      return failure();
    numElements *= dim;
  }
  return numElements;
}

FailureOr<int64_t> getElementByteWidth(Type elementType) {
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

FailureOr<int64_t> getVectorLaneCount(Type elementType) {
  auto elementBytes = getElementByteWidth(elementType);
  if (failed(elementBytes) || *elementBytes <= 0)
    return failure();
  constexpr int64_t kVectorBytes = 256;
  return kVectorBytes / *elementBytes;
}

FailureOr<int64_t> getVectorLanesForMemref(MemRefType type) {
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

bool isSupportedVectorTile(MemRefType type) {
  return succeeded(getVectorLanesForMemref(type));
}

// The lower-once cache lookup: the memref `tensor` was already materialized to,
// or a null Value if absent (or the cache is null). Single owner of the
// find/end idiom the raw-parse materializers share.
static Value lookupLoweredMemref(DenseMap<Value, Value> *cache, Value tensor) {
  if (!cache)
    return {};
  auto it = cache->find(tensor);
  return it != cache->end() ? it->second : Value{};
}

FailureOr<MemRefType>
getVectorHelperMemrefType(Value tensor,
                          DenseMap<Value, Value> *loweredMemrefByValue) {
  FailureOr<MemRefType> bridged = failure();
  if (Value cached = lookupLoweredMemref(loweredMemrefByValue, tensor)) {
    if (auto handoffType = dyn_cast<MemRefType>(cached.getType()))
      bridged = handoffType;
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

FailureOr<Value> castMemrefToExpected(PatternRewriter &rewriter, Location loc, Value value,
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

// Cast `src` to the memref type bridged from `tensor`'s `!tla.tensor` type (the
// recurring "adapt this memref to what a tensor operand expects" idiom).
static FailureOr<Value> castToBridgedType(PatternRewriter &rewriter, Location loc, Value src,
                                          Value tensor) {
  auto expected = getBridgedTensorMemrefType(tensor);
  if (failed(expected))
    return failure();
  return castMemrefToExpected(rewriter, loc, src, *expected);
}

// Build a rank-2 `memref.subview` of `base` at (`coord0`, `coord1`) with sizes
// `shape` and a `{rowStride, 1}` strided layout. The shared subview core used by
// both the tile-view and copy-subview rank-2 materializers.
static Value buildRank2CoordSubview(PatternRewriter &rewriter, Location loc, Value base,
                                    ArrayRef<int64_t> shape, int64_t rowStride, Value coord0,
                                    Value coord1) {
  auto baseType = cast<MemRefType>(base.getType());
  auto subviewLayout = StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                                              ArrayRef<int64_t>{rowStride, 1});
  auto subviewType =
      MemRefType::get(shape, baseType.getElementType(), subviewLayout, baseType.getMemorySpace());
  SmallVector<int64_t> staticOffsets{ShapedType::kDynamic, ShapedType::kDynamic};
  SmallVector<int64_t> staticSizes{shape.begin(), shape.end()};
  SmallVector<int64_t> staticStrides{1, 1};
  return rewriter
      .create<mlir::memref::SubViewOp>(loc, subviewType, base, ValueRange{coord0, coord1},
                                       ValueRange{}, ValueRange{}, staticOffsets, staticSizes,
                                       staticStrides)
      .getResult();
}

FailureOr<Value> materializeSingleCoordIndex(PatternRewriter &rewriter, Location loc,
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

FailureOr<SmallVector<Value, 2>> materializeRank2CoordIndices(PatternRewriter &rewriter,
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

FailureOr<Value> materializeTileViewMemref(PatternRewriter &rewriter, Location loc,
                                                  Value tensor, ::tla::TileViewOp tileView,
                                                  bool useVectorHelperType,
                                                  DenseMap<Value, Value> *loweredMemrefByValue) {
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
                                          loweredMemrefByValue);
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

    Value subview = buildRank2CoordSubview(rewriter, loc, *baseMemref, info->shape,
                                           sourceInfo->shape[1], (*coordPair)[0], (*coordPair)[1]);
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

// The `!tla.ptr` operand of a make_tensor / make_tensor_like tensor, or null.
static Value ptrOfMakeTensor(Value tensor) {
  if (auto mtl = tensor.getDefiningOp<::tla::MakeTensorLikeOp>())
    return mtl.getPtr();
  if (auto mt = tensor.getDefiningOp<::tla::MakeTensorOp>())
    return mt.getPtr();
  return {};
}

FailureOr<MemRefType> getVectorHelperArgMemrefType(Value operand) {
  Value ptr = ptrOfMakeTensor(operand);
  if (ptr && !ptr.getDefiningOp<::tla::IntToPtrOp>())
    return failure();

  auto bridged = getBridgedTensorMemrefType(operand);
  if (failed(bridged))
    return failure();
  if (ptr) {
    if (auto allocationElements = getStaticAllocationElementCount(ptr);
        succeeded(allocationElements))
      return MemRefType::get({*allocationElements}, bridged->getElementType(),
                             AffineMap(), bridged->getMemorySpace());

    SmallVector<int64_t, 4> originDims, contigStrides;
    if (tryGmOriginLayout(operand.getType(), originDims, contigStrides)) {
      auto stridedLayout = StridedLayoutAttr::get(
          operand.getContext(), ShapedType::kDynamic, contigStrides);
      return MemRefType::get(originDims, bridged->getElementType(),
                             stridedLayout, bridged->getMemorySpace());
    }
  }
  if (bridged->getRank() == 1)
    return *bridged;
  auto viewElements = getStaticNumElements(bridged->getShape());
  if (bridged->getRank() != 2 || failed(viewElements))
    return failure();
  return MemRefType::get({*viewElements}, bridged->getElementType(),
                         AffineMap(), bridged->getMemorySpace());
}

FailureOr<Value>
materializeBaseMemref(PatternRewriter &rewriter, Location loc, Value tensor,
                      DenseMap<Value, Value> *loweredMemrefByValue) {
  if (Value cached = lookupLoweredMemref(loweredMemrefByValue, tensor))
    return castToBridgedType(rewriter, loc, cached, tensor);

  if (auto castOp = tensor.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp.getNumOperands() == 1 && isa<MemRefType>(castOp.getOperand(0).getType()))
      return castToBridgedType(rewriter, loc, castOp.getOperand(0), tensor);
  }

  // tla.make_tensor{,_like} is a consumer-local view of its pointer address.
  if (isa_and_nonnull<::tla::MakeTensorLikeOp, ::tla::MakeTensorOp>(
          tensor.getDefiningOp())) {
    Value ptr = ptrOfMakeTensor(tensor);
    if (!ptr || !ptr.getDefiningOp<::tla::IntToPtrOp>())
      return failure();
    auto expected = getBridgedTensorMemrefType(tensor);
    if (failed(expected))
      return failure();
    auto base = materializePtrValueAsMemref(
        rewriter, loc, ptr, *expected, tensor.getDefiningOp());
    if (failed(base))
      return failure();
    return castToBridgedType(rewriter, loc, *base, tensor);
  }

  if (auto tileView = tensor.getDefiningOp<::tla::TileViewOp>()) {
    Value source = tileView.getSource();
    auto info = parseTensorInfo(tensor.getType());
    if (succeeded(info) && (info->shape.size() == 1 || info->shape.size() == 2) &&
        succeeded(getVectorHelperMemrefType(tensor)))
      return materializeTileViewMemref(rewriter, loc, tensor, tileView,
                                       /*useVectorHelperType=*/true, loweredMemrefByValue);
    if (isa<MemRefType>(source.getType())) {
      if (succeeded(info) && (info->shape.size() == 1 || info->shape.size() == 2))
        return materializeTileViewMemref(rewriter, loc, tensor, tileView,
                                         /*useVectorHelperType=*/false,
                                         loweredMemrefByValue);
      return castToBridgedType(rewriter, loc, source, tensor);
    }
    return materializeBaseMemref(rewriter, loc, source, loweredMemrefByValue);
  }

  if (isa<MemRefType>(tensor.getType()))
    return tensor;

  return failure();
}

FailureOr<Value> materializeCopySubview1D(PatternRewriter &rewriter, Location loc,
                                                 Value tensor,
                                                 DenseMap<Value, Value> *loweredMemrefByValue) {
  auto info = parseTensorInfo(tensor.getType());
  if (failed(info))
    return failure();
  if (info->shape.size() != 1 || info->coord.size() != 1 || info->layoutTag != "row_major")
    return failure();
  if (info->shape[0] == ShapedType::kDynamic || info->coord[0] == ShapedType::kDynamic)
    return failure();

  auto baseMemref = materializeBaseMemref(rewriter, loc, tensor, loweredMemrefByValue);
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

FailureOr<Value> materializeCopySubviewRank2(
    PatternRewriter &rewriter, Location loc, Value tensor,
    DenseMap<Value, Value> *loweredMemrefByValue, ArrayRef<int64_t> concreteShape) {
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

  auto viewElements = getStaticNumElements(shape);
  if (failed(viewElements))
    return failure();

  Value flatMemref;
  if (Value ptr = ptrOfMakeTensor(tensor);
      ptr && ptr.getDefiningOp<::tla::IntToPtrOp>()) {
    auto allocationElements = getStaticAllocationElementCount(ptr);
    auto bridged = getBridgedTensorMemrefType(tensor);
    if (succeeded(bridged)) {
      int64_t storageElements = succeeded(allocationElements)
                                    ? *allocationElements
                                    : *viewElements;
      auto flatType = MemRefType::get(
          {storageElements}, bridged->getElementType(), AffineMap(),
          bridged->getMemorySpace());
      auto materialized = materializePtrValueAsMemref(
          rewriter, loc, ptr, flatType, tensor.getDefiningOp());
      if (succeeded(materialized))
        flatMemref = *materialized;
    }
  }
  if (flatMemref) {
    auto flatType = dyn_cast<MemRefType>(flatMemref.getType());
    if (!flatType || flatType.getRank() != 1)
      return failure();
    if (flatType.hasStaticShape() && flatType.getDimSize(0) < *viewElements)
      return failure();

    Value viewMemref = flatMemref;
    if (!flatType.hasStaticShape() || flatType.getDimSize(0) != *viewElements) {
      auto viewType = MemRefType::get(
          {*viewElements}, flatType.getElementType(), AffineMap(),
          flatType.getMemorySpace());
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value size = rewriter.create<arith::ConstantIndexOp>(loc, *viewElements);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      viewMemref = rewriter
                       .create<mlir::memref::ReinterpretCastOp>(
                           loc, viewType, flatMemref, zero, ValueRange{size},
                           ValueRange{one})
                       .getResult();
      flatType = viewType;
    }

    auto expandedType = MemRefType::get({shape[0], shape[1]},
                                        flatType.getElementType(), MemRefLayoutAttrInterface{},
                                        flatType.getMemorySpace());
    auto expanded = rewriter
                        .create<mlir::memref::ExpandShapeOp>(
                            loc, expandedType, viewMemref,
                            ArrayRef<ReassociationIndices>{ReassociationIndices{0, 1}})
                        .getResult();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value subview = buildRank2CoordSubview(rewriter, loc, expanded, shape, shape[1], zero, zero);
    if (loweredMemrefByValue)
      (*loweredMemrefByValue)[tensor] = subview;
    return subview;
  }

  auto tileView = tensor.getDefiningOp<::tla::TileViewOp>();
  if (!tileView)
    return failure();
  Value source = tileView.getSource();
  auto baseMemref = materializeBaseMemref(rewriter, loc, source, loweredMemrefByValue);
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
  return buildRank2CoordSubview(rewriter, loc, *baseMemref, shape, rowStride, (*coordPair)[0],
                                (*coordPair)[1]);
}

FailureOr<Value> materializeCopySubview(PatternRewriter &rewriter, Location loc,
                                               Value tensor,
                                               DenseMap<Value, Value> *loweredMemrefByValue,
                                               ArrayRef<int64_t> concreteShape) {
  if (auto subview = materializeCopySubview1D(rewriter, loc, tensor, loweredMemrefByValue);
      succeeded(subview))
    return subview;
  return materializeCopySubviewRank2(rewriter, loc, tensor, loweredMemrefByValue,
                                     concreteShape);
}


// ---------------------------------------------------------------------------
// Descriptor derivation walk: seed root (function-argument) tensors, then a
// pre-order walk over tla.tile_view / tla.make_tensor / tla.make_tensor_like
// producing a TensorDescriptor for each. Populates descriptorByValue. Shared
// so the cube-region pass (and any future consumer) can build descriptors.
// ---------------------------------------------------------------------------

mlir::LogicalResult TlaTensorMemrefLowering::deriveDescriptors(mlir::ModuleOp module) {
  bool derivationFailed = false;
  auto &tensorDescriptorByValue = descriptorByValue;
    auto getOrCreateConstant = [&](Operation *anchor, int64_t value, unsigned bits) -> Value {
      return constants.get(anchor, value, bits);
    };

    auto materializePackedIndexPair = [&](Operation *op, Value packedValue,
                                          StringRef kind) -> FailureOr<std::array<Value, 2>> {
      auto emitPackedError = [&](Twine message) -> FailureOr<std::array<Value, 2>> {
        op->emitError() << message;
        return failure();
      };

      SmallVector<int64_t, 2> leaves;
      if (kind == "shape") {
        auto shapeTy = dyn_cast<::tla::ShapeType>(packedValue.getType());
        if (!shapeTy || failed(::tla::getTlaIndexTreeLeaves(shapeTy.getTree(), leaves))) {
          return emitPackedError("expected flat rank-2 tla.shape operand");
        }
        if (shapeTy.getTree().size() == 1 && leaves.size() == 1) {
          leaves = {1, leaves[0]};
        } else if (shapeTy.getTree().size() != 2) {
          return emitPackedError("expected flat rank-2 tla.shape operand");
        }
      } else {
        auto coordTy = dyn_cast<::tla::CoordType>(packedValue.getType());
        if (!coordTy || failed(::tla::getTlaIndexTreeLeaves(coordTy.getTree(), leaves))) {
          return emitPackedError("expected flat rank-2 tla.coord operand");
        }
        if (coordTy.getTree().size() == 1 && leaves.size() == 1) {
          leaves = {0, leaves[0]};
        } else if (coordTy.getTree().size() != 2) {
          return emitPackedError("expected flat rank-2 tla.coord operand");
        }
      }
      if (leaves.size() != 2) {
        return emitPackedError(Twine("tla.") + kind +
                               " descriptor v1 requires exactly 2 packed elements");
      }

      SmallVector<Value, 2> dynamicValues;
      if (llvm::any_of(leaves, [](int64_t leaf) { return leaf == ShapedType::kDynamic; })) {
        if (kind == "shape") {
          auto makeShape = packedValue.getDefiningOp<::tla::MakeShapeOp>();
          if (!makeShape) {
            return emitPackedError("dynamic tla.shape operands must come from tla.make_shape");
          }
          dynamicValues.append(makeShape.getDynElems().begin(), makeShape.getDynElems().end());
        } else {
          auto makeCoord = packedValue.getDefiningOp<::tla::MakeCoordOp>();
          if (!makeCoord) {
            return emitPackedError("dynamic tla.coord operands must come from tla.make_coord");
          }
          dynamicValues.append(makeCoord.getDynElems().begin(), makeCoord.getDynElems().end());
        }
      }

      std::array<Value, 2> result{};
      size_t dynamicIndex = 0;
      for (auto [index, leaf] : llvm::enumerate(leaves)) {
        if (leaf == ShapedType::kDynamic) {
          if (dynamicIndex >= dynamicValues.size()) {
            return emitPackedError(Twine("packed tla.") + kind +
                                   " type/operand dynamic element count mismatch");
          }
          Value dynamicValue = dynamicValues[dynamicIndex++];
          if (!dynamicValue.getType().isIndex()) {
            return emitPackedError(Twine("tla.") + kind + " dynamic operands must be index type");
          }
          result[index] = dynamicValue;
          continue;
        }

        result[index] = getOrCreateConstant(op, leaf, 0);
      }

      if (dynamicIndex != dynamicValues.size()) {
        return emitPackedError(Twine("packed tla.") + kind +
                               " type/operand dynamic element count mismatch");
      }
      return result;
    };

    auto unpackTileOffsetsAndShape = [&](Operation *op) -> FailureOr<std::array<Value, 4>> {
      Value row;
      Value col;
      Value shape0;
      Value shape1;
      if (op->getNumOperands() == 5) {
        row = op->getOperand(1);
        col = op->getOperand(2);
        shape0 = op->getOperand(3);
        shape1 = op->getOperand(4);
      } else {
        auto shapePair = materializePackedIndexPair(op, op->getOperand(1), "shape");
        if (failed(shapePair))
          return failure();
        shape0 = (*shapePair)[0];
        shape1 = (*shapePair)[1];
        // Prefer the result ``!tla.tensor`` coord segment over ``!tla.coord`` printing:
        // the tensor metadata (e.g. ``0,?`` for outer B along N) stays correct even when
        // the packed coord type string loses ``?`` and would mis-bind dynamic operands.
        if (auto tileOp = dyn_cast<::tla::TileViewOp>(op)) {
          if (auto resTlaTy = dyn_cast<::tla::TlaTensorType>(tileOp.getResult().getType())) {
            SmallVector<int64_t, 4> coordLeaves;
            if (succeeded(
                    ::tla::getTlaIndexTreeLeaves(resTlaTy.getCoord().getTree(), coordLeaves)) &&
                coordLeaves.size() == 2) {
              auto mc = tileOp.getCoord().getDefiningOp<::tla::MakeCoordOp>();
              if (mc) {
                unsigned dynInTensor = 0;
                for (int64_t l : coordLeaves)
                  if (l == ShapedType::kDynamic)
                    ++dynInTensor;
                if (dynInTensor == mc.getNumOperands()) {
                  size_t di = 0;
                  row = coordLeaves[0] == ShapedType::kDynamic
                            ? mc.getDynElems()[di++]
                            : getOrCreateConstant(op, coordLeaves[0], 0);
                  col = coordLeaves[1] == ShapedType::kDynamic
                            ? mc.getDynElems()[di++]
                            : getOrCreateConstant(op, coordLeaves[1], 0);
                  return std::array<Value, 4>{row, col, shape0, shape1};
                }
              }
            }
          }
        }
        auto coordPair = materializePackedIndexPair(op, op->getOperand(2), "coord");
        if (failed(coordPair))
          return failure();
        row = (*coordPair)[0];
        col = (*coordPair)[1];
      }
      return std::array<Value, 4>{row, col, shape0, shape1};
    };

    // Build ``tile_view`` result descriptors for ``!tla.tensor`` sources from operand SSA
    // (shape/coord packs or explicit index operands). Linear layouts
    // (RowMajor/ColumnMajor) may use dynamic shape in the type (``?``) filled from ``sh0/sh1``
    // operands, and dynamic stride (``?``) taken from the parent tile descriptor stride SSA.
    // Packed layouts may also carry dynamic leaves when they can be derived from explicit
    // shape operands or inherited from parent descriptors. Absolute coord and cropped origin
    // follow TLA
    // ``TileViewImpl``: ``abs = parent.abs + tileCoord`` and
    // ``origin_i = min(tileShape_i, parent.origin_i - tileCoord_i)``.
    // buildTileViewResultDescriptorFromParent now lives in the shared header; it
    // takes the constant factory explicitly.
    auto buildTileViewResultDescriptorFromParent =
        [&](Operation *op, Value base, MemRefType bridgedBaseType, const TileTypeInfo &info,
            const TensorDescriptor &parent, Value row, Value col, Value sh0,
            Value sh1) -> FailureOr<TensorDescriptor> {
      return ::tla::buildTileViewResultDescriptorFromParent(op, base, bridgedBaseType, info, parent,
                                                            row, col, sh0, sh1, getOrCreateConstant);
    };

    // Seed descriptors for root tensor-typed function arguments so nested
    // tla.tile_view uses can treat them as full-root views even when the frontend
    // now spells Python tla.Tensor kernel params as !tla.tensor.
    auto seedRootTensorDescriptors = [&](Operation *funcOp, Block &entryBlock) {
      Location loc = funcOp->getLoc();
      OpBuilder builder(&entryBlock, entryBlock.begin());
      auto seedDescriptor = [&](Value descriptorValue, Value baseValue, Type tileType,
                                Type baseType) {
        auto info = decodeTileTypeInfo(tileType);
        if (failed(info))
          return;
        MemRefType bridgedBaseType;
        if (auto memrefType = llvm::dyn_cast<MemRefType>(baseType)) {
          bridgedBaseType = memrefType;
        } else {
          auto bridged = bridgeTlaTensorType(tileType);
          if (failed(bridged))
            return;
          bridgedBaseType = *bridged;
        }
        FailureOr<TensorDescriptor> desc =
            buildTensorDescriptorFromTensorInfo(builder, funcOp, baseValue, bridgedBaseType, *info);
        if (failed(desc))
          return;
        tensorDescriptorByValue[descriptorValue] = *desc;
      };

      for (BlockArgument arg : entryBlock.getArguments()) {
        seedDescriptor(arg, arg, arg.getType(), arg.getType());
      }

      for (Operation &op : entryBlock) {
        auto castOp = llvm::dyn_cast<UnrealizedConversionCastOp>(op);
        if (!castOp || castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
          continue;
        Value base = castOp.getOperand(0);
        if (!llvm::isa<BlockArgument>(base))
          continue;
        seedDescriptor(castOp.getResult(0), base, castOp.getResult(0).getType(), base.getType());
      }
    };

    for (::tla::FuncOp funcOp : module.getOps<::tla::FuncOp>()) {
      if (funcOp.getBody().empty())
        continue;
      seedRootTensorDescriptors(funcOp.getOperation(), funcOp.getBody().front());
    }
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.empty())
        continue;
      seedRootTensorDescriptors(funcOp.getOperation(), funcOp.getBody().front());
    }

    // Stage 0/1: derive descriptors for tile-producing ops in SSA order.
    // Mixed tla.tile_view/tla.make_tensor_like chains rely on
    // producer descriptors being available when their users are visited.
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto tileOp = llvm::dyn_cast<::tla::TileViewOp>(op)) {
        if ((op->getNumOperands() != 5 && op->getNumOperands() != 3) || op->getNumResults() != 1) {
          op->emitError() << "expected tla.tile_view to have exactly 3 or 5 operands and 1 result";
          derivationFailed = true;
          return;
        }

        auto rowColShape = unpackTileOffsetsAndShape(op);
        if (failed(rowColShape)) {
          derivationFailed = true;
          return;
        }
        Value row = (*rowColShape)[0];
        Value col = (*rowColShape)[1];
        Value shape0 = (*rowColShape)[2];
        Value shape1 = (*rowColShape)[3];
        if (!row.getType().isIndex() || !col.getType().isIndex() || !shape0.getType().isIndex() ||
            !shape1.getType().isIndex()) {
          op->emitError() << "tla.tile_view row/col/shape operands must be index type";
          derivationFailed = true;
          return;
        }

        auto resultInfo = decodeTileTypeInfo(tileOp.getResult().getType());
        if (failed(resultInfo)) {
          op->emitError() << "tla.tile_view currently requires a structured tla.tensor result "
                             "type";
          derivationFailed = true;
          return;
        }
        if (resultInfo->rank != 2) {
          op->emitError() << "tla.tile_view descriptor v1 supports only rank-2 tiles";
          derivationFailed = true;
          return;
        }

        Value source = tileOp.getOperand(0);
        if (auto srcMemref = dyn_cast<MemRefType>(source.getType())) {
          FailureOr<MemRefType> bridgedBaseType = srcMemref;

          auto explicitLayout = getExplicitTensorLayoutTagAttr(op);
          if (succeeded(explicitLayout)) {
            if (*explicitLayout != resultInfo->layoutTag) {
              op->emitError() << "tla.tile_view layouttag must match result tensor layout_tag";
              derivationFailed = true;
              return;
            }
          } else if (auto layoutTagAttr = op->getAttrOfType<StringAttr>("layouttag")) {
            op->emitError() << "unsupported tla.tile_view layouttag '" << layoutTagAttr.getValue()
                            << "'";
            derivationFailed = true;
            return;
          }

          OpBuilder builder(op);
          FailureOr<SmallVector<Value, 4>> coordDyn =
              packRank2DynamicMetadataLeaves(op, resultInfo->coordDims, row, col, "coord");
          Value srcDim0 = builder.create<mlir::memref::DimOp>(op->getLoc(), source, 0);
          Value srcDim1 = builder.create<mlir::memref::DimOp>(op->getLoc(), source, 1);
          Value rest0 = builder.create<arith::SubIOp>(op->getLoc(), srcDim0, row);
          Value rest1 = builder.create<arith::SubIOp>(op->getLoc(), srcDim1, col);
          Value origin0Dyn = builder.create<arith::MinSIOp>(op->getLoc(), shape0, rest0);
          Value origin1Dyn = builder.create<arith::MinSIOp>(op->getLoc(), shape1, rest1);
          FailureOr<SmallVector<Value, 4>> originDyn = packRank2DynamicMetadataLeaves(
              op, resultInfo->originShapeDims, origin0Dyn, origin1Dyn, "origin_shape");
          if (failed(coordDyn) || failed(originDyn)) {
            derivationFailed = true;
            return;
          }
          FailureOr<TensorDescriptor> desc = buildTensorDescriptorFromTensorInfo(
              builder, op, source, *bridgedBaseType, *resultInfo, *coordDyn, *originDyn);
          if (failed(desc)) {
            derivationFailed = true;
            return;
          }
          tensorDescriptorByValue[tileOp.getResult()] = *desc;
          return;
        }

        if (llvm::isa<::tla::TlaTensorType>(source.getType())) {
          auto parentIt = tensorDescriptorByValue.find(source);
          if (parentIt == tensorDescriptorByValue.end()) {
            op->emitError()
                << "missing descriptor for tla.tile_view source tile; expected source to be "
                   "produced by tla.tile_view/tla.make_tensor_like in this pass";
            derivationFailed = true;
            return;
          }
          const TensorDescriptor &parent = parentIt->second;
          if (!validateTensorDescriptorV1(
                  op, parent, "malformed parent tensor descriptor for tla.tile_view source tile",
                  /*requireShapeOperands=*/false)) {
            derivationFailed = true;
            return;
          }
          if (resultInfo->rank != parent.rank || resultInfo->addressSpace != parent.addrspace ||
              resultInfo->elementType != parent.elementType) {
            op->emitError() << "tla.tile_view result tile metadata must match parent descriptor "
                               "(rank/element type/addrspace) when source is a tile";
            derivationFailed = true;
            return;
          }

          auto explicitLayout = getExplicitTensorLayoutTagAttr(op);
          if (succeeded(explicitLayout)) {
            if (*explicitLayout != resultInfo->layoutTag) {
              op->emitError() << "tla.tile_view layouttag must match result tensor layout_tag";
              derivationFailed = true;
              return;
            }
          } else if (auto layoutTagAttr = op->getAttrOfType<StringAttr>("layouttag")) {
            op->emitError() << "unsupported tla.tile_view layouttag '" << layoutTagAttr.getValue()
                            << "'";
            derivationFailed = true;
            return;
          }

          auto bridgedParent = dyn_cast<MemRefType>(parent.bridgedBaseMemrefType);
          if (!bridgedParent) {
            op->emitError() << "tla.tile_view parent descriptor missing bridged memref type";
            derivationFailed = true;
            return;
          }
          FailureOr<TensorDescriptor> desc = buildTileViewResultDescriptorFromParent(
              op, parent.base, bridgedParent, *resultInfo, parent, row, col, shape0, shape1);
          if (failed(desc)) {
            derivationFailed = true;
            return;
          }
          tensorDescriptorByValue[tileOp.getResult()] = *desc;
          return;
        }

        op->emitError() << "tla.tile_view source must be either a builtin memref or !tla.tensor";
        derivationFailed = true;
        return;
      }

      if (llvm::isa<::tla::MakeTensorLikeOp>(op)) {
        if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
          op->emitError()
              << "expected tla.make_tensor_like to have exactly 2 operands and 1 result";
          derivationFailed = true;
          return;
        }

        Value ptrValue = op->getOperand(0);
        if (!llvm::isa<::tla::PtrType>(ptrValue.getType())) {
          op->emitError() << "tla.make_tensor_like pointer operand must be !tla.ptr";
          derivationFailed = true;
          return;
        }

        Value likeTile = op->getOperand(1);
        auto parentIt = tensorDescriptorByValue.find(likeTile);
        if (parentIt == tensorDescriptorByValue.end()) {
          op->emitError()
              << "missing descriptor for tla.make_tensor_like reference tile; expected source "
                 "to be produced by tla.tile_view/tla.make_tensor_like in this pass";
          derivationFailed = true;
          return;
        }
        const TensorDescriptor &parent = parentIt->second;
        if (!validateTensorDescriptorV1(
                op, parent,
                "malformed parent tensor descriptor for tla.make_tensor_like reference tile",
                /*requireShapeOperands=*/true)) {
          derivationFailed = true;
          return;
        }

        auto childInfo = decodeTileTypeInfo(op->getResult(0).getType());
        if (failed(childInfo)) {
          op->emitError() << "tla.make_tensor_like currently requires a structured tla.tensor "
                             "result type";
          derivationFailed = true;
          return;
        }
        if (childInfo->rank != parent.rank) {
          op->emitError()
              << "tla.make_tensor_like result tile rank must match reference descriptor "
                 "(rank)";
          derivationFailed = true;
          return;
        }

        int64_t flatElemCount = ShapedType::kDynamic;
        if (auto n = getStaticAllocationElementCount(ptrValue); succeeded(n) && *n > 0) {
          flatElemCount = *n;
        } else if (childInfo->originShapeDims.size() >= 2 &&
                   childInfo->originShapeDims[0] != ShapedType::kDynamic &&
                   childInfo->originShapeDims[1] != ShapedType::kDynamic) {
          int64_t dim0 = childInfo->originShapeDims[0];
          int64_t dim1 = childInfo->originShapeDims[1];
          if (dim0 > 0 && dim1 > 0)
            flatElemCount = dim0 * dim1;
        }
        auto bridgedBaseType =
            buildHivmMemrefType(op->getContext(), {flatElemCount}, childInfo->mlirElementType,
                                childInfo->tlaAddressSpace);
        if (failed(bridgedBaseType)) {
          op->emitError()
              << "tla.make_tensor_like buffer memref must be bridgeable to builtin memref type";
          derivationFailed = true;
          return;
        }

        OpBuilder builder(op);
        auto layoutTagAttr = op->getAttrOfType<StringAttr>("layoutTag");
        if (!layoutTagAttr)
          layoutTagAttr = op->getAttrOfType<StringAttr>("layouttag");
        if (!layoutTagAttr) {
          op->emitError() << "tla.make_tensor_like requires a layoutTag attribute";
          derivationFailed = true;
          return;
        }
        auto layoutTag = parseTensorLayoutTagAttr(layoutTagAttr.getValue());
        if (failed(layoutTag)) {
          op->emitError() << "unsupported tla.make_tensor_like layoutTag '"
                          << layoutTagAttr.getValue() << "'";
          derivationFailed = true;
          return;
        }
        if (*layoutTag != childInfo->layoutTag) {
          op->emitError() << "tla.make_tensor_like layoutTag must match result tensor layout_tag";
          derivationFailed = true;
          return;
        }
        Value typedBuffer = ptrValue;
        auto materializeLeafFromTypeOrParent = [&](int64_t leaf, Value parentValue,
                                                   StringRef fieldName) -> FailureOr<Value> {
          if (leaf == ShapedType::kDynamic) {
            if (parentValue && parentValue.getType().isIndex())
              return parentValue;
            op->emitError() << "dynamic tensor metadata leaf in " << fieldName
                            << " is not supported for tla.make_tensor_like without parent SSA";
            return failure();
          }
          return getOrCreateConstant(op, leaf, 0);
        };
        auto ceilDivIndexByPositiveConst = [&](Value numerator,
                                               int64_t divisor) -> FailureOr<Value> {
          if (divisor <= 0) {
            op->emitError() << "packed shape dynamic leaf requires positive divisor, got "
                            << divisor;
            return failure();
          }
          Value divisorV = getOrCreateConstant(op, divisor, 0);
          Value one = getOrCreateConstant(op, 1, 0);
          Value adjusted = builder.create<arith::AddIOp>(
              op->getLoc(), numerator, builder.create<arith::SubIOp>(op->getLoc(), divisorV, one));
          return builder.create<arith::DivSIOp>(op->getLoc(), adjusted, divisorV).getResult();
        };
        auto materializePackedShapeDynamicLeafFromOrigin = [&](ArrayRef<int64_t> leaves,
                                                               size_t idx) -> FailureOr<Value> {
          // Packed layout shape trees flatten as (m0,m1),(n0,n1).
          // For zN/nZ/zZ/L0C dynamic logical extents live in m1 / n1:
          //   m1 <- ceil_div(origin0, m0), n1 <- ceil_div(origin1, n0).
          // m0 / n0 are layout constants (tile fractal factors), not runtime-varying.
          auto supportedPackedLayout = childInfo->layoutTag == TensorLayoutTag::zN ||
                                       childInfo->layoutTag == TensorLayoutTag::nZ ||
                                       childInfo->layoutTag == TensorLayoutTag::zZ ||
                                       childInfo->layoutTag == TensorLayoutTag::L0C;
          if (!supportedPackedLayout) {
            op->emitError() << "dynamic packed shape leaf at index " << idx
                            << " has no SSA derivation rule for layout "
                            << stringifyTensorLayoutTag(childInfo->layoutTag);
            return failure();
          }
          if (leaves.size() < 4) {
            op->emitError() << "packed shape must have 4 leaves for layout "
                            << stringifyTensorLayoutTag(childInfo->layoutTag);
            return failure();
          }
          if (idx == 1) {
            if (leaves[0] == ShapedType::kDynamic) {
              op->emitError()
                  << "dynamic packed shape leaf index 1 requires static divisor leaf index 0";
              return failure();
            }
            return ceilDivIndexByPositiveConst(parent.originShape0, leaves[0]);
          }
          if (idx == 3) {
            if (leaves[2] == ShapedType::kDynamic) {
              op->emitError()
                  << "dynamic packed shape leaf index 3 requires static divisor leaf index 2";
              return failure();
            }
            return ceilDivIndexByPositiveConst(parent.originShape1, leaves[2]);
          }
          op->emitError() << "dynamic packed shape leaf at index " << idx
                          << " is unsupported; only indices 1 and 3 may be dynamic";
          return failure();
        };
        auto materializePackedLeafFromTypeOrParent = [&](ArrayRef<int64_t> leaves,
                                                         ArrayRef<Value> parentLeaves, size_t idx,
                                                         StringRef fieldName) -> FailureOr<Value> {
          int64_t leaf = leaves[idx];
          if (leaf == ShapedType::kDynamic) {
            if (fieldName == "packed shape") {
              FailureOr<Value> derived = materializePackedShapeDynamicLeafFromOrigin(leaves, idx);
              if (succeeded(derived))
                return *derived;
            }
            if (idx < parentLeaves.size() && parentLeaves[idx] &&
                parentLeaves[idx].getType().isIndex())
              return parentLeaves[idx];
            op->emitError() << "dynamic tensor metadata leaf in " << fieldName
                            << " is not supported for tla.make_tensor_like without parent SSA";
            return failure();
          }
          return getOrCreateConstant(op, leaf, 0);
        };
        auto materializePackedStrideDynamicLeafFromShape = [&](ArrayRef<Value> packedShapeLeaves,
                                                               size_t idx) -> FailureOr<Value> {
          auto mulShapeLeaves = [&](size_t a, size_t b, size_t c) -> FailureOr<Value> {
            if (packedShapeLeaves.size() <= std::max({a, b, c})) {
              op->emitError() << "dynamic packed stride derivation requires packed shape leaves "
                              << a << ", " << b << ", " << c;
              return failure();
            }
            Value ab = builder.create<arith::MulIOp>(op->getLoc(), packedShapeLeaves[a],
                                                     packedShapeLeaves[b]);
            return builder.create<arith::MulIOp>(op->getLoc(), ab, packedShapeLeaves[c])
                .getResult();
          };
          // Layout-coupled packed stride derivation from the remapped fractal shape leaves.
          // zN/L0C: stride[3] = ceil_div_rows * c0 * ele_num_per_c0 = shape[1]*shape[0]*shape[2]
          if ((childInfo->layoutTag == TensorLayoutTag::zN ||
               childInfo->layoutTag == TensorLayoutTag::L0C) &&
              idx == 3) {
            return mulShapeLeaves(/*a=*/1, /*b=*/0, /*c=*/2);
          }
          // nZ/zZ: stride[1] = ceil_div_cols * c0 * ele_num_per_c0 = shape[3]*shape[2]*shape[0]
          if ((childInfo->layoutTag == TensorLayoutTag::nZ ||
               childInfo->layoutTag == TensorLayoutTag::zZ) &&
              idx == 1) {
            return mulShapeLeaves(/*a=*/3, /*b=*/2, /*c=*/0);
          }
          op->emitError() << "dynamic packed stride leaf at index " << idx
                          << " has no SSA derivation rule for layout "
                          << stringifyTensorLayoutTag(childInfo->layoutTag);
          return failure();
        };

        FailureOr<Value> coord0 =
            materializeLeafFromTypeOrParent(childInfo->coordDims[0], parent.rowOffset, "coord");
        FailureOr<Value> coord1 =
            materializeLeafFromTypeOrParent(childInfo->coordDims[1], parent.colOffset, "coord");
        FailureOr<Value> origin0 = materializeLeafFromTypeOrParent(
            childInfo->originShapeDims[0], parent.originShape0, "origin_shape");
        FailureOr<Value> origin1 = materializeLeafFromTypeOrParent(
            childInfo->originShapeDims[1], parent.originShape1, "origin_shape");
        if (failed(coord0) || failed(coord1) || failed(origin0) || failed(origin1)) {
          derivationFailed = true;
          return;
        }

        Value stride0;
        Value stride1;
        Value shape0;
        Value shape1;
        SmallVector<Value, 4> packedShape;
        SmallVector<Value, 4> packedStride;

        if (isLinearLayout(childInfo->layoutTag)) {
          FailureOr<Value> shape0Or =
              materializeLeafFromTypeOrParent(childInfo->shapeDims[0], parent.shape0, "shape");
          FailureOr<Value> shape1Or =
              materializeLeafFromTypeOrParent(childInfo->shapeDims[1], parent.shape1, "shape");
          FailureOr<Value> stride0Or =
              materializeLeafFromTypeOrParent(childInfo->strideDims[0], parent.stride0, "stride");
          FailureOr<Value> stride1Or =
              materializeLeafFromTypeOrParent(childInfo->strideDims[1], parent.stride1, "stride");
          if (failed(shape0Or) || failed(shape1Or) || failed(stride0Or) || failed(stride1Or)) {
            derivationFailed = true;
            return;
          }
          shape0 = *shape0Or;
          shape1 = *shape1Or;
          stride0 = *stride0Or;
          stride1 = *stride1Or;
        } else {
          if (!isPackedLayout(childInfo->layoutTag)) {
            op->emitError() << "unsupported tla.make_tensor_like layout for descriptor v1";
            derivationFailed = true;
            return;
          }
          packedShape.reserve(childInfo->shapeDims.size());
          packedStride.reserve(childInfo->strideDims.size());
          for (size_t i = 0; i < childInfo->shapeDims.size(); ++i) {
            FailureOr<Value> leaf = materializePackedLeafFromTypeOrParent(
                childInfo->shapeDims, parent.packedShape, i, "packed shape");
            if (failed(leaf)) {
              derivationFailed = true;
              return;
            }
            packedShape.push_back(*leaf);
          }
          for (size_t i = 0; i < childInfo->strideDims.size(); ++i) {
            FailureOr<Value> leaf;
            bool dynamicStrideLeaf = childInfo->strideDims[i] == ShapedType::kDynamic;
            bool packedStrideUnavailable = i >= parent.packedStride.size();
            bool layoutChanged = parent.layoutTag != childInfo->layoutTag;
            if (dynamicStrideLeaf && (packedStrideUnavailable || layoutChanged)) {
              leaf = materializePackedStrideDynamicLeafFromShape(packedShape, i);
            } else {
              leaf = materializePackedLeafFromTypeOrParent(childInfo->strideDims,
                                                           parent.packedStride, i, "packed stride");
            }
            if (failed(leaf)) {
              derivationFailed = true;
              return;
            }
            packedStride.push_back(*leaf);
          }
          shape0 = *origin0;
          shape1 = *origin1;
          stride0 = packedStride[0];
          stride1 = packedStride[1];
        }

        TensorDescriptor desc{
            typedBuffer,
            *bridgedBaseType,
            *coord0,
            *coord1,
            stride0,
            stride1,
            shape0,
            shape1,
            *origin0,
            *origin1,
            *coord0,
            *coord1,
            childInfo->layoutTag,
            childInfo->addressSpace,
            childInfo->elementType,
            childInfo->rank,
            std::move(packedShape),
            std::move(packedStride),
        };
        tensorDescriptorByValue[op->getResult(0)] = std::move(desc);
        return;
      }

      if (llvm::isa<::tla::MakeTensorOp>(op)) {
        if (op->getNumOperands() != 3 || op->getNumResults() != 1) {
          op->emitError()
              << "expected tla.make_tensor to have exactly 3 operands and 1 result";
          derivationFailed = true;
          return;
        }

        Value ptrValue = op->getOperand(0);
        if (!llvm::isa<::tla::PtrType>(ptrValue.getType())) {
          op->emitError() << "tla.make_tensor pointer operand must be !tla.ptr";
          derivationFailed = true;
          return;
        }
        Value layoutValue = op->getOperand(1);
        Value coordValue = op->getOperand(2);
        auto makeLayout = layoutValue.getDefiningOp<::tla::MakeLayoutOp>();
        if (!makeLayout) {
          op->emitError()
              << "tla.make_tensor layout operand must come from tla.make_layout";
          derivationFailed = true;
          return;
        }

        auto childInfo = decodeTileTypeInfo(op->getResult(0).getType());
        if (failed(childInfo)) {
          op->emitError() << "tla.make_tensor currently requires a structured tla.tensor "
                             "result type";
          derivationFailed = true;
          return;
        }
        if (!isLinearLayout(childInfo->layoutTag)) {
          op->emitError() << "tla.make_tensor currently supports only linear layouts "
                             "(RowMajor/ColumnMajor); packed layouts require make_tensor_like";
          derivationFailed = true;
          return;
        }

        // Buffer element count for the synthetic !tla.memref type: prefer a static 1D
        // length from an HIVM pointer-cast bridge (allocator-backed ptr), else multiply
        // the first two origin_shape dims (e.g. inttoptr-backed ptr with static layout).
        int64_t flatElemCount = ShapedType::kDynamic;
        if (auto n = getStaticAllocationElementCount(ptrValue); succeeded(n) && *n > 0) {
          flatElemCount = *n;
        } else if (childInfo->originShapeDims.size() >= 2 &&
                   childInfo->originShapeDims[0] != ShapedType::kDynamic &&
                   childInfo->originShapeDims[1] != ShapedType::kDynamic) {
          int64_t dim0 = childInfo->originShapeDims[0];
          int64_t dim1 = childInfo->originShapeDims[1];
          if (dim0 > 0 && dim1 > 0)
            flatElemCount = dim0 * dim1;
        }
        auto bridgedBaseType =
            buildHivmMemrefType(op->getContext(), {flatElemCount}, childInfo->mlirElementType,
                                childInfo->tlaAddressSpace);
        if (failed(bridgedBaseType)) {
          op->emitError()
              << "tla.make_tensor buffer memref must be bridgeable to builtin memref type";
          derivationFailed = true;
          return;
        }

        Value typedBuffer = ptrValue;

        // Materialize index-tree leaves from the operand defining ops. Static leaves
        // become constants; dynamic leaves are pulled from tla.make_shape/make_stride/
        // make_coord dyn-elems in leaf order. A derived dynamic leaf that is not directly
        // operand-backed (e.g. rank-1 linear stride0 = extent*stride with dynamic extent)
        // is rejected with a clear error. ``childInfo`` already promotes rank-1 linear to
        // rank-2, so the leading synthetic ``1``/``0`` leaves are static here.
        auto materializeLeaves = [&](Value packedValue, ArrayRef<int64_t> leaves,
                                     StringRef kind) -> FailureOr<SmallVector<Value, 4>> {
          SmallVector<Value, 4> result;
          unsigned dynLeafCount = 0;
          for (int64_t leaf : leaves)
            if (leaf == ShapedType::kDynamic)
              ++dynLeafCount;
          SmallVector<Value, 4> dynElems;
          if (dynLeafCount > 0) {
            if (kind == "shape") {
              if (auto ms = packedValue.getDefiningOp<::tla::MakeShapeOp>())
                dynElems.append(ms.getDynElems().begin(), ms.getDynElems().end());
            } else if (kind == "stride") {
              if (auto mst = packedValue.getDefiningOp<::tla::MakeStrideOp>())
                dynElems.append(mst.getDynElems().begin(), mst.getDynElems().end());
            } else {
              if (auto mc = packedValue.getDefiningOp<::tla::MakeCoordOp>())
                dynElems.append(mc.getDynElems().begin(), mc.getDynElems().end());
            }
            if (dynElems.size() < dynLeafCount) {
              op->emitError()
                  << "tla.make_tensor " << kind
                  << " has a derived dynamic leaf that is not directly operand-backed "
                     "(e.g. rank-1 stride with dynamic extent); pass explicit leaves via tla.make_"
                  << kind;
              return failure();
            }
          }
          size_t di = 0;
          for (int64_t leaf : leaves) {
            if (leaf == ShapedType::kDynamic) {
              Value dv = dynElems[di++];
              if (!dv.getType().isIndex()) {
                op->emitError() << "tla.make_tensor " << kind
                                << " dynamic operands must be index type";
                return failure();
              }
              result.push_back(dv);
            } else {
              result.push_back(getOrCreateConstant(op, leaf, 0));
            }
          }
          return result;
        };

        auto shapeLeaves =
            materializeLeaves(makeLayout.getShape(), childInfo->shapeDims, "shape");
        auto strideLeaves =
            materializeLeaves(makeLayout.getStride(), childInfo->strideDims, "stride");
        auto coordLeaves = materializeLeaves(coordValue, childInfo->coordDims, "coord");
        if (failed(shapeLeaves) || failed(strideLeaves) || failed(coordLeaves)) {
          derivationFailed = true;
          return;
        }
        Value shape0 = (*shapeLeaves)[0];
        Value shape1 = (*shapeLeaves)[1];
        Value stride0 = (*strideLeaves)[0];
        Value stride1 = (*strideLeaves)[1];
        Value coord0 = (*coordLeaves)[0];
        Value coord1 = (*coordLeaves)[1];

        // Origin defaults to shape; honor an explicit make_layout origin operand.
        Value origin0 = shape0;
        Value origin1 = shape1;
        if (Value originOperand = makeLayout.getOriginShape()) {
          auto originLeaves =
              materializeLeaves(originOperand, childInfo->originShapeDims, "shape");
          if (failed(originLeaves)) {
            derivationFailed = true;
            return;
          }
          origin0 = (*originLeaves)[0];
          origin1 = (*originLeaves)[1];
        }

        SmallVector<Value, 4> packedShape;
        SmallVector<Value, 4> packedStride;
        TensorDescriptor desc{
            typedBuffer,
            *bridgedBaseType,
            coord0,
            coord1,
            stride0,
            stride1,
            shape0,
            shape1,
            origin0,
            origin1,
            coord0,
            coord1,
            childInfo->layoutTag,
            childInfo->addressSpace,
            childInfo->elementType,
            childInfo->rank,
            std::move(packedShape),
            std::move(packedStride),
        };
        tensorDescriptorByValue[op->getResult(0)] = std::move(desc);
        return;
      }
    });
  return derivationFailed ? failure() : success();
}


mlir::LogicalResult TlaTensorMemrefLowering::lowerTileProducerToSubview(
    mlir::Operation *op, mlir::PatternRewriter &rewriter, AllocatorOffsetState *allocatorState) {
  if (op->getNumResults() != 1) {
    op->emitError() << "expected tile-view op to have exactly 1 result during subview lowering";
    return failure();
  }
  auto descIt = descriptorByValue.find(op->getResult(0));
  if (descIt == descriptorByValue.end()) {
    op->emitError() << "missing descriptor for " << op->getName().getStringRef()
                    << " result during subview lowering";
    return failure();
  }
  const TensorDescriptor &desc = descIt->second;
  if (!validateTensorDescriptorV1(op, desc, "malformed descriptor for tile subview lowering",
                                  /*requireShapeOperands=*/true)) {
    return failure();
  }

  FailureOr<Value> baseMemref = getOrMaterializeDescriptorBaseMemref(
      rewriter, op->getLoc(), desc, allocatorState, op, loweredMemrefByValue);
  if (failed(baseMemref))
    return failure();
  auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
  if (!baseType) {
    op->emitError() << "expected descriptor base to materialize to memref type";
    return failure();
  }

  Value one = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);
  Value subview;
  if (baseType.getRank() == 1) {
    // Flattened 1D buffers are already in the runtime ABI shape expected by
    // downstream lowering; avoid fabricating extra subviews that later pass
    // stages will erase anyway.
    subview = *baseMemref;
  } else {
    subview = rewriter
                  .create<mlir::memref::SubViewOp>(
                      op->getLoc(), *baseMemref, ValueRange{desc.rowOffset, desc.colOffset},
                      ValueRange{desc.shape0, desc.shape1}, ValueRange{one, one})
                  .getResult();
  }
  Value tileView =
      rewriter
          .create<UnrealizedConversionCastOp>(op->getLoc(), TypeRange{op->getResult(0).getType()},
                                              ValueRange{subview})
          .getResult(0);
  descriptorByValue[tileView] = desc;
  descriptorByValue.erase(op->getResult(0));
  rewriter.replaceOp(op, tileView);
  return success();
}

mlir::LogicalResult TlaTensorMemrefLowering::lowerTileProducers(
    mlir::ModuleOp module, AllocatorOffsetState *allocatorState,
    llvm::SmallVectorImpl<mlir::Operation *> &toErase) {
  stageDeadTileProducers(module, toErase);

  SmallVector<Operation *, 8> tileViewOps;
  module.walk([&](Operation *op) {
    if (llvm::isa<::tla::TileViewOp, ::tla::MakeTensorLikeOp, ::tla::MakeTensorOp>(op))
      tileViewOps.push_back(op);
  });
  for (Operation *op : tileViewOps) {
    if (!op || !op->getBlock() || llvm::is_contained(toErase, op))
      continue;
    PatternRewriter rewriter(op->getContext());
    rewriter.setInsertionPoint(op);
    if (failed(lowerTileProducerToSubview(op, rewriter, allocatorState)))
      return failure();
  }

  // Erase dead tile-construction handles after loop conversion.
  SmallVector<Operation *, 8> deadMakeTensorLikeOps;
  module.walk([&](::tla::MakeTensorLikeOp op) {
    if (op.getResult().use_empty())
      deadMakeTensorLikeOps.push_back(op.getOperation());
  });
  for (Operation *o : deadMakeTensorLikeOps) {
    if (!o->getBlock() || llvm::is_contained(toErase, o))
      continue;
    o->erase();
  }
  SmallVector<Operation *, 8> deadMakeTensorOps;
  module.walk([&](::tla::MakeTensorOp op) {
    if (op.getResult().use_empty())
      deadMakeTensorOps.push_back(op.getOperation());
  });
  for (Operation *o : deadMakeTensorOps) {
    if (!o->getBlock() || llvm::is_contained(toErase, o))
      continue;
    o->erase();
  }
  return success();
}

} // namespace tla
