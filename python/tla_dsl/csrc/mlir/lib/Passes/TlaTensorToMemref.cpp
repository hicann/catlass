#include "Passes/TlaTensorToMemref.h"

#include "PassesCommon.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

namespace tla {

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

void stageDeadTensorDescriptors(mlir::Operation *root,
                            llvm::SmallVectorImpl<mlir::Operation *> &toErase) {
  bool progress = true;
  while (progress) {
    progress = false;
    SmallVector<Operation *, 8> newlyDead;
    root->walk([&](Operation *op) {
      if (!llvm::isa<::tla::TensorDescOp>(op) ||
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

  if (diagnosticOp)
    diagnosticOp->emitError() << "expected tla.tensor_desc base to be memref or !tla.ptr";
  return failure();
}

mlir::FailureOr<mlir::Value> materializeTileMemrefFromDescriptor(
    mlir::OpBuilder &builder, mlir::Location loc, const TensorDescriptor &desc,
    mlir::Operation *diagnosticOp,
    llvm::DenseMap<mlir::Value, mlir::Value> &baseMemrefCache) {
  FailureOr<Value> baseMemref = getOrMaterializeDescriptorBaseMemref(
      builder, loc, desc, diagnosticOp, baseMemrefCache);
  if (failed(baseMemref))
    return failure();
  auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
  if (!baseType) {
    if (diagnosticOp)
      diagnosticOp->emitError() << "expected descriptor base to materialize to memref type";
    return failure();
  }
  // Flattened 1D buffers are already in the runtime ABI shape expected by
  // downstream lowering; avoid fabricating subviews that downstream passes
  // erase anyway.
  if (baseType.getRank() == 1)
    return *baseMemref;
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  return builder
      .create<mlir::memref::SubViewOp>(loc, *baseMemref,
                                       ValueRange{desc.rowOffset, desc.colOffset},
                                       ValueRange{desc.shape0, desc.shape1}, ValueRange{one, one})
      .getResult();
}

mlir::FailureOr<mlir::Value>
getOrMaterializeDescriptorBaseMemref(mlir::OpBuilder &builder, mlir::Location loc,
                                     const TensorDescriptor &desc,
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
    return materializeDescriptorBaseMemref(builder, loc, desc, diagnosticOp);

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
      materializeDescriptorBaseMemref(builder, loc, desc, diagnosticOp);
  if (failed(materialized))
    return failure();
  baseMemrefCache[desc.base] = *materialized;
  return *materialized;
}


// ---------------------------------------------------------------------------
// Vector tile memref materialization (TlaVectorRegionPass).
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


// The `!tla.ptr` operand of a make_tensor / make_tensor_like tensor (or the base
// of a tla.tensor_desc when it is a ptr), or null.
// The !tla.ptr base of a tile. tla-lower-tensor-desc is the sole descriptor
// producer, so every tile here is a tla.tensor_desc; return its base when it is
// a !tla.ptr (the inttoptr boundary left by tla-lower-ptr), else null.
static Value ptrOfTensorDesc(Value tensor) {
  if (auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>())
    return llvm::isa<::tla::PtrType>(descOp.getBase().getType()) ? descOp.getBase() : Value();
  return {};
}

FailureOr<MemRefType> getVectorHelperArgMemrefType(Value operand) {
  Value ptr = ptrOfTensorDesc(operand);
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
  if (loweredMemrefByValue) {
    auto it = loweredMemrefByValue->find(tensor);
    if (it != loweredMemrefByValue->end() && it->second)
      return castToBridgedType(rewriter, loc, it->second, tensor);
  }

  if (auto castOp = tensor.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp.getNumOperands() == 1 && isa<MemRefType>(castOp.getOperand(0).getType()))
      return castToBridgedType(rewriter, loc, castOp.getOperand(0), tensor);
  }

  // A descriptor backed by a kernel-argument memref views that memref directly.
  if (auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
      descOp && isa<MemRefType>(descOp.getBase().getType()))
    return castToBridgedType(rewriter, loc, descOp.getBase(), tensor);

  // tla.tensor_desc (the sole tile producer after tla-lower-tensor-desc): the
  // tensor views its ptr's address. The ptr is the inttoptr boundary left by
  // tla-lower-ptr (any ptr_add / tensor_ptr offset was already folded into the
  // byte address), so materialize it directly via materializePtrValueAsMemref.
  if (isa_and_nonnull<::tla::TensorDescOp>(tensor.getDefiningOp())) {
    Value ptr = ptrOfTensorDesc(tensor);
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

  if (isa<MemRefType>(tensor.getType()))
    return tensor;

  return failure();
}

// Build a rank-1, `numElements`-wide reinterpret_cast of `baseMemref` at element
// `offset` (dynamic stride-1 layout). Single flat-tile subview constructor shared
// by the copy-subview lowering and the vector helper's per-lane tiles.
Value materializeFlatReinterpretSubview(OpBuilder &builder, Location loc, Value baseMemref,
                                        Value offset, int64_t numElements) {
  auto baseType = cast<MemRefType>(baseMemref.getType());
  auto layout = StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic,
                                       ArrayRef<int64_t>{1});
  auto tileType = MemRefType::get({numElements}, baseType.getElementType(), layout,
                                  baseType.getMemorySpace());
  Value size = builder.create<arith::ConstantIndexOp>(loc, numElements);
  Value stride = builder.create<arith::ConstantIndexOp>(loc, 1);
  return builder
      .create<mlir::memref::ReinterpretCastOp>(loc, tileType, baseMemref, offset,
                                               ValueRange{size}, ValueRange{stride})
      .getResult();
}

static FailureOr<Value> materializeCopySubview1D(PatternRewriter &rewriter, Location loc,
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

  int64_t subviewOffset = baseType.getDimSize(0) == info->shape[0] ? 0 : info->coord[0];
  if (subviewOffset == 0 && baseType.hasStaticShape() && baseType.getDimSize(0) == info->shape[0])
    return *baseMemref;

  Value offset = rewriter.create<arith::ConstantIndexOp>(loc, subviewOffset);
  return materializeFlatReinterpretSubview(rewriter, loc, *baseMemref, offset, info->shape[0]);
}

static FailureOr<Value> materializeCopySubviewRank2(
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

  // tla.tensor_desc: materialize the base memref (inttoptr ptr-backed flat buffer
  // or a kernel-arg memref) and take a coord-aware rank-2 subview at the
  // descriptor's (row_offset, col_offset). Handles both a rank-1 flat base
  // (reinterpret_cast with the descriptor's strides) and a rank-2 base (subview).
  if (auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>()) {
    Value base;
    Value ptr = ptrOfTensorDesc(tensor); // descOp.getBase() when it is a !tla.ptr
    if (ptr && ptr.getDefiningOp<::tla::IntToPtrOp>()) {
      auto allocationElements = getStaticAllocationElementCount(ptr);
      auto bridged = getBridgedTensorMemrefType(tensor);
      if (failed(bridged))
        return failure();
      int64_t storageElements = succeeded(allocationElements) ? *allocationElements
                                                              : *viewElements;
      auto flatType = MemRefType::get(
          {storageElements}, bridged->getElementType(), AffineMap(),
          bridged->getMemorySpace());
      auto materialized = materializePtrValueAsMemref(
          rewriter, loc, ptr, flatType, tensor.getDefiningOp());
      if (failed(materialized))
        return failure();
      base = *materialized;
    } else if (isa<MemRefType>(descOp.getBase().getType())) {
      base = descOp.getBase(); // kernel-arg memref base
    } else {
      return failure();
    }

    auto baseType = dyn_cast<MemRefType>(base.getType());
    if (!baseType)
      return failure();
    if (baseType.getRank() == 1) {
      // Flat ptr-backed base (UB/L1): reinterpret the tile shape directly at the
      // descriptor's (row,col) offset with the descriptor's strides. Unlike an
      // expand_shape to the tile shape, this views a sub-tile of a larger
      // allocation correctly -- no element-count constraint, and the real row
      // stride (full buffer width) comes from the descriptor, not the tile width.
      Value rowStride = descOp.getStride0();
      Value colStride = descOp.getStride1();
      Value flatOffset = rewriter.create<arith::AddIOp>(
          loc, rewriter.create<arith::MulIOp>(loc, descOp.getRowOffset(), rowStride),
          descOp.getColOffset());
      auto layout = StridedLayoutAttr::get(
          rewriter.getContext(), ShapedType::kDynamic,
          ArrayRef<int64_t>{ShapedType::kDynamic, ShapedType::kDynamic});
      auto tileType =
          MemRefType::get(shape, baseType.getElementType(), layout, baseType.getMemorySpace());
      Value s0 = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
      Value s1 = rewriter.create<arith::ConstantIndexOp>(loc, shape[1]);
      Value tile = rewriter
                       .create<mlir::memref::ReinterpretCastOp>(
                           loc, tileType, base, flatOffset, ValueRange{s0, s1},
                           ValueRange{rowStride, colStride})
                       .getResult();
      if (loweredMemrefByValue)
        (*loweredMemrefByValue)[tensor] = tile;
      return tile;
    }
    if (baseType.getRank() != 2)
      return failure();
    int64_t rowStride = baseType.hasStaticShape() ? baseType.getDimSize(1) : info->strides[0];
    if (rowStride <= 0 || rowStride == ShapedType::kDynamic)
      rowStride = ShapedType::kDynamic;
    Value subview = buildRank2CoordSubview(rewriter, loc, base, shape, rowStride,
                                           descOp.getRowOffset(), descOp.getColOffset());
    if (loweredMemrefByValue)
      (*loweredMemrefByValue)[tensor] = subview;
    return subview;
  }

  // Every rank-2 copy operand is a tla.tensor_desc (produced by tla-lower-tensor-desc)
  // and handled above; nothing else reaches here.
  return failure();
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
// Producer-side descriptor derivation owned by tla-lower-tensor-desc: seed root
// tensor values, then walk tla.tile_view / tla.make_tensor / tla.make_tensor_like
// in pre-order and produce a TensorDescriptor for every result. Downstream passes
// read the materialized tla.tensor_desc ops instead of invoking this walk.
// ---------------------------------------------------------------------------

// Descriptor consumers only read materialized tla.tensor_desc operations.
mlir::LogicalResult collectMaterializedTensorDescriptors(
    mlir::func::FuncOp funcOp,
    llvm::DenseMap<mlir::Value, TensorDescriptor> &descriptorByValue) {
  descriptorByValue.clear();
  bool collectionFailed = false;
  funcOp.walk([&](Operation *op) {
    if (auto descOp = llvm::dyn_cast<::tla::TensorDescOp>(op)) {
      auto desc = descriptorFromTensorDescOp(descOp);
      if (failed(desc)) {
        collectionFailed = true;
        return;
      }
      descriptorByValue[descOp.getResult()] = *desc;
      return;
    }

    if (llvm::isa<::tla::TileViewOp, ::tla::MakeTensorOp, ::tla::MakeTensorLikeOp>(op)) {
      op->emitError("raw tensor view producer reached a descriptor consumer; "
                    "expected tla-lower-tensor-desc to materialize tla.tensor_desc");
      collectionFailed = true;
    }
  });
  return failure(collectionFailed);
}

// ---------------------------------------------------------------------------
// Copy-route runtime lowering (shared by tla-cube-region / tla-vector-region).
// ---------------------------------------------------------------------------

static mlir::FailureOr<hivm::AddressSpace> resolveHivmAddressSpace(MLIRContext *ctx,
                                                                   StringRef addressSpace) {
  auto tlaAddressSpace = symbolizeAddressSpace(addressSpace);
  if (!tlaAddressSpace)
    return failure();
  FailureOr<Attribute> memorySpaceOr = mapTlaAddressSpaceToHivmMemspace(ctx, *tlaAddressSpace);
  if (failed(memorySpaceOr))
    return failure();
  auto memorySpaceAttr = dyn_cast<hivm::AddressSpaceAttr>(*memorySpaceOr);
  if (!memorySpaceAttr)
    return failure();
  return memorySpaceAttr.getAddressSpace();
}

static StringRef copyRuntimeElemSuffix(StringRef elementType) {
  if (elementType == "f32")
    return "float";
  if (elementType == "f16")
    return "half";
  if (elementType == "bf16")
    return "bf16";
  return {};
}

std::string getCopyRouteCallee(MLIRContext *ctx, StringRef srcAddrspace, StringRef dstAddrspace,
                               TensorLayoutTag srcLayout, TensorLayoutTag dstLayout,
                               StringRef srcElementType, StringRef dstElementType,
                               StringRef extraDesc) {
  FailureOr<hivm::AddressSpace> srcSpace = resolveHivmAddressSpace(ctx, srcAddrspace);
  FailureOr<hivm::AddressSpace> dstSpace = resolveHivmAddressSpace(ctx, dstAddrspace);
  if (failed(srcSpace) || failed(dstSpace))
    return {};
  StringRef dstElem = dstElementType.empty() ? srcElementType : dstElementType;

  // Copy routing is keyed by explicit (addrspace, layout-tag) pairs. Runtime
  // symbol names encode both endpoint layout tags so future layout variants can
  // be added as new explicit routes instead of overloading addrspace-only names.
  if (*srcSpace == hivm::AddressSpace::UB && *dstSpace == hivm::AddressSpace::L1 &&
      srcLayout == TensorLayoutTag::RowMajor && dstLayout == TensorLayoutTag::zN) {
    if (srcElementType != dstElem)
      return {};
    StringRef suffix = copyRuntimeElemSuffix(srcElementType);
    if (suffix.empty())
      return {};
    return Twine("copy_ubuf_row_major_to_cbuf_zN_").concat(suffix).str();
  }
  if (*srcSpace == hivm::AddressSpace::GM && *dstSpace == hivm::AddressSpace::L1 &&
      srcLayout == TensorLayoutTag::RowMajor && dstLayout == TensorLayoutTag::zN) {
    if (srcElementType != dstElem)
      return {};
    StringRef suffix = copyRuntimeElemSuffix(srcElementType);
    if (suffix.empty())
      return {};
    return Twine("copy_gm_row_major_to_cbuf_zN_").concat(suffix).str();
  }
  if (*srcSpace == hivm::AddressSpace::GM && *dstSpace == hivm::AddressSpace::L1 &&
      srcLayout == TensorLayoutTag::ColumnMajor && dstLayout == TensorLayoutTag::nZ) {
    if (srcElementType != dstElem)
      return {};
    StringRef suffix = copyRuntimeElemSuffix(srcElementType);
    if (suffix.empty())
      return {};
    return Twine("copy_gm_column_major_to_cbuf_nZ_").concat(suffix).str();
  }
  if (*srcSpace == hivm::AddressSpace::L1 && *dstSpace == hivm::AddressSpace::L0A &&
      srcLayout == TensorLayoutTag::zN && dstLayout == TensorLayoutTag::zN) {
    if (srcElementType != dstElem)
      return {};
    StringRef suffix = copyRuntimeElemSuffix(srcElementType);
    if (suffix.empty())
      return {};
    return Twine("copy_cbuf_zN_to_ca_zN_").concat(suffix).str();
  }
  if (*srcSpace == hivm::AddressSpace::L1 && *dstSpace == hivm::AddressSpace::L0A &&
      srcLayout == TensorLayoutTag::nZ && dstLayout == TensorLayoutTag::zN) {
    if (srcElementType != dstElem)
      return {};
    StringRef suffix = copyRuntimeElemSuffix(srcElementType);
    if (suffix.empty())
      return {};
    return Twine("copy_cbuf_nZ_to_ca_zN_").concat(suffix).str();
  }
  if (*srcSpace == hivm::AddressSpace::L1 && *dstSpace == hivm::AddressSpace::L0B &&
      srcLayout == TensorLayoutTag::zN && dstLayout == TensorLayoutTag::nZ) {
    if (srcElementType != dstElem)
      return {};
    StringRef suffix = copyRuntimeElemSuffix(srcElementType);
    if (suffix.empty())
      return {};
    return Twine("copy_cbuf_zN_to_cb_nZ_").concat(suffix).str();
  }
  if (*srcSpace == hivm::AddressSpace::L1 && *dstSpace == hivm::AddressSpace::L0B &&
      srcLayout == TensorLayoutTag::nZ && dstLayout == TensorLayoutTag::nZ) {
    if (srcElementType != dstElem)
      return {};
    StringRef suffix = copyRuntimeElemSuffix(srcElementType);
    if (suffix.empty())
      return {};
    return Twine("copy_cbuf_nZ_to_cb_nZ_").concat(suffix).str();
  }
  // L0C (fp32 MMAD acc) -> GM row-major: dst may be f32 / f16 / bf16 (narrowing on fixpipe).
  if (*srcSpace == hivm::AddressSpace::L0C && *dstSpace == hivm::AddressSpace::GM &&
      srcLayout == TensorLayoutTag::L0C && dstLayout == TensorLayoutTag::RowMajor) {
    if (srcElementType != "f32")
      return {};
    StringRef suffix = copyRuntimeElemSuffix(dstElem);
    if (suffix.empty())
      return {};
    return Twine("copy_cc_to_gm_row_major_").concat(suffix).str();
  }
  // L0C (fp32 MMAD acc) -> UB row-major: dst may be f32 / f16 / bf16 (narrowing on fixpipe).
  if (*srcSpace == hivm::AddressSpace::L0C && *dstSpace == hivm::AddressSpace::UB &&
      srcLayout == TensorLayoutTag::L0C && dstLayout == TensorLayoutTag::RowMajor) {
    if (srcElementType != "f32")
      return {};
    StringRef suffix = copyRuntimeElemSuffix(dstElem);
    if (suffix.empty())
      return {};
    return Twine("copy_cc_to_ubuf_row_major_").concat(extraDesc).concat("_").concat(suffix).str();
  }
  return {};
}

static SmallVector<Value, 8> buildRowMajorCopyPayload(OpBuilder &builder, Location loc,
                                                      const TensorDescriptor &desc) {
  return {
      castValueToI64(builder, loc, desc.shape0),
      castValueToI64(builder, loc, desc.shape1),
      castValueToI64(builder, loc, desc.stride0),
      castValueToI64(builder, loc, desc.stride1),
      castValueToI64(builder, loc, desc.absCoord0),
      castValueToI64(builder, loc, desc.absCoord1),
      castValueToI64(builder, loc, desc.originShape0),
      castValueToI64(builder, loc, desc.originShape1),
  };
}

static SmallVector<Value, 12> buildPackedCopyPayload(OpBuilder &builder, Location loc,
                                                     const TensorDescriptor &desc) {
  return {
      castValueToI64(builder, loc, desc.packedShape[0]),
      castValueToI64(builder, loc, desc.packedShape[1]),
      castValueToI64(builder, loc, desc.packedShape[2]),
      castValueToI64(builder, loc, desc.packedShape[3]),
      castValueToI64(builder, loc, desc.packedStride[0]),
      castValueToI64(builder, loc, desc.packedStride[1]),
      castValueToI64(builder, loc, desc.packedStride[2]),
      castValueToI64(builder, loc, desc.packedStride[3]),
      castValueToI64(builder, loc, desc.rowOffset),
      castValueToI64(builder, loc, desc.colOffset),
      castValueToI64(builder, loc, desc.originShape0),
      castValueToI64(builder, loc, desc.originShape1),
  };
}

SmallVector<Value, 20> buildCopyPayloadForRoute(OpBuilder &builder, Location loc,
                                                const TensorDescriptor &srcDesc,
                                                const TensorDescriptor &dstDesc) {
  SmallVector<Value, 20> payload;
  auto append = [&](ArrayRef<Value> values) { payload.append(values.begin(), values.end()); };
  if (isLinearLayout(srcDesc.layoutTag))
    append(buildRowMajorCopyPayload(builder, loc, srcDesc));
  else
    append(buildPackedCopyPayload(builder, loc, srcDesc));

  if (isLinearLayout(dstDesc.layoutTag))
    append(buildRowMajorCopyPayload(builder, loc, dstDesc));
  else
    append(buildPackedCopyPayload(builder, loc, dstDesc));
  return payload;
}

static bool isAicTemplateRuntimeCall(StringRef name) {
  if (name == "mmad_float_float_float" || name == "mmad_half_half_float" ||
      name == "mmad_bf16_bf16_float")
    return true;
  if (!(name.starts_with("copy_")))
    return false;
  if (!(name.ends_with("_float") || name.ends_with("_half") || name.ends_with("_bf16")))
    return false;
  return name.starts_with("copy_gm_row_major_to_cbuf_zN_") ||
         name.starts_with("copy_gm_column_major_to_cbuf_nZ_") ||
         name.starts_with("copy_cbuf_zN_to_ca_zN_") ||
         name.starts_with("copy_cbuf_nZ_to_ca_zN_") ||
         name.starts_with("copy_cbuf_zN_to_cb_nZ_") ||
         name.starts_with("copy_cbuf_nZ_to_cb_nZ_") ||
         name.starts_with("copy_cc_to_ubuf_row_major_") ||
         name.starts_with("copy_cc_to_gm_row_major_");
}

static bool isAivTemplateRuntimeCall(StringRef name) {
  return name.starts_with("copy_ubuf_row_major_to_cbuf_zN_");
}

static void annotateAicTemplateRuntimeCall(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  func->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE),
                UnitAttr::get(ctx));
  func->setAttr(hivm::TFuncCoreTypeAttr::name,
                hivm::TFuncCoreTypeAttr::get(ctx, toFuncCoreType(HivmCoreKind::AIC)));
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
}

static void annotateAivTemplateRuntimeCall(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  func->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE),
                UnitAttr::get(ctx));
  func->setAttr(hivm::TFuncCoreTypeAttr::name,
                hivm::TFuncCoreTypeAttr::get(ctx, toFuncCoreType(HivmCoreKind::AIV)));
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
}

func::FuncOp getOrCreateRuntimeCall(ModuleOp module, StringRef name, ArrayRef<Type> operandTypes,
                                    ArrayRef<Type> resultTypes) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
    if (isAicTemplateRuntimeCall(name))
      annotateAicTemplateRuntimeCall(existing);
    if (isAivTemplateRuntimeCall(name))
      annotateAivTemplateRuntimeCall(existing);
    return existing;
  }
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = builder.getFunctionType(operandTypes, resultTypes);
  auto func = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
  func.setPrivate();
  if (isAicTemplateRuntimeCall(name))
    annotateAicTemplateRuntimeCall(func);
  if (isAivTemplateRuntimeCall(name))
    annotateAivTemplateRuntimeCall(func);
  return func;
}

} // namespace tla
