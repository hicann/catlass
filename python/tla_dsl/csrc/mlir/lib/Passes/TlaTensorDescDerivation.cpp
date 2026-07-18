#include "Passes/TlaTensorDescDerivation.h"

#include "PassesCommon.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <array>

namespace tla {

static mlir::Value makeIndexConstant(mlir::OpBuilder& builder, mlir::Location loc, int64_t value)
{
    return builder.create<arith::ConstantIndexOp>(loc, value);
}

static mlir::FailureOr<mlir::Value> makeStaticTensorInfoIndex(
    mlir::OpBuilder& builder, mlir::Operation* op, int64_t value, llvm::StringRef fieldName)
{
    if (value == ShapedType::kDynamic) {
        op->emitError() << "dynamic tensor metadata leaf in " << fieldName
                        << " is not yet supported in LowerToStdPass descriptor extraction";
        return failure();
    }
    return makeIndexConstant(builder, op->getLoc(), value);
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> materializeStaticTensorInfoIndices(
    mlir::OpBuilder& builder, mlir::Operation* op, llvm::ArrayRef<int64_t> values, llvm::StringRef fieldName)
{
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

static mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> materializeTensorInfoIndicesWithDynamicValues(
    mlir::OpBuilder& builder, mlir::Operation* op, llvm::ArrayRef<int64_t> values,
    llvm::ArrayRef<mlir::Value> dynamicValues, llvm::StringRef fieldName)
{
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

static mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> packRank2DynamicMetadataLeaves(
    mlir::Operation* op, llvm::ArrayRef<int64_t> leafDims, mlir::Value axis0, mlir::Value axis1,
    llvm::StringRef fieldName)
{
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

static mlir::FailureOr<TensorDescriptor> buildTensorDescriptorFromTensorInfo(
    mlir::OpBuilder& builder, mlir::Operation* op, mlir::Value base, mlir::Type bridgedBaseMemrefType,
    const TileTypeInfo& info, llvm::ArrayRef<mlir::Value> coordDynamicValues = {},
    llvm::ArrayRef<mlir::Value> originShapeDynamicValues = {})
{
    FailureOr<SmallVector<Value, 4>> coord =
        coordDynamicValues.empty() ?
            materializeStaticTensorInfoIndices(builder, op, info.coordDims, "coord") :
            materializeTensorInfoIndicesWithDynamicValues(builder, op, info.coordDims, coordDynamicValues, "coord");
    FailureOr<SmallVector<Value, 4>> originShape =
        originShapeDynamicValues.empty() ?
            materializeStaticTensorInfoIndices(builder, op, info.originShapeDims, "origin_shape") :
            materializeTensorInfoIndicesWithDynamicValues(
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

    return TensorDescriptor{
        base,
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

using ConstantFactory = llvm::function_ref<mlir::Value(mlir::Operation* anchor, int64_t value, unsigned bits)>;

mlir::Value IndexConstantCache::get(mlir::Operation* anchor, int64_t value, unsigned bits)
{
    Key key{value, bits};
    Block* scopeBlock = nullptr;
    if (auto tlaFunc = anchor->getParentOfType<::tla::FuncOp>()) {
        scopeBlock = &tlaFunc.getBody().front();
    } else if (auto func = anchor->getParentOfType<mlir::func::FuncOp>()) {
        scopeBlock = &func.getBody().front();
    } else if (auto module = anchor->getParentOfType<ModuleOp>()) {
        scopeBlock = &module.getBodyRegion().front();
    } else {
        scopeBlock = anchor->getBlock();
    }
    auto& cache = byScope[scopeBlock];
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

static mlir::FailureOr<TensorDescriptor> buildTileViewResultDescriptorFromParent(
    mlir::Operation* op, mlir::Value base, mlir::MemRefType bridgedBaseType, const TileTypeInfo& info,
    const TensorDescriptor& parent, mlir::Value row, mlir::Value col, mlir::Value sh0, mlir::Value sh1,
    ConstantFactory getConstant)
{
    Location loc = op->getLoc();
    OpBuilder b(op);

    if (!isPackedLayout(info.layoutTag) && !isLinearLayout(info.layoutTag)) {
        op->emitError() << "tile_view: unsupported layout tag for descriptor lowering";
        return failure();
    }

    // createOrFold so static abs-coord / origin arithmetic folds to a constant
    // in place (at the tile_view) instead of leaving an arith op for downstream
    // passes to clone -- tla-lower-tensor-desc is the sole descriptor producer,
    // and the vector helper consumes these operands directly.
    Value abs0 = b.createOrFold<arith::AddIOp>(loc, parent.absCoord0, row);
    Value abs1 = b.createOrFold<arith::AddIOp>(loc, parent.absCoord1, col);
    Value rest0 = b.createOrFold<arith::SubIOp>(loc, parent.originShape0, row);
    Value rest1 = b.createOrFold<arith::SubIOp>(loc, parent.originShape1, col);
    Value origin0 = b.createOrFold<arith::MinSIOp>(loc, sh0, rest0);
    Value origin1 = b.createOrFold<arith::MinSIOp>(loc, sh1, rest1);

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
                op->emitError() << "tile_view: packed shape dynamic leaf requires positive divisor, got " << divisor;
                return failure();
            }
            Value divisorV = getConstant(op, divisor, 0);
            Value one = getConstant(op, 1, 0);
            Value adjusted =
                b.createOrFold<arith::AddIOp>(loc, numerator, b.createOrFold<arith::SubIOp>(loc, divisorV, one));
            return b.createOrFold<arith::DivSIOp>(loc, adjusted, divisorV);
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
                    op->emitError() << "tile_view: dynamic packed shape leaf index 1 requires static leaf index 0";
                    return failure();
                }
                return ceilDivIndexByPositiveConst(sh0, info.shapeDims[0]);
            }
            if (idx == 3) {
                if (info.shapeDims[2] == ShapedType::kDynamic) {
                    op->emitError() << "tile_view: dynamic packed shape leaf index 3 requires static leaf index 2";
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

    return TensorDescriptor{
        base,
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

mlir::LogicalResult TensorDescriptorDerivation::derive(mlir::func::FuncOp funcOp)
{
    bool derivationFailed = false;
    auto& tensorDescriptorByValue = descriptorByValue;
    tensorDescriptorByValue.clear();
    Operation* root = funcOp.getOperation();
    auto isTensorScfCarrier = [](Value value) {
        if (Operation* def = value.getDefiningOp())
            return isa<scf::IfOp, scf::ForOp, scf::WhileOp>(def);

        auto blockArg = dyn_cast<BlockArgument>(value);
        if (!blockArg)
            return false;
        Operation* parent = blockArg.getOwner()->getParentOp();
        if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent))
            return blockArg.getOwner() == forOp.getBody() && blockArg.getArgNumber() > 0;
        if (auto whileOp = dyn_cast_or_null<scf::WhileOp>(parent))
            return blockArg.getOwner() == whileOp.getBeforeBody() || blockArg.getOwner() == whileOp.getAfterBody();
        return false;
    };

    llvm::DenseSet<Value> scfDependentTensorValues;
    auto deferScfDependentProducer = [&](Value result, Value source) {
        if (!isTensorScfCarrier(source) && !scfDependentTensorValues.contains(source))
            return false;
        scfDependentTensorValues.insert(result);
        return true;
    };
    auto getOrCreateConstant = [&](Operation* anchor, int64_t value, unsigned bits) -> Value {
        return constants.get(anchor, value, bits);
    };

    auto materializePackedIndexPair = [&](Operation* op, Value packedValue,
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
            return emitPackedError(Twine("tla.") + kind + " descriptor v1 requires exactly 2 packed elements");
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
                    return emitPackedError(
                        Twine("packed tla.") + kind + " type/operand dynamic element count mismatch");
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
            return emitPackedError(Twine("packed tla.") + kind + " type/operand dynamic element count mismatch");
        }
        return result;
    };

    auto unpackTileOffsetsAndShape = [&](Operation* op) -> FailureOr<std::array<Value, 4>> {
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
                    if (succeeded(::tla::getTlaIndexTreeLeaves(resTlaTy.getCoord().getTree(), coordLeaves)) &&
                        coordLeaves.size() == 2) {
                        auto mc = tileOp.getCoord().getDefiningOp<::tla::MakeCoordOp>();
                        if (mc) {
                            unsigned dynInTensor = 0;
                            for (int64_t l : coordLeaves)
                                if (l == ShapedType::kDynamic)
                                    ++dynInTensor;
                            if (dynInTensor == mc.getNumOperands()) {
                                size_t di = 0;
                                row = coordLeaves[0] == ShapedType::kDynamic ?
                                          mc.getDynElems()[di++] :
                                          getOrCreateConstant(op, coordLeaves[0], 0);
                                col = coordLeaves[1] == ShapedType::kDynamic ?
                                          mc.getDynElems()[di++] :
                                          getOrCreateConstant(op, coordLeaves[1], 0);
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
    // The producer-local descriptor builder takes the constant factory
    // explicitly.
    auto buildTileViewResultDescriptorFromParent =
        [&](Operation* op, Value base, MemRefType bridgedBaseType, const TileTypeInfo& info,
            const TensorDescriptor& parent, Value row, Value col, Value sh0, Value sh1) -> FailureOr<TensorDescriptor> {
        return ::tla::buildTileViewResultDescriptorFromParent(
            op, base, bridgedBaseType, info, parent, row, col, sh0, sh1, getOrCreateConstant);
    };

    // Seed tensor casts of already-bridged function arguments.
    auto seedRootTensorDescriptors = [&](Operation* funcOp, Block& entryBlock) {
        Location loc = funcOp->getLoc();
        OpBuilder builder(&entryBlock, entryBlock.begin());
        auto seedDescriptor = [&](Value descriptorValue, Value baseValue, Type tileType) {
            auto info = decodeTileTypeInfo(tileType);
            if (failed(info))
                return;
            auto bridgedBaseType = llvm::dyn_cast<MemRefType>(baseValue.getType());
            if (!bridgedBaseType)
                return;
            FailureOr<TensorDescriptor> desc =
                buildTensorDescriptorFromTensorInfo(builder, funcOp, baseValue, bridgedBaseType, *info);
            if (failed(desc))
                return;
            tensorDescriptorByValue[descriptorValue] = *desc;
        };

        for (Operation& op : entryBlock) {
            auto castOp = llvm::dyn_cast<UnrealizedConversionCastOp>(op);
            if (!castOp || castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
                continue;
            Value base = castOp.getOperand(0);
            if (!llvm::isa<BlockArgument>(base))
                continue;
            seedDescriptor(castOp.getResult(0), base, castOp.getResult(0).getType());
        }
    };

    // The standard pipeline has already lowered the Python kernel container and
    // bridged its tensor arguments before descriptor materialization.
    seedRootTensorDescriptors(root, funcOp.getBody().front());

    // Derive descriptors for tile-producing ops in SSA order: mixed
    // tla.tile_view/tla.make_tensor_like chains rely on producer descriptors
    // being available when their users are visited.
    funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
        // This method is producer-side and is called repeatedly while
        // tla-lower-tensor-desc rewrites structural SCF carriers. Descriptors
        // reconstructed at SCF region entries seed producer chains in that region.
        if (auto descOp = llvm::dyn_cast<::tla::TensorDescOp>(op)) {
            auto desc = descriptorFromTensorDescOp(descOp);
            if (failed(desc)) {
                derivationFailed = true;
                return;
            }
            tensorDescriptorByValue[descOp.getResult()] = *desc;
            return;
        }
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
                    op->emitError() << "unsupported tla.tile_view layouttag '" << layoutTagAttr.getValue() << "'";
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
                    if (deferScfDependentProducer(tileOp.getResult(), source))
                        return;
                    op->emitError() << "missing descriptor for tla.tile_view source tile; expected "
                                       "a materialized tensor producer or structural SCF carrier";
                    derivationFailed = true;
                    return;
                }
                const TensorDescriptor& parent = parentIt->second;
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
                    op->emitError() << "unsupported tla.tile_view layouttag '" << layoutTagAttr.getValue() << "'";
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
                op->emitError() << "expected tla.make_tensor_like to have exactly 2 operands and 1 result";
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
                if (deferScfDependentProducer(op->getResult(0), likeTile))
                    return;
                op->emitError() << "missing descriptor for tla.make_tensor_like reference tile; "
                                   "expected a materialized tensor producer or structural SCF carrier";
                derivationFailed = true;
                return;
            }
            const TensorDescriptor& parent = parentIt->second;
            if (!validateTensorDescriptorV1(
                    op, parent, "malformed parent tensor descriptor for tla.make_tensor_like reference tile",
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
                op->emitError() << "tla.make_tensor_like result tile rank must match reference descriptor "
                                   "(rank)";
                derivationFailed = true;
                return;
            }

            int64_t flatElemCount = ShapedType::kDynamic;
            if (auto n = getStaticAllocationElementCount(ptrValue); succeeded(n) && *n > 0) {
                flatElemCount = *n;
            } else if (
                childInfo->originShapeDims.size() >= 2 && childInfo->originShapeDims[0] != ShapedType::kDynamic &&
                childInfo->originShapeDims[1] != ShapedType::kDynamic) {
                int64_t dim0 = childInfo->originShapeDims[0];
                int64_t dim1 = childInfo->originShapeDims[1];
                if (dim0 > 0 && dim1 > 0)
                    flatElemCount = dim0 * dim1;
            }
            auto bridgedBaseType = buildHivmMemrefType(
                op->getContext(), {flatElemCount}, childInfo->mlirElementType, childInfo->tlaAddressSpace);
            if (failed(bridgedBaseType)) {
                op->emitError() << "tla.make_tensor_like buffer memref must be bridgeable to builtin memref type";
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
                op->emitError() << "unsupported tla.make_tensor_like layoutTag '" << layoutTagAttr.getValue() << "'";
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
            auto ceilDivIndexByPositiveConst = [&](Value numerator, int64_t divisor) -> FailureOr<Value> {
                if (divisor <= 0) {
                    op->emitError() << "packed shape dynamic leaf requires positive divisor, got " << divisor;
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
                auto supportedPackedLayout =
                    childInfo->layoutTag == TensorLayoutTag::zN || childInfo->layoutTag == TensorLayoutTag::nZ ||
                    childInfo->layoutTag == TensorLayoutTag::zZ || childInfo->layoutTag == TensorLayoutTag::L0C;
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
                        op->emitError() << "dynamic packed shape leaf index 1 requires static divisor leaf index 0";
                        return failure();
                    }
                    return ceilDivIndexByPositiveConst(parent.originShape0, leaves[0]);
                }
                if (idx == 3) {
                    if (leaves[2] == ShapedType::kDynamic) {
                        op->emitError() << "dynamic packed shape leaf index 3 requires static divisor leaf index 2";
                        return failure();
                    }
                    return ceilDivIndexByPositiveConst(parent.originShape1, leaves[2]);
                }
                op->emitError() << "dynamic packed shape leaf at index " << idx
                                << " is unsupported; only indices 1 and 3 may be dynamic";
                return failure();
            };
            auto materializePackedLeafFromTypeOrParent = [&](ArrayRef<int64_t> leaves, ArrayRef<Value> parentLeaves,
                                                             size_t idx, StringRef fieldName) -> FailureOr<Value> {
                int64_t leaf = leaves[idx];
                if (leaf == ShapedType::kDynamic) {
                    if (fieldName == "packed shape") {
                        FailureOr<Value> derived = materializePackedShapeDynamicLeafFromOrigin(leaves, idx);
                        if (succeeded(derived))
                            return *derived;
                    }
                    if (idx < parentLeaves.size() && parentLeaves[idx] && parentLeaves[idx].getType().isIndex())
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
                        op->emitError() << "dynamic packed stride derivation requires packed shape leaves " << a << ", "
                                        << b << ", " << c;
                        return failure();
                    }
                    Value ab = builder.create<arith::MulIOp>(op->getLoc(), packedShapeLeaves[a], packedShapeLeaves[b]);
                    return builder.create<arith::MulIOp>(op->getLoc(), ab, packedShapeLeaves[c]).getResult();
                };
                // Layout-coupled packed stride derivation from the remapped fractal shape leaves.
                // zN/L0C: stride[3] = ceil_div_rows * c0 * ele_num_per_c0 = shape[1]*shape[0]*shape[2]
                if ((childInfo->layoutTag == TensorLayoutTag::zN || childInfo->layoutTag == TensorLayoutTag::L0C) &&
                    idx == 3) {
                    return mulShapeLeaves(/*a=*/1, /*b=*/0, /*c=*/2);
                }
                // nZ/zZ: stride[1] = ceil_div_cols * c0 * ele_num_per_c0 = shape[3]*shape[2]*shape[0]
                if ((childInfo->layoutTag == TensorLayoutTag::nZ || childInfo->layoutTag == TensorLayoutTag::zZ) &&
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
            FailureOr<Value> origin0 =
                materializeLeafFromTypeOrParent(childInfo->originShapeDims[0], parent.originShape0, "origin_shape");
            FailureOr<Value> origin1 =
                materializeLeafFromTypeOrParent(childInfo->originShapeDims[1], parent.originShape1, "origin_shape");
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
                        leaf = materializePackedLeafFromTypeOrParent(
                            childInfo->strideDims, parent.packedStride, i, "packed stride");
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
                op->emitError() << "expected tla.make_tensor to have exactly 3 operands and 1 result";
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
                op->emitError() << "tla.make_tensor layout operand must come from tla.make_layout";
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
            } else if (
                childInfo->originShapeDims.size() >= 2 && childInfo->originShapeDims[0] != ShapedType::kDynamic &&
                childInfo->originShapeDims[1] != ShapedType::kDynamic) {
                int64_t dim0 = childInfo->originShapeDims[0];
                int64_t dim1 = childInfo->originShapeDims[1];
                if (dim0 > 0 && dim1 > 0)
                    flatElemCount = dim0 * dim1;
            }
            auto bridgedBaseType = buildHivmMemrefType(
                op->getContext(), {flatElemCount}, childInfo->mlirElementType, childInfo->tlaAddressSpace);
            if (failed(bridgedBaseType)) {
                op->emitError() << "tla.make_tensor buffer memref must be bridgeable to builtin memref type";
                derivationFailed = true;
                return;
            }

            Value typedBuffer = ptrValue;

            // Materialize index-tree leaves from the operand defining ops. Static leaves
            // become constants; dynamic leaves are pulled from tla.make_shape/make_stride/
            // make_coord dyn-elems in leaf order. ``childInfo`` already promotes rank-1
            // linear to rank-2, so the leading synthetic ``1``/``0`` leaves are static
            // here. The promoted leading stride may be derived (extent * elemStride) and
            // is not present on make_stride; that case is handled after shape leaves are
            // available below.
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
                            op->emitError() << "tla.make_tensor " << kind << " dynamic operands must be index type";
                            return failure();
                        }
                        result.push_back(dv);
                    } else {
                        result.push_back(getOrCreateConstant(op, leaf, 0));
                    }
                }
                return result;
            };

            auto shapeLeaves = materializeLeaves(makeLayout.getShape(), childInfo->shapeDims, "shape");
            if (failed(shapeLeaves)) {
                derivationFailed = true;
                return;
            }

            // Rank-1 linear layout is normalized to rank-2 as shape=(1,E),
            // stride=(E*S,S). When E is dynamic, stride0 is derived rather than
            // represented by a make_stride operand. Recover it from the
            // materialized extent and element stride.
            FailureOr<SmallVector<Value, 4>> strideLeaves;
            {
                unsigned strideDynLeafCount = 0;
                for (int64_t leaf : childInfo->strideDims)
                    if (leaf == ShapedType::kDynamic)
                        ++strideDynLeafCount;
                SmallVector<Value, 4> strideDynElems;
                if (auto makeStride = makeLayout.getStride().getDefiningOp<::tla::MakeStrideOp>())
                    strideDynElems.append(makeStride.getDynElems().begin(), makeStride.getDynElems().end());

                bool needsDerivedLeadingStride = strideDynLeafCount > 0 && strideDynElems.size() < strideDynLeafCount &&
                                                 childInfo->shapeDims.size() == 2 &&
                                                 childInfo->strideDims.size() == 2 && childInfo->shapeDims[0] == 1 &&
                                                 childInfo->strideDims[0] == ShapedType::kDynamic;
                if (needsDerivedLeadingStride) {
                    unsigned trailingDynNeeded = childInfo->strideDims[1] == ShapedType::kDynamic ? 1u : 0u;
                    if (strideDynElems.size() < trailingDynNeeded) {
                        op->emitError() << "tla.make_tensor stride has a derived dynamic leaf that is not "
                                           "directly operand-backed (e.g. rank-1 stride with dynamic extent); "
                                           "pass explicit leaves via tla.make_stride";
                        derivationFailed = true;
                        return;
                    }
                    Value elemStride = childInfo->strideDims[1] == ShapedType::kDynamic ?
                                           strideDynElems[0] :
                                           getOrCreateConstant(op, childInfo->strideDims[1], 0);
                    if (!elemStride.getType().isIndex()) {
                        op->emitError() << "tla.make_tensor stride dynamic operands must be index type";
                        derivationFailed = true;
                        return;
                    }
                    OpBuilder builder(op);
                    Value leading = builder.create<arith::MulIOp>(op->getLoc(), (*shapeLeaves)[1], elemStride);
                    strideLeaves = SmallVector<Value, 4>{leading, elemStride};
                } else {
                    strideLeaves = materializeLeaves(makeLayout.getStride(), childInfo->strideDims, "stride");
                }
            }

            auto coordLeaves = materializeLeaves(coordValue, childInfo->coordDims, "coord");
            if (failed(strideLeaves) || failed(coordLeaves)) {
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
                auto originLeaves = materializeLeaves(originOperand, childInfo->originShapeDims, "shape");
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

} // namespace tla
