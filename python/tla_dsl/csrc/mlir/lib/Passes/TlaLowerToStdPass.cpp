#include "PassesCommon.h"
#include "PassesInternal.h"
#include "mlir/IR/IRMapping.h"

namespace tla {
namespace {

class TlaLowerToStdPass : public PassWrapper<TlaLowerToStdPass, OperationPass<ModuleOp>> {
private:
  // Region provenance used to route execution-unit-specific rewrites.
  static constexpr StringLiteral kExecUnitAttrName = "tla.exec_unit";
  static constexpr StringLiteral kExecUnitCube = "cube";
  static constexpr StringLiteral kExecUnitVector = "vector";

  // Pass-local tensor layout semantics (descriptor v1).
  // This model drives rewrites and validation; it is not serialized in IR.
  enum class TensorLayoutTag {
    Unknown,
    RowMajor,
    ColumnMajor,
    zN,
    zZ,
    nZ,
    L0C,
  };

  struct TensorDescriptor {
    Value base;
    Type bridgedBaseMemrefType;
    Value rowOffset;
    Value colOffset;
    Value stride0;
    Value stride1;
    Value shape0;
    Value shape1;
    Value originShape0;
    Value originShape1;
    Value absCoord0;
    Value absCoord1;
    TensorLayoutTag layoutTag = TensorLayoutTag::Unknown;
    std::string addrspace;
    std::string elementType;
    int64_t rank = 0;
    SmallVector<Value, 4> packedShape;
    SmallVector<Value, 4> packedStride;
  };

  /// Staged deletes for bridge casts / control-flow wrappers during lowering.
  /// `tla.alloc_ptr` offsets are assigned in `tla-alloc-ptr-to-hivm-pointer-cast`
  /// (always run before this pass via `buildTlaPipeline`).
  struct AllocatorOffsetState {
    SmallVectorImpl<Operation *> *toErase = nullptr;
  };

  struct VAddVariant {
    // Signatures are matched against printed MLIR types.
    const char *vectorType;
    const char *scalarType;
    const char *maskType;
    // Runtime callee symbol emitted/declared in-module.
    const char *llvmIntrinsicName;
  };

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerToStdPass)

  StringRef getArgument() const override { return "tla-lower-to-std"; }
  StringRef getName() const override { return "TlaLowerToStdPass"; }
  StringRef getDescription() const override { return "Lower basic Tla ops to standard MLIR."; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect,
                    scf::SCFDialect, hivm::HIVMDialect>();
  }

  static std::string typeToString(Type type) {
    std::string storage;
    llvm::raw_string_ostream os(storage);
    type.print(os);
    return os.str();
  }

  struct TileTypeInfo {
    SmallVector<int64_t, 4> shapeDims;
    SmallVector<int64_t, 4> strideDims;
    SmallVector<int64_t, 4> coordDims;
    SmallVector<int64_t, 4> originShapeDims;
    std::string addressSpace;
    std::string elementType;
    Type mlirElementType;
    AddressSpace tlaAddressSpace = AddressSpace::gm;
    TensorLayoutTag layoutTag = TensorLayoutTag::Unknown;
    int64_t rank = 0;
  };

  static bool isPackedLayout(TensorLayoutTag layoutTag) {
    return layoutTag == TensorLayoutTag::zN || layoutTag == TensorLayoutTag::zZ ||
           layoutTag == TensorLayoutTag::nZ || layoutTag == TensorLayoutTag::L0C;
  }

  static bool isLinearLayout(TensorLayoutTag layoutTag) {
    return layoutTag == TensorLayoutTag::RowMajor || layoutTag == TensorLayoutTag::ColumnMajor;
  }

  static FailureOr<hivm::AddressSpace> resolveHivmAddressSpace(MLIRContext *ctx,
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

  // Decode ``!tla.tensor`` metadata from the structured ``tla::TlaTensorType``.
  static FailureOr<TileTypeInfo> decodeTileTypeInfo(Type tileType) {
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

  static bool parseMemrefTypeMetadata(Type memrefType, std::string &addrspace,
                                      std::string &elementType, int64_t &rank) {
    auto tlaMemref = dyn_cast<::tla::MemrefType>(memrefType);
    if (!tlaMemref)
      return false;

    rank = static_cast<int64_t>(tlaMemref.getShape().size());
    std::string elemBuf;
    llvm::raw_string_ostream elemOs(elemBuf);
    elemOs << tlaMemref.getElementType();
    elemOs.flush();
    elementType = std::move(elemBuf);
    addrspace = stringifyAddressSpace(tlaMemref.getAddressSpace()).str();
    return !elementType.empty() && !addrspace.empty();
  }

  static bool validateTensorDescriptorV1(Operation *op, const TensorDescriptor &desc,
                                         StringRef errorMessage, bool requireShapeOperands) {
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

  static StringRef stringifyTensorLayoutTag(TensorLayoutTag layoutTag) {
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

  static FailureOr<TensorLayoutTag> convertTlaLayoutTag(::LayoutTag layoutTag) {
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

  static FailureOr<TensorLayoutTag> parseTensorLayoutTagAttr(StringRef layouttag) {
    auto layoutTag = symbolizeLayoutTag(layouttag);
    if (!layoutTag)
      return failure();
    return convertTlaLayoutTag(*layoutTag);
  }

  static Value makeIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
    return builder.create<arith::ConstantIndexOp>(loc, value);
  }

  static Value makeI64Constant(OpBuilder &builder, Location loc, int64_t value) {
    return builder.create<arith::ConstantIntOp>(loc, value, 64);
  }

  static FailureOr<Value> makeZeroValue(OpBuilder &builder, Location loc, Type type) {
    if (isa<FloatType>(type))
      return builder.create<arith::ConstantOp>(loc, type, builder.getFloatAttr(type, 0.0))
          .getResult();
    if (isa<IntegerType>(type))
      return builder.create<arith::ConstantOp>(loc, type, builder.getIntegerAttr(type, 0))
          .getResult();
    return failure();
  }

  static FailureOr<Value> makeStaticTensorInfoIndex(OpBuilder &builder, Operation *op,
                                                    int64_t value, StringRef fieldName) {
    if (value == ShapedType::kDynamic) {
      op->emitError() << "dynamic tensor metadata leaf in " << fieldName
                      << " is not yet supported in LowerToStdPass descriptor extraction";
      return failure();
    }
    return makeIndexConstant(builder, op->getLoc(), value);
  }

  static FailureOr<SmallVector<Value, 4>>
  materializeStaticTensorInfoIndices(OpBuilder &builder, Operation *op, ArrayRef<int64_t> values,
                                     StringRef fieldName) {
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

  static FailureOr<SmallVector<Value, 4>> materializeTensorInfoIndicesWithDynamicValues(
      OpBuilder &builder, Operation *op, ArrayRef<int64_t> values, ArrayRef<Value> dynamicValues,
      StringRef fieldName) {
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

  /// For rank-2 ``coord`` / ``origin_shape`` segments,
  /// ``materializeTensorInfoIndicesWithDynamicValues`` consumes one SSA per ``?`` **in leaf index
  /// order**.  ``tla.tile_view`` unpacks (row, col) and (sh0, sh1); map dynamic leaf i to axis i
  /// (coord: row/col, origin: shape0/shape1).
  static FailureOr<SmallVector<Value, 4>> packRank2DynamicMetadataLeaves(Operation *op,
                                                                         ArrayRef<int64_t> leafDims,
                                                                         Value axis0, Value axis1,
                                                                         StringRef fieldName) {
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

  static FailureOr<TensorDescriptor>
  buildTensorDescriptorFromTensorInfo(OpBuilder &builder, Operation *op, Value base,
                                      Type bridgedBaseMemrefType, const TileTypeInfo &info,
                                      ArrayRef<Value> coordDynamicValues = {},
                                      ArrayRef<Value> originShapeDynamicValues = {}) {
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

  static void pushStagedErase(SmallVectorImpl<Operation *> *toErase, Operation *op) {
    if (toErase && op && !llvm::is_contained(*toErase, op))
      toErase->push_back(op);
  }

  static Value castValueToI64(OpBuilder &builder, Location loc, Value value) {
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

  static SmallVector<Value, 8> buildRowMajorCopyPayload(OpBuilder &builder, Location loc,
                                                        const TensorDescriptor &desc) {
    // Row-major runtime payload carries both tile extent and origin layout.
    // dma.cpp builds a TLA tensor from origin shape/stride plus abs coord so
    // nested tile_view copies preserve the root row stride.
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

  static SmallVector<Value, 20> buildCopyPayloadForRoute(OpBuilder &builder, Location loc,
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

  static FailureOr<TensorLayoutTag> getExplicitTensorLayoutTagAttr(Operation *op) {
    auto layoutTagAttr = op->getAttrOfType<StringAttr>("layouttag");
    if (!layoutTagAttr)
      return failure();
    return parseTensorLayoutTagAttr(layoutTagAttr.getValue());
  }

  static bool parseMemrefMetadataOrEmit(Operation *op, Type memrefType, StringRef parseError,
                                        std::string &addrspace, std::string &elementType,
                                        int64_t &rank) {
    if (!parseMemrefTypeMetadata(memrefType, addrspace, elementType, rank)) {
      op->emitError() << parseError;
      return false;
    }
    return true;
  }

  static FailureOr<MemRefType> bridgeTlaMemrefType(Type tlaMemrefType) {
    return bridgeTlaFuncMemrefType(tlaMemrefType);
  }

  // Bridge structured !tla.tensor<...> to builtin memref<..., memspace>.
  // Root tensor arguments reuse the tensor's full static shape as the backing
  // storage extent so descriptor-driven view lowering can materialize subviews
  // from first-class tensor-typed kernel parameters.
  static FailureOr<MemRefType> bridgeTlaTensorType(Type tlaTensorType) {
    return bridgeTlaFuncTensorType(tlaTensorType);
  }

  static bool hasNoResultUses(Operation *op) {
    return llvm::all_of(op->getResults(), [](Value result) { return result.use_empty(); });
  }

  static bool isTlaTensorBridgeCast(UnrealizedConversionCastOp op) {
    return llvm::any_of(op->getOperandTypes(),
                        [](Type type) { return succeeded(decodeTileTypeInfo(type)); }) ||
           llvm::any_of(op->getResultTypes(),
                        [](Type type) { return succeeded(decodeTileTypeInfo(type)); });
  }

  static bool isTlaTensorType(Type type) { return llvm::isa<::tla::TlaTensorType>(type); }

  static bool hasTlaTensorCallSurface(func::CallOp callOp) {
    return llvm::any_of(callOp.getOperandTypes(), isTlaTensorType) ||
           llvm::any_of(callOp.getResultTypes(), isTlaTensorType);
  }

  static bool isDeadTensorBridgeCast(UnrealizedConversionCastOp op) {
    return hasNoResultUses(op.getOperation()) && isTlaTensorBridgeCast(op);
  }

  static LogicalResult bridgeFuncTensorEntryAbi(ModuleOp module) {
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      FunctionType funcType = funcOp.getFunctionType();

      if (funcOp.empty()) {
        bool changed = false;
        SmallVector<Type, 8> bridgedInputs;
        bridgedInputs.reserve(funcType.getNumInputs());
        for (Type input : funcType.getInputs()) {
          FailureOr<MemRefType> bridged = bridgeTlaTensorType(input);
          if (failed(bridged)) {
            bridgedInputs.push_back(input);
            continue;
          }
          bridgedInputs.push_back(*bridged);
          changed = true;
        }

        SmallVector<Type, 4> bridgedResults;
        bridgedResults.reserve(funcType.getNumResults());
        for (Type result : funcType.getResults()) {
          FailureOr<MemRefType> bridged = bridgeTlaTensorType(result);
          if (failed(bridged)) {
            bridgedResults.push_back(result);
            continue;
          }
          bridgedResults.push_back(*bridged);
          changed = true;
        }

        if (changed)
          funcOp.setType(FunctionType::get(funcOp.getContext(), bridgedInputs, bridgedResults));
        continue;
      }

      SmallVector<Type, 8> bridgedInputs;
      bridgedInputs.reserve(funcType.getNumInputs());
      SmallVector<std::pair<BlockArgument, Type>, 8> argsToBridge;

      for (BlockArgument arg : funcOp.getArguments()) {
        Type argType = arg.getType();
        FailureOr<MemRefType> bridged = bridgeTlaTensorType(argType);
        if (failed(bridged)) {
          bridgedInputs.push_back(argType);
          continue;
        }
        bridgedInputs.push_back(*bridged);
        argsToBridge.push_back({arg, *bridged});
      }

      if (argsToBridge.empty())
        continue;

      funcOp.setType(FunctionType::get(funcOp.getContext(), bridgedInputs, funcType.getResults()));

      for (auto [arg, bridgedType] : argsToBridge) {
        arg.setType(bridgedType);
        SmallVector<UnrealizedConversionCastOp, 4> castsToErase;
        for (Operation *user : llvm::make_early_inc_range(arg.getUsers())) {
          auto castOp = llvm::dyn_cast<UnrealizedConversionCastOp>(user);
          if (!castOp || castOp->getNumOperands() != 1 || castOp->getNumResults() != 1 ||
              castOp.getResult(0).getType() != bridgedType)
            continue;
          castOp.getResult(0).replaceAllUsesWith(arg);
          castsToErase.push_back(castOp);
        }
        for (UnrealizedConversionCastOp castOp : castsToErase)
          castOp.erase();
      }
    }
    return success();
  }

  static void eraseDeadMaterializations(ModuleOp module) {
    bool progress = true;
    while (progress) {
      progress = false;
      SmallVector<Operation *, 8> toErase;
      module.walk([&](Operation *op) {
        if (!hasNoResultUses(op))
          return;
        if (llvm::isa<mlir::memref::SubViewOp>(op)) {
          toErase.push_back(op);
          return;
        }
#if defined(TLA_DSL_ENABLE_HIVM)
        if (llvm::isa<hivm::PointerCastOp>(op))
          toErase.push_back(op);
#endif
      });
      for (Operation *op : toErase) {
        if (!op->getBlock())
          continue;
        op->erase();
        progress = true;
      }
    }
  }

  struct EraseDeadTensorBridgeCastPattern : public OpRewritePattern<UnrealizedConversionCastOp> {
    using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                  PatternRewriter &rewriter) const override {
      if (!isDeadTensorBridgeCast(op))
        return failure();
      rewriter.eraseOp(op);
      return success();
    }
  };

  template <typename OpT> struct EraseDeadOpPattern : public OpRewritePattern<OpT> {
    using OpRewritePattern<OpT>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const override {
      if (!hasNoResultUses(op.getOperation()))
        return failure();
      rewriter.eraseOp(op);
      return success();
    }
  };

  static Value createFallbackBaseMemrefCast(PatternRewriter &rewriter, Location loc,
                                            const TensorDescriptor &desc) {
    return rewriter
        .create<UnrealizedConversionCastOp>(loc, TypeRange{desc.bridgedBaseMemrefType},
                                            ValueRange{desc.base})
        .getResult(0);
  }

  // If ptr is lowered by tla-alloc-ptr-to-hivm-pointer-cast, return static 1D element count.
  static FailureOr<int64_t> getStatic1DElementCountFromHivmPtrBridge(Value ptrValue) {
    Value cur = ptrValue;
    if (auto bridge = cur.getDefiningOp<::tla::HivmMemrefAsPtrOp>())
      cur = bridge.getMemref();
    else if (auto ucc = cur.getDefiningOp<UnrealizedConversionCastOp>())
      cur = ucc->getOperand(0);
    if (auto pc = cur.getDefiningOp<hivm::PointerCastOp>()) {
      auto mr = dyn_cast<MemRefType>(pc.getResult().getType());
      if (mr && mr.getRank() == 1 && mr.getDimSize(0) != ShapedType::kDynamic)
        return mr.getShape()[0];
    }
    return failure();
  }

  static FailureOr<Value> materializeDescriptorBaseMemref(PatternRewriter &rewriter, Location loc,
                                                          const TensorDescriptor &desc,
                                                          AllocatorOffsetState *allocatorState,
                                                          Operation *diagnosticOp) {
    auto memrefType = dyn_cast<MemRefType>(desc.bridgedBaseMemrefType);
    if (!memrefType)
      return failure();

    if (isa<MemRefType>(desc.base.getType()))
      return castMemrefToType(rewriter, loc, desc.base, memrefType);

    if (allocatorState) {
      if (auto ptrBridge = dyn_cast_or_null<::tla::HivmMemrefAsPtrOp>(desc.base.getDefiningOp())) {
        FailureOr<Value> ptrResult = materializePtrValueAsMemref(
            rewriter, loc, ptrBridge.getResult(), memrefType, diagnosticOp);
        if (succeeded(ptrResult)) {
          pushStagedErase(allocatorState->toErase, ptrBridge.getOperation());
          return *ptrResult;
        }
      }
      auto castOp = dyn_cast_or_null<UnrealizedConversionCastOp>(desc.base.getDefiningOp());
      if (castOp && castOp->getNumOperands() == 1) {
        Value source = castOp->getOperand(0);
        if (isa<MemRefType>(source.getType())) {
          FailureOr<Value> cast = castMemrefToType(rewriter, loc, source, memrefType);
          if (failed(cast))
            return failure();
          pushStagedErase(allocatorState->toErase, castOp.getOperation());
          return *cast;
        }
        if (auto ifOp = source.getDefiningOp<scf::IfOp>()) {
          FailureOr<Value> ifResult = materializePtrIfAsMemref(rewriter, loc, ifOp, memrefType,
                                                               *allocatorState, diagnosticOp);
          if (succeeded(ifResult)) {
            pushStagedErase(allocatorState->toErase, castOp.getOperation());
            return *ifResult;
          }
        }
        FailureOr<Value> ptrResult =
            materializePtrValueAsMemref(rewriter, loc, source, memrefType, diagnosticOp);
        if (succeeded(ptrResult)) {
          pushStagedErase(allocatorState->toErase, castOp.getOperation());
          return *ptrResult;
        }
      }
    }
    return createFallbackBaseMemrefCast(rewriter, loc, desc);
  }

  static FailureOr<Value> materializeTlaMemrefValue(PatternRewriter &rewriter, Location loc,
                                                    Value value, MemRefType memrefType) {
    if (auto ucc = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      Value src = ucc->getOperand(0);
      if (isa<MemRefType>(src.getType()))
        return castMemrefToType(rewriter, loc, src, memrefType);
    }
    return rewriter
        .create<UnrealizedConversionCastOp>(loc, TypeRange{memrefType}, ValueRange{value})
        .getResult(0);
  }

  /// TLA dtype token -> suffix on Cube copy runtime symbols (matches mlir/bc/Cube/dma.cpp).
  static StringRef copyRuntimeElemSuffix(StringRef elementType) {
    if (elementType == "f32")
      return "float";
    if (elementType == "f16")
      return "half";
    if (elementType == "bf16")
      return "bf16";
    return {};
  }

  static std::string getCopyRouteCallee(MLIRContext *ctx, StringRef srcAddrspace,
                                        StringRef dstAddrspace, TensorLayoutTag srcLayout,
                                        TensorLayoutTag dstLayout, StringRef srcElementType,
                                        StringRef dstElementType) {
    FailureOr<hivm::AddressSpace> srcSpace = resolveHivmAddressSpace(ctx, srcAddrspace);
    FailureOr<hivm::AddressSpace> dstSpace = resolveHivmAddressSpace(ctx, dstAddrspace);
    if (failed(srcSpace) || failed(dstSpace))
      return {};
    StringRef dstElem = dstElementType.empty() ? srcElementType : dstElementType;

    // Copy routing is keyed by explicit (addrspace, layout-tag) pairs.
    // Runtime symbol names encode both endpoint layout tags so future layout
    // variants can be added as new explicit routes instead of overloading
    // addrspace-only names.
    // Current TLA MLIR carries explicit layout tags through tla.make_tensor_like and
    // optionally on tla.tile_view.
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
    return {};
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
           name.starts_with("copy_cc_to_gm_row_major_");
  }

  static bool isAivTemplateRuntimeCall(StringRef name) {
    return name == "copy_gm_to_ubuf_1d_float" || name == "copy_ubuf_to_gm_1d_float";
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

  static bool isSetCtrlOp(Operation *op, bool expectedEnable, int64_t expectedIdx) {
    if (!llvm::isa<hivm::SetCtrlOp>(op))
      return false;
    auto enableAttr = op->getAttrOfType<BoolAttr>("enable");
    auto idxAttr = op->getAttrOfType<IntegerAttr>("idx");
    return enableAttr && idxAttr && enableAttr.getValue() == expectedEnable &&
           idxAttr.getInt() == expectedIdx;
  }

  // Preserve wrapper context before flattening for ops that still dispatch on
  // execution unit metadata.
  static void annotateExecUnit(Region &region, StringRef execUnit, MLIRContext *ctx) {
    auto unitAttr = StringAttr::get(ctx, execUnit);
    region.walk([&](Operation *nestedOp) {
      StringRef name = nestedOp->getName().getStringRef();
      if (name != "tla.add")
        return;
      nestedOp->setAttr(kExecUnitAttrName, unitAttr);
    });
  }

  static StringRef getExecUnit(Operation *op) {
    auto attr = op->getAttrOfType<StringAttr>(kExecUnitAttrName);
    if (!attr)
      return {};
    return attr.getValue();
  }

  static llvm::StringMap<bool> collectVectorKernelNames(ModuleOp module) {
    llvm::StringMap<bool> vectorKernelNames;
    module.walk([&](::tla::FuncOp funcOp) {
      bool isVectorKernel = false;
      funcOp.walk([&](Operation *op) {
        if (llvm::isa<::tla::VectorOp, ::tla::AddOp>(op)) {
          isVectorKernel = true;
          return WalkResult::interrupt();
        }
        auto execUnit = op->getAttrOfType<StringAttr>(kExecUnitAttrName);
        if (execUnit && execUnit.getValue() == kExecUnitVector) {
          isVectorKernel = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!isVectorKernel)
        return;
      if (auto symName = funcOp->getAttrOfType<StringAttr>("sym_name"))
        vectorKernelNames[symName.getValue()] = true;
    });
    return vectorKernelNames;
  }

  static void injectVectorCtrlPrologueIntoFuncs(ModuleOp module,
                                                const llvm::StringMap<bool> &vectorKernelNames) {
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.isDeclaration() || funcOp.empty())
        continue;
      auto it = vectorKernelNames.find(funcOp.getSymName());
      if (it == vectorKernelNames.end() || !it->second)
        continue;

      Block &entry = funcOp.front();
      OpBuilder builder = OpBuilder::atBlockBegin(&entry);
      auto i64Type = builder.getI64Type();
      auto getCtrl = getOrCreateRuntimeCall(module, "llvm.hivm.GET.CTRL", {}, {i64Type});
      auto sbitset0 =
          getOrCreateRuntimeCall(module, "llvm.hivm.SBITSET0", {i64Type, i64Type}, {i64Type});
      auto setCtrl = getOrCreateRuntimeCall(module, "llvm.hivm.SET.CTRL", {i64Type}, {});

      auto ctrl = builder.create<func::CallOp>(funcOp.getLoc(), getCtrl, ValueRange{}).getResult(0);
      auto bit = builder.create<arith::ConstantIntOp>(funcOp.getLoc(), 56, 64);
      auto updated = builder.create<func::CallOp>(funcOp.getLoc(), sbitset0, ValueRange{ctrl, bit})
                         .getResult(0);
      builder.create<func::CallOp>(funcOp.getLoc(), setCtrl, ValueRange{updated});
    }
  }

  // Return an existing private callee declaration or create it in-module.
  static func::FuncOp getOrCreateRuntimeCall(ModuleOp module, StringRef name,
                                             ArrayRef<Type> operandTypes,
                                             ArrayRef<Type> resultTypes = {}) {
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

  static void pushStagedErase(SmallVectorImpl<Operation *> &toErase, Operation *op) {
    if (!op || llvm::is_contained(toErase, op))
      return;
    toErase.push_back(op);
  }

  static FailureOr<Value> castMemrefToType(PatternRewriter &rewriter, Location loc, Value value,
                                           MemRefType memrefType) {
    if (value.getType() == memrefType)
      return value;
    if (!isa<MemRefType>(value.getType()))
      return failure();
    return rewriter.create<mlir::memref::CastOp>(loc, memrefType, value).getResult();
  }

  static FailureOr<Value> materializeCallOperandAsType(PatternRewriter &rewriter,
                                                       func::CallOp callOp, Value operand,
                                                       Type expectedType) {
    if (operand.getType() == expectedType)
      return operand;

    auto expectedMemrefType = dyn_cast<MemRefType>(expectedType);
    if (!expectedMemrefType)
      return failure();

    if (isa<MemRefType>(operand.getType()))
      return castMemrefToType(rewriter, callOp.getLoc(), operand, expectedMemrefType);

    auto castOp = operand.getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp || castOp.getNumOperands() != 1 || castOp.getNumResults() != 1 ||
        !isTlaTensorType(operand.getType()))
      return failure();

    Value source = castOp.getOperand(0);
    if (source.getType() == expectedType)
      return source;
    if (isa<MemRefType>(source.getType()))
      return castMemrefToType(rewriter, callOp.getLoc(), source, expectedMemrefType);
    return failure();
  }

  static LogicalResult rewriteTensorTypedFuncCalls(ModuleOp module) {
    SmallVector<func::CallOp, 16> calls;
    module.walk([&](func::CallOp callOp) { calls.push_back(callOp); });

    for (func::CallOp callOp : calls) {
      if (!callOp || !callOp->getBlock())
        continue;

      auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
      if (!callee) {
        if (!hasTlaTensorCallSurface(callOp))
          continue;
        callOp.emitError() << "cannot lower tla.tensor call @" << callOp.getCallee()
                           << "; callee symbol was not found";
        return failure();
      }

      FunctionType calleeType = callee.getFunctionType();
      if (calleeType.getNumInputs() != callOp.getNumOperands() ||
          calleeType.getNumResults() != callOp.getNumResults()) {
        callOp.emitError() << "cannot lower tla.tensor call @" << callOp.getCallee()
                           << "; callee signature arity does not match call";
        return failure();
      }

      bool needsRewrite = hasTlaTensorCallSurface(callOp);
      for (auto [operand, expectedType] :
           llvm::zip_equal(callOp.getOperands(), calleeType.getInputs())) {
        if (operand.getType() != expectedType)
          needsRewrite = true;
      }
      for (auto [result, expectedType] :
           llvm::zip_equal(callOp.getResults(), calleeType.getResults())) {
        if (result.getType() != expectedType)
          needsRewrite = true;
      }
      if (!needsRewrite)
        continue;

      PatternRewriter rewriter(callOp.getContext());
      rewriter.setInsertionPoint(callOp);
      SmallVector<Value, 8> newOperands;
      newOperands.reserve(callOp.getNumOperands());
      for (auto [operand, expectedType] :
           llvm::zip_equal(callOp.getOperands(), calleeType.getInputs())) {
        FailureOr<Value> bridged =
            materializeCallOperandAsType(rewriter, callOp, operand, expectedType);
        if (failed(bridged)) {
          callOp.emitError() << "cannot lower tla.tensor operand for call @" << callOp.getCallee()
                             << "; expected a materialized memref bridge";
          return failure();
        }
        newOperands.push_back(*bridged);
      }

      auto newCall = rewriter.create<func::CallOp>(callOp.getLoc(), callOp.getCallee(),
                                                   calleeType.getResults(), newOperands);
      for (auto [oldResult, newResult] :
           llvm::zip_equal(callOp.getResults(), newCall.getResults())) {
        if (oldResult.getType() == newResult.getType()) {
          oldResult.replaceAllUsesWith(newResult);
          continue;
        }
        if (!isTlaTensorType(oldResult.getType())) {
          callOp.emitError() << "cannot lower non-tensor result type mismatch for call @"
                             << callOp.getCallee();
          return failure();
        }
        auto bridge = rewriter.create<UnrealizedConversionCastOp>(
            callOp.getLoc(), TypeRange{oldResult.getType()}, ValueRange{newResult});
        oldResult.replaceAllUsesWith(bridge.getResult(0));
      }
      rewriter.eraseOp(callOp);
    }

    return success();
  }

  static MemRefType getDynamicStridedMemrefType(MemRefType memrefType) {
    SmallVector<int64_t, 4> dynamicShape(memrefType.getRank(), ShapedType::kDynamic);
    SmallVector<int64_t, 4> dynamicStrides(memrefType.getRank(), ShapedType::kDynamic);
    auto layout =
        StridedLayoutAttr::get(memrefType.getContext(), ShapedType::kDynamic, dynamicStrides);
    return MemRefType::get(dynamicShape, memrefType.getElementType(), layout,
                           memrefType.getMemorySpace());
  }

  static bool hasOnlyStagedResultUsers(Operation *op, ArrayRef<Operation *> stagedErase) {
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

  static void stageDeadTileProducers(ModuleOp module, SmallVectorImpl<Operation *> &toErase) {
    bool progress = true;
    while (progress) {
      progress = false;
      SmallVector<Operation *, 8> newlyDead;
      module.walk([&](Operation *op) {
        if (!llvm::isa<::tla::TileViewOp, ::tla::MakeTensorLikeOp>(op) ||
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

  static FailureOr<Value> materializePtrValueAsMemref(PatternRewriter &rewriter, Location loc,
                                                      Value ptrValue, MemRefType memrefType,
                                                      Operation *diagnosticOp) {
    if (auto bridge = ptrValue.getDefiningOp<::tla::HivmMemrefAsPtrOp>()) {
      Value src = bridge.getMemref();
      if (isa<MemRefType>(src.getType()))
        return castMemrefToType(rewriter, loc, src, memrefType);
    }
    diagnosticOp->emitError()
        << "pointer memref materialization expects `tla.hivm_memref_as_ptr` (from "
           "`tla-alloc-ptr-to-hivm-pointer-cast`); got unsupported ptr def: "
        << ptrValue;
    return failure();
  }

  static FailureOr<Value> materializePtrIfAsMemref(PatternRewriter &rewriter, Location loc,
                                                   scf::IfOp ifOp, MemRefType memrefType,
                                                   AllocatorOffsetState &allocatorState,
                                                   Operation *diagnosticOp) {
    if (ifOp->getNumResults() != 1 || !isa<::tla::PtrType>(ifOp.getResult(0).getType()))
      return failure();

    scf::YieldOp thenYield = ifOp.thenYield();
    scf::YieldOp elseYield = ifOp.elseYield();
    if (!thenYield || !elseYield || thenYield.getNumOperands() != 1 ||
        elseYield.getNumOperands() != 1)
      return failure();

    auto newIf = rewriter.create<scf::IfOp>(loc, TypeRange{memrefType}, ifOp.getCondition(),
                                            /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(newIf.thenBlock());
      FailureOr<Value> thenMemref = materializePtrValueAsMemref(
          rewriter, thenYield.getLoc(), thenYield.getOperand(0), memrefType, diagnosticOp);
      if (failed(thenMemref))
        return failure();
      rewriter.create<scf::YieldOp>(thenYield.getLoc(), ValueRange{*thenMemref});
    }
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(newIf.elseBlock());
      FailureOr<Value> elseMemref = materializePtrValueAsMemref(
          rewriter, elseYield.getLoc(), elseYield.getOperand(0), memrefType, diagnosticOp);
      if (failed(elseMemref))
        return failure();
      rewriter.create<scf::YieldOp>(elseYield.getLoc(), ValueRange{*elseMemref});
    }
    pushStagedErase(allocatorState.toErase, ifOp.getOperation());
    return newIf.getResult(0);
  }

  struct LowerTlaSplatPattern : public OpRewritePattern<::tla::SplatOp> {
    using OpRewritePattern<::tla::SplatOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::tla::SplatOp op, PatternRewriter &rewriter) const override {
      auto valueAttr = op->getAttrOfType<Attribute>("value");
      if (!valueAttr) {
        op.emitError() << "expected tla.splat to have a 'value' attribute";
        return failure();
      }

      Type resultType = rewriter.getF32Type();
      if (auto floatAttr = llvm::dyn_cast<FloatAttr>(valueAttr))
        resultType = floatAttr.getType();
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(valueAttr))
        resultType = intAttr.getType();

      auto newAttr = llvm::dyn_cast<TypedAttr>(valueAttr);
      if (!newAttr) {
        if (auto intAttr = llvm::dyn_cast<IntegerAttr>(valueAttr)) {
          newAttr = IntegerAttr::get(resultType, intAttr.getInt());
        } else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(valueAttr)) {
          newAttr = FloatAttr::get(resultType, floatAttr.getValueAsDouble());
        }
      }
      if (!newAttr) {
        op.emitError() << "unsupported tla.splat value attribute type";
        return failure();
      }

      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, newAttr);
      return success();
    }
  };

  struct LowerTlaMmadPattern : public OpRewritePattern<::tla::MmadOp> {
    LowerTlaMmadPattern(MLIRContext *ctx, ModuleOp module,
                        DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue,
                        SmallVectorImpl<Operation *> &toErase)
        : OpRewritePattern<::tla::MmadOp>(ctx), module(module),
          tensorDescriptorByValue(tensorDescriptorByValue), toErase(toErase) {}

    LogicalResult matchAndRewrite(::tla::MmadOp op, PatternRewriter &rewriter) const override {
      if (op->getNumOperands() < 3)
        return success();

      Value acc = op->getOperand(0);
      Value lhs = op->getOperand(1);
      Value rhs = op->getOperand(2);
      Type accType = acc.getType();
      Type lhsType = lhs.getType();
      Type rhsType = rhs.getType();

      auto initAttr = op->getAttr("init_c");
      bool initVal = false;
      if (auto boolAttr = llvm::dyn_cast_or_null<BoolAttr>(initAttr)) {
        initVal = boolAttr.getValue();
      } else if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(initAttr)) {
        initVal = intAttr.getInt() != 0;
      }

      auto i1Type = rewriter.getI1Type();
      auto i64Type = rewriter.getI64Type();
      auto i8Type = rewriter.getI8Type();

      auto initConst = rewriter.create<arith::ConstantIntOp>(op.getLoc(), initVal, 1);
      auto unitFlagConst = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 8);

      auto lhsInfo = TlaLowerToStdPass::decodeTileTypeInfo(lhsType);
      auto rhsInfo = TlaLowerToStdPass::decodeTileTypeInfo(rhsType);
      auto accInfo = TlaLowerToStdPass::decodeTileTypeInfo(accType);
      if (failed(lhsInfo) || failed(rhsInfo) || failed(accInfo)) {
        op.emitError() << "tla.mmad currently requires structured tla.tensor operand types";
        return failure();
      }
      if (lhsInfo->rank != 2 || rhsInfo->rank != 2 || accInfo->rank != 2) {
        op.emitError() << "tla.mmad currently supports rank-2 tiles only";
        return failure();
      }
      if (lhsInfo->addressSpace != "l0a" || rhsInfo->addressSpace != "l0b" ||
          accInfo->addressSpace != "l0c") {
        op.emitError()
            << "unsupported tla.mmad tile addrspaces; expected acc in l0c, lhs in l0a, rhs in l0b";
        return failure();
      }
      bool supportedF16Route = lhsInfo->elementType == "f16" && rhsInfo->elementType == "f16" &&
                               accInfo->elementType == "f32";
      bool supportedBf16Route = lhsInfo->elementType == "bf16" && rhsInfo->elementType == "bf16" &&
                                accInfo->elementType == "f32";
      bool supportedF32Route = lhsInfo->elementType == "f32" && rhsInfo->elementType == "f32" &&
                               accInfo->elementType == "f32";
      if (!supportedF16Route && !supportedBf16Route && !supportedF32Route) {
        op.emitError() << "unsupported tla.mmad element types; expected f16,f16 -> f32, bf16,bf16 "
                          "-> f32, or f32,f32 -> f32 (L0C accumulator is fp32)";
        return failure();
      }

      auto maybeStaticShapeCheck = [&](int64_t lhsM, int64_t lhsK, int64_t rhsK, int64_t rhsN,
                                       int64_t accM, int64_t accN) -> LogicalResult {
        if (lhsM == ShapedType::kDynamic || lhsK == ShapedType::kDynamic ||
            rhsK == ShapedType::kDynamic || rhsN == ShapedType::kDynamic ||
            accM == ShapedType::kDynamic || accN == ShapedType::kDynamic) {
          return success();
        }
        if (lhsK != rhsK || lhsM != accM || rhsN != accN) {
          op.emitError() << "unsupported tla.mmad tile shape contract; expected lhs(MxK), "
                            "rhs(KxN), acc(MxN)";
          return failure();
        }
        return success();
      };
      if (failed(maybeStaticShapeCheck(lhsInfo->originShapeDims[0], lhsInfo->originShapeDims[1],
                                       rhsInfo->originShapeDims[0], rhsInfo->originShapeDims[1],
                                       accInfo->originShapeDims[0], accInfo->originShapeDims[1])))
        return failure();
      if (accInfo->layoutTag != TensorLayoutTag::L0C || lhsInfo->layoutTag != TensorLayoutTag::zN ||
          rhsInfo->layoutTag != TensorLayoutTag::nZ) {
        op.emitError()
            << "unsupported tla.mmad operand layout; expected acc L0Clayout, lhs zN, rhs nZ";
        return failure();
      }

      auto materializeTensorOperand = [&](Value tensor, Type tensorType) -> FailureOr<Value> {
        if (auto castOp = tensor.getDefiningOp<UnrealizedConversionCastOp>()) {
          if (castOp->getNumOperands() == 1 &&
              llvm::isa<MemRefType>(castOp->getOperand(0).getType())) {
            toErase.push_back(castOp.getOperation());
            return castOp->getOperand(0);
          }
        }
        auto memrefType = TlaLowerToStdPass::bridgeTlaTensorType(tensorType);
        if (failed(memrefType))
          return failure();
        return rewriter
            .create<UnrealizedConversionCastOp>(op.getLoc(), TypeRange{*memrefType},
                                                ValueRange{tensor})
            .getResult(0);
      };

      FailureOr<Value> lhsMemref = materializeTensorOperand(lhs, lhsType);
      FailureOr<Value> rhsMemref = materializeTensorOperand(rhs, rhsType);
      FailureOr<Value> accMemref = materializeTensorOperand(acc, accType);
      if (failed(lhsMemref) || failed(rhsMemref) || failed(accMemref)) {
        op.emitError() << "failed to bridge tla.mmad operands to memref values";
        return failure();
      }

      // Match tla.copy runtime ABI: pass dynamic strided memrefs to the C stub
      // (same as buildRuntimeMemref in LowerTlaCopyPattern).
      auto toRuntimeMemref = [&](Value v) -> FailureOr<Value> {
        auto baseType = dyn_cast<MemRefType>(v.getType());
        if (!baseType) {
          op.emitError() << "tla.mmad memref operand must have memref type";
          return failure();
        }
        MemRefType runtimeType = TlaLowerToStdPass::getDynamicStridedMemrefType(baseType);
        return TlaLowerToStdPass::castMemrefToType(rewriter, op.getLoc(), v, runtimeType);
      };
      FailureOr<Value> lhsRuntime = toRuntimeMemref(*lhsMemref);
      FailureOr<Value> rhsRuntime = toRuntimeMemref(*rhsMemref);
      FailureOr<Value> accRuntime = toRuntimeMemref(*accMemref);
      if (failed(lhsRuntime) || failed(rhsRuntime) || failed(accRuntime))
        return failure();

      auto materializeIndexDim = [&](Value tensor, int64_t staticOriginDim, StringRef fieldName,
                                     bool takeSecondDim) -> FailureOr<Value> {
        auto it = tensorDescriptorByValue.find(tensor);
        if (it != tensorDescriptorByValue.end()) {
          Value dim = takeSecondDim ? it->second.originShape1 : it->second.originShape0;
          if (dim && dim.getType().isIndex())
            return dim;
        }
        if (staticOriginDim == ShapedType::kDynamic) {
          op.emitError() << "tla.mmad requires " << fieldName
                         << " from tensor descriptor SSA when type origin_shape is dynamic";
          return failure();
        }
        return rewriter.create<arith::ConstantIndexOp>(op.getLoc(), staticOriginDim).getResult();
      };
      FailureOr<Value> mIndex = materializeIndexDim(lhs, lhsInfo->originShapeDims[0], "M", false);
      FailureOr<Value> kIndex = materializeIndexDim(lhs, lhsInfo->originShapeDims[1], "K", true);
      FailureOr<Value> nIndex = materializeIndexDim(rhs, rhsInfo->originShapeDims[1], "N", true);
      if (failed(mIndex) || failed(kIndex) || failed(nIndex))
        return failure();

      auto castIndexToI64 = [&](Value v) -> Value {
        return rewriter.create<arith::IndexCastOp>(op.getLoc(), i64Type, v).getResult();
      };
      Value mI64 = castIndexToI64(*mIndex);
      Value kI64 = castIndexToI64(*kIndex);
      Value nI64 = castIndexToI64(*nIndex);

      SmallVector<Type, 8> operandTypes = {(*lhsRuntime).getType(),
                                           (*rhsRuntime).getType(),
                                           (*accRuntime).getType(),
                                           i64Type,
                                           i64Type,
                                           i64Type,
                                           i1Type,
                                           i8Type};
      StringRef calleeName = supportedF16Route    ? "mmad_half_half_float"
                             : supportedBf16Route ? "mmad_bf16_bf16_float"
                                                  : "mmad_float_float_float";
      auto callee = TlaLowerToStdPass::getOrCreateRuntimeCall(module, calleeName, operandTypes);
      SmallVector<Value, 8> operands = {
          *lhsRuntime, *rhsRuntime, *accRuntime,           mI64,
          nI64,        kI64,        initConst.getResult(), unitFlagConst.getResult(),
      };
      rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
      toErase.push_back(op.getOperation());
      return success();
    }

  private:
    ModuleOp module;
    DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue;
    SmallVectorImpl<Operation *> &toErase;
  };

  struct LowerTlaGmAddPattern : public OpRewritePattern<::tla::GmAddOp> {
    LowerTlaGmAddPattern(MLIRContext *ctx, ModuleOp module, SmallVectorImpl<Operation *> &toErase)
        : OpRewritePattern<::tla::GmAddOp>(ctx), module(module), toErase(toErase) {}

    LogicalResult matchAndRewrite(::tla::GmAddOp op, PatternRewriter &rewriter) const override {
      if (op->getNumOperands() != 3 || op->getNumResults() != 0) {
        op.emitError() << "expected tla.gm_add to have exactly 3 operands and 0 results";
        return failure();
      }

      Type lhsType = op->getOperand(0).getType();
      Type rhsType = op->getOperand(1).getType();
      Type dstType = op->getOperand(2).getType();

      std::string lhsAddrspace, lhsElementType;
      std::string rhsAddrspace, rhsElementType;
      std::string dstAddrspace, dstElementType;
      int64_t lhsRank = 0;
      int64_t rhsRank = 0;
      int64_t dstRank = 0;
      if (!TlaLowerToStdPass::parseMemrefMetadataOrEmit(
              op, lhsType, "tla.gm_add currently requires typed !tla.memref operand types",
              lhsAddrspace, lhsElementType, lhsRank) ||
          !TlaLowerToStdPass::parseMemrefMetadataOrEmit(
              op, rhsType, "tla.gm_add currently requires typed !tla.memref operand types",
              rhsAddrspace, rhsElementType, rhsRank) ||
          !TlaLowerToStdPass::parseMemrefMetadataOrEmit(
              op, dstType, "tla.gm_add currently requires typed !tla.memref operand types",
              dstAddrspace, dstElementType, dstRank)) {
        return failure();
      }

      if (lhsRank != 1 || rhsRank != 1 || dstRank != 1) {
        op.emitError() << "tla.gm_add currently supports rank-1 memrefs only";
        return failure();
      }
      if (lhsAddrspace != "gm" || rhsAddrspace != "gm" || dstAddrspace != "gm") {
        op.emitError() << "tla.gm_add requires lhs/rhs/dst in gm addrspace";
        return failure();
      }
      if (lhsElementType != "f16" || rhsElementType != "f16" || dstElementType != "f16") {
        op.emitError() << "tla.gm_add currently supports f16 memrefs only";
        return failure();
      }

      auto lhsInfo = TlaLowerToStdPass::bridgeTlaMemrefType(lhsType);
      auto rhsInfo = TlaLowerToStdPass::bridgeTlaMemrefType(rhsType);
      auto dstInfo = TlaLowerToStdPass::bridgeTlaMemrefType(dstType);
      if (failed(lhsInfo) || failed(rhsInfo) || failed(dstInfo)) {
        op.emitError() << "failed to decode tla.gm_add memref operand types";
        return failure();
      }
      if (lhsInfo->getShape() != rhsInfo->getShape() ||
          lhsInfo->getShape() != dstInfo->getShape()) {
        op.emitError() << "tla.gm_add requires matching operand shapes";
        return failure();
      }

      int64_t length = lhsInfo->getShape()[0];
      auto i64Type = rewriter.getI64Type();
      SmallVector<Type, 4> operandTypes = {*lhsInfo, *rhsInfo, *dstInfo, i64Type};
      auto callee = TlaLowerToStdPass::getOrCreateRuntimeCall(
          module, "_mlir_ciface_tla_gm_add_half", operandTypes);
      Value lhsMemref = rewriter
                            .create<UnrealizedConversionCastOp>(op.getLoc(), TypeRange{*lhsInfo},
                                                                ValueRange{op->getOperand(0)})
                            .getResult(0);
      Value rhsMemref = rewriter
                            .create<UnrealizedConversionCastOp>(op.getLoc(), TypeRange{*rhsInfo},
                                                                ValueRange{op->getOperand(1)})
                            .getResult(0);
      Value dstMemref = rewriter
                            .create<UnrealizedConversionCastOp>(op.getLoc(), TypeRange{*dstInfo},
                                                                ValueRange{op->getOperand(2)})
                            .getResult(0);
      SmallVector<Value, 4> operands = {
          lhsMemref,
          rhsMemref,
          dstMemref,
          rewriter.create<arith::ConstantIntOp>(op.getLoc(), length, 64).getResult(),
      };
      rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
      toErase.push_back(op.getOperation());
      return success();
    }

  private:
    ModuleOp module;
    SmallVectorImpl<Operation *> &toErase;
  };

  struct LowerTlaAddPattern : public OpRewritePattern<::tla::AddOp> {
    LowerTlaAddPattern(MLIRContext *ctx, ModuleOp module, SmallVectorImpl<Operation *> &toErase)
        : OpRewritePattern<::tla::AddOp>(ctx), module(module), toErase(toErase) {}

    LogicalResult matchAndRewrite(::tla::AddOp op, PatternRewriter &rewriter) const override {
      if (op->getNumOperands() != 3 || op->getNumResults() != 0) {
        op.emitError() << "expected tla.add to have exactly 3 operands and 0 results";
        return failure();
      }

      Type lhsType = op->getOperand(1).getType();
      Type rhsType = op->getOperand(2).getType();
      Type dstType = op->getOperand(0).getType();

      std::string lhsAddrspace, lhsElementType;
      std::string rhsAddrspace, rhsElementType;
      std::string dstAddrspace, dstElementType;
      int64_t lhsRank = 0;
      int64_t rhsRank = 0;
      int64_t dstRank = 0;
      if (!TlaLowerToStdPass::parseMemrefMetadataOrEmit(
              op, lhsType, "tla.add currently requires typed !tla.memref operand types",
              lhsAddrspace, lhsElementType, lhsRank) ||
          !TlaLowerToStdPass::parseMemrefMetadataOrEmit(
              op, rhsType, "tla.add currently requires typed !tla.memref operand types",
              rhsAddrspace, rhsElementType, rhsRank) ||
          !TlaLowerToStdPass::parseMemrefMetadataOrEmit(
              op, dstType, "tla.add currently requires typed !tla.memref operand types",
              dstAddrspace, dstElementType, dstRank)) {
        return failure();
      }

      if (lhsRank != 1 || rhsRank != 1 || dstRank != 1) {
        op.emitError() << "tla.add currently supports rank-1 memrefs only";
        return failure();
      }
      if (lhsAddrspace != "ub" || rhsAddrspace != "ub" || dstAddrspace != "ub") {
        op.emitError() << "tla.add requires lhs/rhs/dst in ub addrspace";
        return failure();
      }
      if (lhsElementType != "f32" || rhsElementType != "f32" || dstElementType != "f32") {
        op.emitError() << "tla.add currently supports f32 memrefs only";
        return failure();
      }

      auto lhsInfo = TlaLowerToStdPass::bridgeTlaMemrefType(lhsType);
      auto rhsInfo = TlaLowerToStdPass::bridgeTlaMemrefType(rhsType);
      auto dstInfo = TlaLowerToStdPass::bridgeTlaMemrefType(dstType);
      if (failed(lhsInfo) || failed(rhsInfo) || failed(dstInfo)) {
        op.emitError() << "failed to decode tla.add memref operand types";
        return failure();
      }
      if (lhsInfo->getShape() != rhsInfo->getShape() ||
          lhsInfo->getShape() != dstInfo->getShape()) {
        op.emitError() << "tla.add requires matching operand shapes";
        return failure();
      }

      SmallVector<Type, 4> operandTypes = {*lhsInfo, *rhsInfo, *dstInfo, *dstInfo};
      auto callee = TlaLowerToStdPass::getOrCreateRuntimeCall(module, "_mlir_ciface_vadd_1d_float",
                                                              operandTypes);
      auto lhsMemref = TlaLowerToStdPass::materializeTlaMemrefValue(rewriter, op.getLoc(),
                                                                    op->getOperand(1), *lhsInfo);
      auto rhsMemref = TlaLowerToStdPass::materializeTlaMemrefValue(rewriter, op.getLoc(),
                                                                    op->getOperand(2), *rhsInfo);
      auto dstMemref = TlaLowerToStdPass::materializeTlaMemrefValue(rewriter, op.getLoc(),
                                                                    op->getOperand(0), *dstInfo);
      if (failed(lhsMemref) || failed(rhsMemref) || failed(dstMemref))
        return failure();
      SmallVector<Value, 4> operands = {*lhsMemref, *rhsMemref, *dstMemref, *dstMemref};
      rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
      toErase.push_back(op.getOperation());
      return success();
    }

  private:
    ModuleOp module;
    SmallVectorImpl<Operation *> &toErase;
  };

  struct LowerTlaCubePattern : public OpRewritePattern<::tla::CubeOp> {
    using OpRewritePattern<::tla::CubeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::tla::CubeOp op, PatternRewriter &rewriter) const override {
      if (op->getNumRegions() == 0) {
        rewriter.eraseOp(op);
        return success();
      }
      Region &region = op->getRegion(0);
      if (region.empty()) {
        rewriter.eraseOp(op);
        return success();
      }

      TlaLowerToStdPass::annotateExecUnit(region, TlaLowerToStdPass::kExecUnitCube,
                                          op->getContext());
      Block &body = region.front();
      Block *parentBlock = op->getBlock();
      parentBlock->getOperations().splice(op->getIterator(), body.getOperations(), body.begin(),
                                          body.end());
      rewriter.eraseOp(op);
      return success();
    }
  };

  struct LowerTlaVectorPattern : public OpRewritePattern<::tla::VectorOp> {
    using OpRewritePattern<::tla::VectorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::tla::VectorOp op, PatternRewriter &rewriter) const override {
      if (op->getNumRegions() == 0) {
        rewriter.eraseOp(op);
        return success();
      }
      Region &region = op->getRegion(0);
      if (region.empty()) {
        rewriter.eraseOp(op);
        return success();
      }

      TlaLowerToStdPass::annotateExecUnit(region, TlaLowerToStdPass::kExecUnitVector,
                                          op->getContext());
      Block &body = region.front();
      Block *parentBlock = op->getBlock();
      parentBlock->getOperations().splice(op->getIterator(), body.getOperations(), body.begin(),
                                          body.end());
      rewriter.eraseOp(op);
      return success();
    }
  };

  struct LowerTlaCopyPattern : public OpRewritePattern<::tla::CopyOp> {
    LowerTlaCopyPattern(MLIRContext *ctx, ModuleOp module,
                        DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue,
                        SmallVectorImpl<Operation *> &toErase, AllocatorOffsetState *allocatorState)
        : OpRewritePattern<::tla::CopyOp>(ctx), module(module),
          tensorDescriptorByValue(tensorDescriptorByValue), toErase(toErase),
          allocatorState(allocatorState) {}

    LogicalResult matchAndRewrite(::tla::CopyOp op, PatternRewriter &rewriter) const override {
      if (op->getNumOperands() != 2 || op->getNumResults() != 0) {
        op.emitError() << "expected tla.copy to have exactly 2 operands and 0 results";
        return failure();
      }

      Value dstTile = op->getOperand(0);
      Value srcTile = op->getOperand(1);
      auto dstIt = tensorDescriptorByValue.find(dstTile);
      auto srcIt = tensorDescriptorByValue.find(srcTile);
      if (dstIt == tensorDescriptorByValue.end()) {
        op.emitError() << "missing descriptor for tla.copy dst tile; expected tile produced by "
                          "tla.tile_view/tla.make_tensor_like in this pass";
        return failure();
      }
      if (srcIt == tensorDescriptorByValue.end()) {
        op.emitError() << "missing descriptor for tla.copy src tile; expected tile produced by "
                          "tla.tile_view/tla.make_tensor_like in this pass";
        return failure();
      }

      const TensorDescriptor &dstDesc = dstIt->second;
      const TensorDescriptor &srcDesc = srcIt->second;
      if (!TlaLowerToStdPass::validateTensorDescriptorV1(
              op, dstDesc, "malformed descriptor for tla.copy dst tile operand",
              /*requireShapeOperands=*/true)) {
        return failure();
      }
      if (!TlaLowerToStdPass::validateTensorDescriptorV1(
              op, srcDesc, "malformed descriptor for tla.copy src tile operand",
              /*requireShapeOperands=*/true)) {
        return failure();
      }
      StringRef srcAddrspace = srcDesc.addrspace;
      StringRef dstAddrspace = dstDesc.addrspace;
      bool rankOk = dstDesc.rank == srcDesc.rank;
      bool sameElem = dstDesc.elementType == srcDesc.elementType;
      auto buildRuntimeMemref = [&](const TensorDescriptor &desc) -> FailureOr<Value> {
        FailureOr<Value> baseMemref = TlaLowerToStdPass::materializeDescriptorBaseMemref(
            rewriter, op.getLoc(), desc, allocatorState, op.getOperation());
        if (failed(baseMemref))
          return failure();
        auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
        if (!baseType)
          return failure();
        MemRefType runtimeType = TlaLowerToStdPass::getDynamicStridedMemrefType(baseType);
        return TlaLowerToStdPass::castMemrefToType(rewriter, op.getLoc(), *baseMemref,
                                                   runtimeType);
      };
      std::string calleeName = TlaLowerToStdPass::getCopyRouteCallee(
          op.getContext(), srcAddrspace, dstAddrspace, srcDesc.layoutTag, dstDesc.layoutTag,
          srcDesc.elementType, dstDesc.elementType);
      if (!calleeName.empty()) {
        bool l0cGmNarrow = srcAddrspace == "l0c" && dstAddrspace == "gm" &&
                           srcDesc.layoutTag == TensorLayoutTag::L0C &&
                           dstDesc.layoutTag == TensorLayoutTag::RowMajor &&
                           srcDesc.elementType == "f32" &&
                           (dstDesc.elementType == "f16" || dstDesc.elementType == "bf16");
        if (!rankOk || (!sameElem && !l0cGmNarrow)) {
          op.emitError() << "tla.copy supported route has src/dst descriptor metadata mismatch "
                            "(rank/element type)";
          return failure();
        }

        FailureOr<Value> dstRuntimeMemref = buildRuntimeMemref(dstDesc);
        FailureOr<Value> srcRuntimeMemref = buildRuntimeMemref(srcDesc);
        if (failed(dstRuntimeMemref) || failed(srcRuntimeMemref))
          return failure();
        SmallVector<Value, 20> payload =
            TlaLowerToStdPass::buildCopyPayloadForRoute(rewriter, op.getLoc(), srcDesc, dstDesc);
        SmallVector<Type, 22> operandTypes = {(*srcRuntimeMemref).getType(),
                                              (*dstRuntimeMemref).getType()};
        operandTypes.reserve(2 + payload.size());
        for (Value payloadValue : payload)
          operandTypes.push_back(payloadValue.getType());
        auto callee = TlaLowerToStdPass::getOrCreateRuntimeCall(module, calleeName, operandTypes);
        SmallVector<Value, 22> operands = {*srcRuntimeMemref, *dstRuntimeMemref};
        operands.append(payload.begin(), payload.end());
        rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
        toErase.push_back(op.getOperation());
        return success();
      }

      if (rankOk && sameElem && srcDesc.layoutTag == TensorLayoutTag::RowMajor &&
          dstDesc.layoutTag == TensorLayoutTag::RowMajor) {
        FailureOr<Value> dstRuntimeMemref = buildRuntimeMemref(dstDesc);
        FailureOr<Value> srcRuntimeMemref = buildRuntimeMemref(srcDesc);
        if (failed(dstRuntimeMemref) || failed(srcRuntimeMemref))
          return failure();

        if (srcAddrspace == "gm" && dstAddrspace == "ub") {
          auto srcType = dyn_cast<MemRefType>((*srcRuntimeMemref).getType());
          if (!srcType)
            return failure();
          FailureOr<Value> zeroValue =
              TlaLowerToStdPass::makeZeroValue(rewriter, op.getLoc(), srcType.getElementType());
          if (failed(zeroValue))
            return failure();
          auto padModeAttr = rewriter.getAttr<hivm::PadModeAttr>(hivm::PadMode::PadValue);
          Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
          auto load = rewriter.create<hivm::LoadOp>(op.getLoc(), TypeRange{}, *srcRuntimeMemref,
                                                    *dstRuntimeMemref, padModeAttr, *zeroValue,
                                                    zeroIndex);
          load->removeAttr("init_out_buffer");
          load->removeAttr("may_implicit_transpose_with_last_axis");
          toErase.push_back(op.getOperation());
          return success();
        }

        if (srcAddrspace == "ub" && dstAddrspace == "gm") {
          rewriter.create<hivm::StoreOp>(op.getLoc(), TypeRange{}, *srcRuntimeMemref,
                                         *dstRuntimeMemref);
          toErase.push_back(op.getOperation());
          return success();
        }
      }

      op.emitError() << "tla.copy descriptor/layout combination is unsupported: " << srcAddrspace
                     << "(" << TlaLowerToStdPass::stringifyTensorLayoutTag(srcDesc.layoutTag)
                     << ") -> " << dstAddrspace << "("
                     << TlaLowerToStdPass::stringifyTensorLayoutTag(dstDesc.layoutTag) << ")";
      return failure();
    }

  private:
    ModuleOp module;
    DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue;
    SmallVectorImpl<Operation *> &toErase;
    AllocatorOffsetState *allocatorState;
  };

  template <typename OpT> struct LowerTlaTileViewPattern : public OpRewritePattern<OpT> {
    LowerTlaTileViewPattern(MLIRContext *ctx,
                            DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue,
                            AllocatorOffsetState *allocatorState)
        : OpRewritePattern<OpT>(ctx), tensorDescriptorByValue(tensorDescriptorByValue),
          allocatorState(allocatorState) {}

    LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const override {
      if (op->getNumResults() != 1) {
        op.emitError() << "expected tile-view op to have exactly 1 result during subview lowering";
        return failure();
      }

      auto descIt = tensorDescriptorByValue.find(op->getResult(0));
      if (descIt == tensorDescriptorByValue.end()) {
        op.emitError() << "missing descriptor for " << op->getName().getStringRef()
                       << " result during subview lowering";
        return failure();
      }
      const TensorDescriptor &desc = descIt->second;
      if (!TlaLowerToStdPass::validateTensorDescriptorV1(
              op, desc, "malformed descriptor for tile subview lowering",
              /*requireShapeOperands=*/true)) {
        return failure();
      }

      FailureOr<Value> baseMemref = TlaLowerToStdPass::materializeDescriptorBaseMemref(
          rewriter, op.getLoc(), desc, allocatorState, op.getOperation());
      if (failed(baseMemref))
        return failure();
      auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
      if (!baseType) {
        op.emitError() << "expected descriptor base to materialize to memref type";
        return failure();
      }

      Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
      Value subview;
      if (baseType.getRank() == 1) {
        // Flattened 1D buffers are already in the runtime ABI shape expected by
        // downstream lowering; avoid fabricating extra subviews that later pass
        // stages will erase anyway.
        subview = *baseMemref;
      } else {
        subview = rewriter
                      .create<mlir::memref::SubViewOp>(
                          op.getLoc(), *baseMemref, ValueRange{desc.rowOffset, desc.colOffset},
                          ValueRange{desc.shape0, desc.shape1}, ValueRange{one, one})
                      .getResult();
      }
      Value tileView =
          rewriter
              .create<UnrealizedConversionCastOp>(
                  op.getLoc(), TypeRange{op->getResult(0).getType()}, ValueRange{subview})
              .getResult(0);
      tensorDescriptorByValue[tileView] = desc;
      tensorDescriptorByValue.erase(op->getResult(0));
      rewriter.replaceOp(op, tileView);
      return success();
    }

  private:
    DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue;
    AllocatorOffsetState *allocatorState;
  };

  struct LowerTlaCopyLatePattern : public OpRewritePattern<::tla::CopyOp> {
    LowerTlaCopyLatePattern(MLIRContext *ctx, ModuleOp module,
                            SmallVectorImpl<Operation *> &toErase)
        : OpRewritePattern<::tla::CopyOp>(ctx), module(module), toErase(toErase) {}

    LogicalResult matchAndRewrite(::tla::CopyOp op, PatternRewriter &rewriter) const override {
      if (!op->getBlock() || llvm::is_contained(toErase, op.getOperation()))
        return success();
      if (op->getNumOperands() != 2 || op->getNumResults() != 0)
        return success();

      auto dstInfo = TlaLowerToStdPass::decodeTileTypeInfo(op->getOperand(0).getType());
      auto srcInfo = TlaLowerToStdPass::decodeTileTypeInfo(op->getOperand(1).getType());
      if (failed(dstInfo) || failed(srcInfo)) {
        op.emitError()
            << "tla.copy late cleanup requires structured tla.tensor metadata on both operands";
        return failure();
      }

      std::string calleeName = TlaLowerToStdPass::getCopyRouteCallee(
          op.getContext(), srcInfo->addressSpace, dstInfo->addressSpace, srcInfo->layoutTag,
          dstInfo->layoutTag, srcInfo->elementType, dstInfo->elementType);
      if (calleeName.empty()) {
        op.emitError() << "tla.copy descriptor/layout combination is unsupported: "
                       << srcInfo->addressSpace << "("
                       << TlaLowerToStdPass::stringifyTensorLayoutTag(srcInfo->layoutTag) << ") -> "
                       << dstInfo->addressSpace << "("
                       << TlaLowerToStdPass::stringifyTensorLayoutTag(dstInfo->layoutTag) << ")";
        return failure();
      }
      bool sameElem = srcInfo->elementType == dstInfo->elementType;
      bool l0cGmNarrow = srcInfo->addressSpace == "l0c" && dstInfo->addressSpace == "gm" &&
                         srcInfo->layoutTag == TensorLayoutTag::L0C &&
                         dstInfo->layoutTag == TensorLayoutTag::RowMajor &&
                         srcInfo->elementType == "f32" &&
                         (dstInfo->elementType == "f16" || dstInfo->elementType == "bf16");
      if (srcInfo->rank != dstInfo->rank || (!sameElem && !l0cGmNarrow)) {
        op.emitError() << "tla.copy late cleanup found src/dst descriptor metadata mismatch "
                          "(rank/element type)";
        return failure();
      }

      SmallVector<Type, 2> operandTypes = {op->getOperand(1).getType(),
                                           op->getOperand(0).getType()};
      auto callee = TlaLowerToStdPass::getOrCreateRuntimeCall(module, calleeName, operandTypes);
      SmallVector<Value, 2> operands = {op->getOperand(1), op->getOperand(0)};
      rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
      toErase.push_back(op.getOperation());
      return success();
    }

  private:
    ModuleOp module;
    SmallVectorImpl<Operation *> &toErase;
  };

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::StringMap<bool> vectorKernelNames = collectVectorKernelNames(module);

    // Defer erases to avoid invalidating active walks.
    SmallVector<Operation *, 8> toErase;
    AllocatorOffsetState allocatorState;
    allocatorState.toErase = &toErase;

    // Pass-local descriptor map populated by tile-producing ops.
    DenseMap<Value, TensorDescriptor> tensorDescriptorByValue;

    struct ConstantKey {
      int64_t value;
      unsigned bits;

      bool operator==(const ConstantKey &other) const {
        return value == other.value && bits == other.bits;
      }
    };

    struct ConstantKeyInfo {
      static inline ConstantKey getEmptyKey() { return {std::numeric_limits<int64_t>::min(), 0}; }
      static inline ConstantKey getTombstoneKey() {
        return {std::numeric_limits<int64_t>::min() + 1, 0};
      }
      static unsigned getHashValue(const ConstantKey &key) {
        return llvm::hash_combine(key.value, key.bits);
      }
      static bool isEqual(const ConstantKey &lhs, const ConstantKey &rhs) { return lhs == rhs; }
    };
    DenseMap<Block *, DenseMap<ConstantKey, Value, ConstantKeyInfo>> constantByScope;

    auto getOrCreateConstant = [&](Operation *anchor, int64_t value, unsigned bits) -> Value {
      ConstantKey key{value, bits};
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
      auto &cache = constantByScope[scopeBlock];
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

    if (failed(lowerTlaMutexToStd(module, getOrCreateConstant))) {
      signalPassFailure();
      return;
    }

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
    auto buildTileViewResultDescriptorFromParent =
        [&](Operation *op, Value base, MemRefType bridgedBaseType, const TileTypeInfo &info,
            const TensorDescriptor &parent, Value row, Value col, Value sh0,
            Value sh1) -> FailureOr<TensorDescriptor> {
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
          return getOrCreateConstant(op, dim, 0);
        };
        FailureOr<Value> st0 = materializeRowMajorStride(info.strideDims[0], parent.stride0);
        FailureOr<Value> st1 = materializeRowMajorStride(info.strideDims[1], parent.stride1);
        if (failed(st0) || failed(st1))
          return failure();
        stride0 = *st0;
        stride1 = *st1;
        shape0 = info.shapeDims[0] == ShapedType::kDynamic
                     ? sh0
                     : getOrCreateConstant(op, info.shapeDims[0], 0);
        shape1 = info.shapeDims[1] == ShapedType::kDynamic
                     ? sh1
                     : getOrCreateConstant(op, info.shapeDims[1], 0);
      } else {
        auto ceilDivIndexByPositiveConst = [&](Value numerator,
                                               int64_t divisor) -> FailureOr<Value> {
          if (divisor <= 0) {
            op->emitError()
                << "tile_view: packed shape dynamic leaf requires positive divisor, got "
                << divisor;
            return failure();
          }
          Value divisorV = getOrCreateConstant(op, divisor, 0);
          Value one = getOrCreateConstant(op, 1, 0);
          Value adjusted =
              b.create<arith::AddIOp>(loc, numerator, b.create<arith::SubIOp>(loc, divisorV, one));
          return b.create<arith::DivSIOp>(loc, adjusted, divisorV).getResult();
        };
        auto materializePackedShapeLeaf = [&](size_t idx) -> FailureOr<Value> {
          int64_t leaf = info.shapeDims[idx];
          if (leaf != ShapedType::kDynamic)
            return getOrCreateConstant(op, leaf, 0);
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
            return getOrCreateConstant(op, leaf, 0);
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
          signalPassFailure();
          return;
        }

        auto rowColShape = unpackTileOffsetsAndShape(op);
        if (failed(rowColShape)) {
          signalPassFailure();
          return;
        }
        Value row = (*rowColShape)[0];
        Value col = (*rowColShape)[1];
        Value shape0 = (*rowColShape)[2];
        Value shape1 = (*rowColShape)[3];
        if (!row.getType().isIndex() || !col.getType().isIndex() || !shape0.getType().isIndex() ||
            !shape1.getType().isIndex()) {
          op->emitError() << "tla.tile_view row/col/shape operands must be index type";
          signalPassFailure();
          return;
        }

        auto resultInfo = decodeTileTypeInfo(tileOp.getResult().getType());
        if (failed(resultInfo)) {
          op->emitError() << "tla.tile_view currently requires a structured tla.tensor result "
                             "type";
          signalPassFailure();
          return;
        }
        if (resultInfo->rank != 2) {
          op->emitError() << "tla.tile_view descriptor v1 supports only rank-2 tiles";
          signalPassFailure();
          return;
        }

        Value source = tileOp.getOperand(0);
        if (isa<::tla::MemrefType>(source.getType()) || isa<MemRefType>(source.getType())) {
          FailureOr<MemRefType> bridgedBaseType =
              isa<MemRefType>(source.getType())
                  ? FailureOr<MemRefType>(cast<MemRefType>(source.getType()))
                  : bridgeTlaMemrefType(source.getType());
          if (failed(bridgedBaseType)) {
            op->emitError()
                << "tla.tile_view source memref must be bridgeable to builtin memref type";
            signalPassFailure();
            return;
          }

          auto explicitLayout = getExplicitTensorLayoutTagAttr(op);
          if (succeeded(explicitLayout)) {
            if (*explicitLayout != resultInfo->layoutTag) {
              op->emitError() << "tla.tile_view layouttag must match result tensor layout_tag";
              signalPassFailure();
              return;
            }
          } else if (auto layoutTagAttr = op->getAttrOfType<StringAttr>("layouttag")) {
            op->emitError() << "unsupported tla.tile_view layouttag '" << layoutTagAttr.getValue()
                            << "'";
            signalPassFailure();
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
            signalPassFailure();
            return;
          }
          FailureOr<TensorDescriptor> desc = buildTensorDescriptorFromTensorInfo(
              builder, op, source, *bridgedBaseType, *resultInfo, *coordDyn, *originDyn);
          if (failed(desc)) {
            signalPassFailure();
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
            signalPassFailure();
            return;
          }
          const TensorDescriptor &parent = parentIt->second;
          if (!validateTensorDescriptorV1(
                  op, parent, "malformed parent tensor descriptor for tla.tile_view source tile",
                  /*requireShapeOperands=*/false)) {
            signalPassFailure();
            return;
          }
          if (resultInfo->rank != parent.rank || resultInfo->addressSpace != parent.addrspace ||
              resultInfo->elementType != parent.elementType) {
            op->emitError() << "tla.tile_view result tile metadata must match parent descriptor "
                               "(rank/element type/addrspace) when source is a tile";
            signalPassFailure();
            return;
          }

          auto explicitLayout = getExplicitTensorLayoutTagAttr(op);
          if (succeeded(explicitLayout)) {
            if (*explicitLayout != resultInfo->layoutTag) {
              op->emitError() << "tla.tile_view layouttag must match result tensor layout_tag";
              signalPassFailure();
              return;
            }
          } else if (auto layoutTagAttr = op->getAttrOfType<StringAttr>("layouttag")) {
            op->emitError() << "unsupported tla.tile_view layouttag '" << layoutTagAttr.getValue()
                            << "'";
            signalPassFailure();
            return;
          }

          auto bridgedParent = dyn_cast<MemRefType>(parent.bridgedBaseMemrefType);
          if (!bridgedParent) {
            op->emitError() << "tla.tile_view parent descriptor missing bridged memref type";
            signalPassFailure();
            return;
          }
          FailureOr<TensorDescriptor> desc = buildTileViewResultDescriptorFromParent(
              op, parent.base, bridgedParent, *resultInfo, parent, row, col, shape0, shape1);
          if (failed(desc)) {
            signalPassFailure();
            return;
          }
          tensorDescriptorByValue[tileOp.getResult()] = *desc;
          return;
        }

        op->emitError() << "tla.tile_view source must be either !tla.memref or !tla.tensor";
        signalPassFailure();
        return;
      }

      if (llvm::isa<::tla::MakeTensorLikeOp>(op)) {
        if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
          op->emitError()
              << "expected tla.make_tensor_like to have exactly 2 operands and 1 result";
          signalPassFailure();
          return;
        }

        Value ptrValue = op->getOperand(0);
        if (!llvm::isa<::tla::PtrType>(ptrValue.getType())) {
          op->emitError() << "tla.make_tensor_like pointer operand must be !tla.ptr";
          signalPassFailure();
          return;
        }

        Value likeTile = op->getOperand(1);
        auto parentIt = tensorDescriptorByValue.find(likeTile);
        if (parentIt == tensorDescriptorByValue.end()) {
          op->emitError()
              << "missing descriptor for tla.make_tensor_like reference tile; expected source "
                 "to be produced by tla.tile_view/tla.make_tensor_like in this pass";
          signalPassFailure();
          return;
        }
        const TensorDescriptor &parent = parentIt->second;
        if (!validateTensorDescriptorV1(
                op, parent,
                "malformed parent tensor descriptor for tla.make_tensor_like reference tile",
                /*requireShapeOperands=*/true)) {
          signalPassFailure();
          return;
        }

        auto childInfo = decodeTileTypeInfo(op->getResult(0).getType());
        if (failed(childInfo)) {
          op->emitError() << "tla.make_tensor_like currently requires a structured tla.tensor "
                             "result type";
          signalPassFailure();
          return;
        }
        if (childInfo->rank != parent.rank) {
          op->emitError()
              << "tla.make_tensor_like result tile rank must match reference descriptor "
                 "(rank)";
          signalPassFailure();
          return;
        }

        // Buffer element count for the synthetic !tla.memref type below. Today this path only
        // supports 1D/2D logical tiles: either we read a static 1D length from an HIVM
        // pointer-cast bridge, or we multiply the first two origin_shape dimensions. If we
        // ever need higher-rank tiles here, we will likely flatten the backing memref to 1D
        // and drive everything from a single linear element count.
        int64_t flatElemCount = ShapedType::kDynamic;
        if (auto n = getStatic1DElementCountFromHivmPtrBridge(ptrValue); succeeded(n) && *n > 0) {
          flatElemCount = *n;
        } else if (childInfo->originShapeDims.size() >= 2 &&
                   childInfo->originShapeDims[0] != ShapedType::kDynamic &&
                   childInfo->originShapeDims[1] != ShapedType::kDynamic) {
          int64_t dim0 = childInfo->originShapeDims[0];
          int64_t dim1 = childInfo->originShapeDims[1];
          if (dim0 > 0 && dim1 > 0)
            flatElemCount = dim0 * dim1;
        }
        Type typedBufferType =
            ::tla::MemrefType::get(op->getContext(), {flatElemCount}, childInfo->mlirElementType,
                                   childInfo->tlaAddressSpace);
        auto bridgedBaseType = bridgeTlaMemrefType(typedBufferType);
        if (failed(bridgedBaseType)) {
          op->emitError()
              << "tla.make_tensor_like buffer memref must be bridgeable to builtin memref type";
          signalPassFailure();
          return;
        }

        OpBuilder builder(op);
        auto layoutTagAttr = op->getAttrOfType<StringAttr>("layoutTag");
        if (!layoutTagAttr)
          layoutTagAttr = op->getAttrOfType<StringAttr>("layouttag");
        if (!layoutTagAttr) {
          op->emitError() << "tla.make_tensor_like requires a layoutTag attribute";
          signalPassFailure();
          return;
        }
        auto layoutTag = parseTensorLayoutTagAttr(layoutTagAttr.getValue());
        if (failed(layoutTag)) {
          op->emitError() << "unsupported tla.make_tensor_like layoutTag '"
                          << layoutTagAttr.getValue() << "'";
          signalPassFailure();
          return;
        }
        if (*layoutTag != childInfo->layoutTag) {
          op->emitError() << "tla.make_tensor_like layoutTag must match result tensor layout_tag";
          signalPassFailure();
          return;
        }
        Value typedBuffer = builder
                                .create<UnrealizedConversionCastOp>(
                                    op->getLoc(), TypeRange{typedBufferType}, ValueRange{ptrValue})
                                .getResult(0);
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
          signalPassFailure();
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
            signalPassFailure();
            return;
          }
          shape0 = *shape0Or;
          shape1 = *shape1Or;
          stride0 = *stride0Or;
          stride1 = *stride1Or;
        } else {
          if (!isPackedLayout(childInfo->layoutTag)) {
            op->emitError() << "unsupported tla.make_tensor_like layout for descriptor v1";
            signalPassFailure();
            return;
          }
          packedShape.reserve(childInfo->shapeDims.size());
          packedStride.reserve(childInfo->strideDims.size());
          for (size_t i = 0; i < childInfo->shapeDims.size(); ++i) {
            FailureOr<Value> leaf = materializePackedLeafFromTypeOrParent(
                childInfo->shapeDims, parent.packedShape, i, "packed shape");
            if (failed(leaf)) {
              signalPassFailure();
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
              signalPassFailure();
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
    });

    // Stage 2: descriptor-driven tla.copy lowering.
    // Supported v1 routes emit runtime calls.
    // Unsupported combinations stay as tla.copy with explicit remarks.
    LowerTlaCopyPattern lowerCopy(&getContext(), module, tensorDescriptorByValue, toErase,
                                  &allocatorState);
    SmallVector<::tla::CopyOp, 16> copyOps;
    module.walk([&](::tla::CopyOp op) { copyOps.push_back(op); });
    bool copyLoweringFailed = false;
    for (::tla::CopyOp op : copyOps) {
      if (!op || !op->getBlock())
        continue;
      PatternRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      if (failed(lowerCopy.matchAndRewrite(op, rewriter))) {
        copyLoweringFailed = true;
      }
    }
    if (copyLoweringFailed)
      signalPassFailure();

    stageDeadTileProducers(module, toErase);

    // Stage 2.6: lower tile-view producers to memref.subview
    // on bridged bases.
    // Cast-bridge around subview keeps current tile-typed consumers valid until
    // a full Tla->memref conversion boundary exists.
    LowerTlaTileViewPattern<::tla::TileViewOp> lowerTileView(&getContext(), tensorDescriptorByValue,
                                                             &allocatorState);
    LowerTlaTileViewPattern<::tla::MakeTensorLikeOp> lowerMakeTensorLikeView(
        &getContext(), tensorDescriptorByValue, &allocatorState);
    SmallVector<Operation *, 8> tileViewOps;
    module.walk([&](Operation *op) {
      if (llvm::isa<::tla::TileViewOp, ::tla::MakeTensorLikeOp>(op))
        tileViewOps.push_back(op);
    });
    for (Operation *op : tileViewOps) {
      if (!op || !op->getBlock() || llvm::is_contained(toErase, op))
        continue;
      PatternRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);
      LogicalResult lowered = success();
      if (auto tileOp = llvm::dyn_cast<::tla::TileViewOp>(op)) {
        lowered = lowerTileView.matchAndRewrite(tileOp, rewriter);
      } else if (auto makeTensorLikeOp = llvm::dyn_cast<::tla::MakeTensorLikeOp>(op)) {
        lowered = lowerMakeTensorLikeView.matchAndRewrite(makeTensorLikeOp, rewriter);
      }
      if (failed(lowered)) {
        signalPassFailure();
        return;
      }
    }

    // Stage 7: erase dead tile-construction handles after loop conversion.
    SmallVector<Operation *, 8> deadMakeTensorLikeOps;
    module.walk([&](::tla::MakeTensorLikeOp op) {
      if (op.getResult().use_empty())
        deadMakeTensorLikeOps.push_back(op.getOperation());
    });
    for (Operation *makeTensorLikeOp : deadMakeTensorLikeOps) {
      if (!makeTensorLikeOp->getBlock() || llvm::is_contained(toErase, makeTensorLikeOp))
        continue;
      makeTensorLikeOp->erase();
    }

    // Stage 6A: flatten region wrappers while preserving dispatch metadata for
    // ops that still depend on wrapper execution units.
    // Dispatch policy:
    //   tla.cube   => tla.exec_unit = "cube"
    //   tla.vector => tla.exec_unit = "vector"
    // This metadata is consumed by Stage 6B vector rewrites.
    LowerTlaCubePattern lowerCube(&getContext());
    LowerTlaVectorPattern lowerVector(&getContext());
    SmallVector<Operation *, 16> wrapperOps;
    module.walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (llvm::isa<::tla::CubeOp, ::tla::VectorOp>(op))
        wrapperOps.push_back(op);
    });
    for (Operation *op : wrapperOps) {
      if (!op || !op->getBlock())
        continue;
      PatternRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);
      LogicalResult lowered = success();
      if (auto cubeOp = llvm::dyn_cast<::tla::CubeOp>(op)) {
        lowered = lowerCube.matchAndRewrite(cubeOp, rewriter);
      } else if (auto vectorOp = llvm::dyn_cast<::tla::VectorOp>(op)) {
        lowered = lowerVector.matchAndRewrite(vectorOp, rewriter);
      }
      if (failed(lowered)) {
        signalPassFailure();
        return;
      }
    }

    // Stage 8B: lower residual compute ops.
    LowerTlaMmadPattern lowerMmad(&getContext(), module, tensorDescriptorByValue, toErase);
    LowerTlaGmAddPattern lowerGmAdd(&getContext(), module, toErase);
    LowerTlaAddPattern lowerAdd(&getContext(), module, toErase);
    SmallVector<Operation *, 16> execUnitOps;
    module.walk([&](Operation *op) {
      if (llvm::isa<::tla::MmadOp, ::tla::GmAddOp, ::tla::AddOp>(op))
        execUnitOps.push_back(op);
    });
    for (Operation *op : execUnitOps) {
      if (!op->getBlock())
        continue;
      PatternRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);
      LogicalResult lowered = success();
      if (auto mmadOp = llvm::dyn_cast<::tla::MmadOp>(op)) {
        lowered = lowerMmad.matchAndRewrite(mmadOp, rewriter);
      } else if (auto gmAddOp = llvm::dyn_cast<::tla::GmAddOp>(op)) {
        lowered = lowerGmAdd.matchAndRewrite(gmAddOp, rewriter);
      } else if (auto addOp = llvm::dyn_cast<::tla::AddOp>(op)) {
        lowered = lowerAdd.matchAndRewrite(addOp, rewriter);
      }
      if (failed(lowered)) {
        signalPassFailure();
        return;
      }
    }

    // Stage 8C: late tla.copy cleanup for ops introduced by loop cloning.
    // This path is type-driven (no descriptor map) and lowers only routes with
    // structured tile metadata and matching rank/element type.
    LowerTlaCopyLatePattern lowerCopyLate(&getContext(), module, toErase);
    SmallVector<::tla::CopyOp, 16> lateCopyOps;
    module.walk([&](::tla::CopyOp op) { lateCopyOps.push_back(op); });
    bool lateCopyLoweringFailed = false;
    for (::tla::CopyOp op : lateCopyOps) {
      if (!op || !op->getBlock() || llvm::is_contained(toErase, op.getOperation()))
        continue;
      PatternRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      if (failed(lowerCopyLate.matchAndRewrite(op, rewriter))) {
        lateCopyLoweringFailed = true;
      }
    }
    if (lateCopyLoweringFailed)
      signalPassFailure();

    // Stage P: typed rewrite patterns for simple Tla ops without global
    // greedy simplification so output structure remains stable for fixtures.
    LowerTlaSplatPattern lowerSplat(&getContext());
    LowerTlaReturnToFuncReturnPattern lowerReturn(&getContext());
    LowerTlaFuncToFuncPattern<LowerTlaFuncToFuncAttrPolicy::OmitAttrs> lowerTlaFunc(&getContext());
    auto applyPattern = [&](auto op, auto &pattern) -> LogicalResult {
      if (!op->getBlock())
        return success();
      PatternRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      return pattern.matchAndRewrite(op, rewriter);
    };

    SmallVector<::tla::SplatOp, 8> splatOps;
    module.walk([&](::tla::SplatOp op) { splatOps.push_back(op); });
    for (::tla::SplatOp op : splatOps) {
      if (failed(applyPattern(op, lowerSplat))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<::tla::ReturnOp, 8> returnOps;
    module.walk([&](::tla::ReturnOp op) { returnOps.push_back(op); });
    for (::tla::ReturnOp op : returnOps) {
      if (failed(applyPattern(op, lowerReturn))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<::tla::FuncOp, 8> tlaFuncOps;
    module.walk([&](::tla::FuncOp op) { tlaFuncOps.push_back(op); });
    for (::tla::FuncOp op : tlaFuncOps) {
      if (failed(applyPattern(op, lowerTlaFunc))) {
        signalPassFailure();
        return;
      }
    }

    injectVectorCtrlPrologueIntoFuncs(module, vectorKernelNames);

    // Final cleanup for all staged rewrites.
    DenseSet<Operation *> pendingErase;
    for (Operation *op : toErase) {
      if (op && op->getBlock())
        pendingErase.insert(op);
    }
    bool progress = true;
    while (progress && !pendingErase.empty()) {
      progress = false;
      for (Operation *op : toErase) {
        if (!op || !pendingErase.contains(op) || !op->getBlock())
          continue;
        bool hasLiveResultUses = false;
        for (Value result : op->getResults()) {
          if (!result.use_empty()) {
            hasLiveResultUses = true;
            break;
          }
        }
        if (hasLiveResultUses)
          continue;
        pendingErase.erase(op);
        op->erase();
        progress = true;
      }
    }
    if (!pendingErase.empty()) {
      for (Operation *op : pendingErase) {
        unsigned liveUsers = 0;
        for (Value result : op->getResults())
          liveUsers += std::distance(result.getUsers().begin(), result.getUsers().end());
        op->emitError() << "staged erase failed for '" << op->getName().getStringRef()
                        << "': operation still has " << liveUsers << " live result users";
      }
      signalPassFailure();
      return;
    }

    if (failed(bridgeFuncTensorEntryAbi(module))) {
      signalPassFailure();
      return;
    }
    if (failed(rewriteTensorTypedFuncCalls(module))) {
      signalPassFailure();
      return;
    }

    ConversionTarget target(getContext());
    target.addLegalDialect<::tla::TlaDialect, arith::ArithDialect, func::FuncDialect,
                           mlir::memref::MemRefDialect, scf::SCFDialect, hivm::HIVMDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<::tla::TileViewOp, ::tla::CopyOp, ::tla::MakeTensorLikeOp,
                        ::tla::LoadOp, ::tla::StoreOp, ::tla::FuncOp,
                        ::tla::ReturnOp, ::tla::SplatOp, ::tla::MutexOp,
                        ::tla::MutexLockOp, ::tla::MutexUnlockOp, ::tla::CrossFlagOp,
                        ::tla::CrossCoreSetFlagOp, ::tla::CrossCoreWaitFlagOp, ::tla::CubeOp,
                        ::tla::VectorOp, ::tla::MmadOp, ::tla::AddOp>();
    target.addDynamicallyLegalOp<::tla::MakeShapeOp, ::tla::MakeCoordOp, ::tla::MakeStrideOp,
                                 ::tla::MakeLayoutOp, ::tla::AllocPtrOp, ::tla::RecastPtrOp>(
        [](Operation *op) { return !hasNoResultUses(op); });
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](UnrealizedConversionCastOp op) { return !isDeadTensorBridgeCast(op); });
    target.addDynamicallyLegalOp<mlir::memref::SubViewOp>(
        [](mlir::memref::SubViewOp op) { return !hasNoResultUses(op.getOperation()); });
#if defined(TLA_DSL_ENABLE_HIVM)
    target.addDynamicallyLegalOp<hivm::PointerCastOp>(
        [](hivm::PointerCastOp op) { return !hasNoResultUses(op.getOperation()); });
#endif

    for (int cleanupPass = 0; cleanupPass < 2; ++cleanupPass) {
      RewritePatternSet patterns(&getContext());
      patterns
          .add<EraseDeadTensorBridgeCastPattern, EraseDeadOpPattern<::tla::MakeShapeOp>,
               EraseDeadOpPattern<::tla::MakeCoordOp>, EraseDeadOpPattern<::tla::MakeStrideOp>,
               EraseDeadOpPattern<::tla::MakeLayoutOp>, EraseDeadOpPattern<::tla::AllocPtrOp>,
               EraseDeadOpPattern<::tla::RecastPtrOp>, EraseDeadOpPattern<mlir::memref::SubViewOp>
#if defined(TLA_DSL_ENABLE_HIVM)
               ,
               EraseDeadOpPattern<hivm::PointerCastOp>
#endif
               >(&getContext());
      if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
    eraseDeadMaterializations(module);
  }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerToStdPass() { return std::make_unique<TlaLowerToStdPass>(); }

void registerTlaLowerToStdPass() { PassRegistration<TlaLowerToStdPass>(); }

} // namespace tla
