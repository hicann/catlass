#include "Passes/TlaTensorDescriptor.h"

#include "PassesCommon.h"

#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace tla {

mlir::FailureOr<mlir::MemRefType> bridgeTlaTensorType(mlir::Type tlaTensorType)
{
    return bridgeTlaFuncTensorType(tlaTensorType);
}

bool isPackedLayout(TensorLayoutTag layoutTag)
{
    return layoutTag == TensorLayoutTag::zN || layoutTag == TensorLayoutTag::zZ || layoutTag == TensorLayoutTag::nZ ||
           layoutTag == TensorLayoutTag::L0C;
}

bool isLinearLayout(TensorLayoutTag layoutTag)
{
    return layoutTag == TensorLayoutTag::RowMajor || layoutTag == TensorLayoutTag::ColumnMajor;
}

llvm::StringRef stringifyTensorLayoutTag(TensorLayoutTag layoutTag)
{
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

mlir::FailureOr<TensorLayoutTag> convertTlaLayoutTag(::LayoutTag layoutTag)
{
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

mlir::FailureOr<TensorLayoutTag> parseTensorLayoutTagAttr(llvm::StringRef layouttag)
{
    auto layoutTag = symbolizeLayoutTag(layouttag);
    if (!layoutTag)
        return failure();
    return convertTlaLayoutTag(*layoutTag);
}

mlir::FailureOr<TensorLayoutTag> getExplicitTensorLayoutTagAttr(mlir::Operation* op)
{
    auto layoutTagAttr = op->getAttrOfType<StringAttr>("layouttag");
    if (!layoutTagAttr)
        return failure();
    return parseTensorLayoutTagAttr(layoutTagAttr.getValue());
}

mlir::FailureOr<TileTypeInfo> decodeTileTypeInfo(mlir::Type tileType)
{
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
            info.strideDims = {extent == ShapedType::kDynamic ? ShapedType::kDynamic : stride * extent, stride};
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

mlir::FailureOr<ParsedTensorInfo> parseTensorInfo(mlir::Type tensorType)
{
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

mlir::MemRefType getDynamicStridedMemrefType(mlir::MemRefType memrefType)
{
    SmallVector<int64_t, 4> dynamicShape(memrefType.getRank(), ShapedType::kDynamic);
    SmallVector<int64_t, 4> dynamicStrides(memrefType.getRank(), ShapedType::kDynamic);
    auto layout = StridedLayoutAttr::get(memrefType.getContext(), ShapedType::kDynamic, dynamicStrides);
    return MemRefType::get(dynamicShape, memrefType.getElementType(), layout, memrefType.getMemorySpace());
}

bool validateTensorDescriptorV1(
    mlir::Operation* op, const TensorDescriptor& desc, llvm::StringRef errorMessage, bool requireShapeOperands)
{
    if (!desc.bridgedBaseMemrefType || desc.rank != 2 || desc.addrspace.empty() || desc.elementType.empty() ||
        !desc.rowOffset.getType().isIndex() || !desc.colOffset.getType().isIndex() ||
        !desc.stride0.getType().isIndex() || !desc.stride1.getType().isIndex() ||
        !desc.originShape0.getType().isIndex() || !desc.originShape1.getType().isIndex() ||
        !desc.absCoord0.getType().isIndex() || !desc.absCoord1.getType().isIndex()) {
        op->emitError() << errorMessage;
        return false;
    }
    if (requireShapeOperands && (!desc.shape0.getType().isIndex() || !desc.shape1.getType().isIndex())) {
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

// Allocation capacity is optional provenance, not part of pointer identity.
// Preserve it only across joins whose alternatives prove the same capacity;
// otherwise tensor consumers build a view from their own shape/layout.
static FailureOr<int64_t> inferStaticAllocationSizeBytes(
    Value address, llvm::DenseSet<Value> visiting,
    llvm::DenseMap<Value, int64_t> assumedCapacities = llvm::DenseMap<Value, int64_t>())
{
    if (auto assumed = assumedCapacities.find(address); assumed != assumedCapacities.end())
        return assumed->second;
    if (!visiting.insert(address).second)
        return failure();

    if (auto blockArg = dyn_cast<BlockArgument>(address)) {
        if (auto forOp = dyn_cast_or_null<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
            unsigned argNumber = blockArg.getArgNumber();
            if (blockArg.getOwner() == forOp.getBody() && argNumber > 0 && argNumber - 1 < forOp.getInitArgs().size()) {
                unsigned iterArgNumber = argNumber - 1;
                auto initSize =
                    inferStaticAllocationSizeBytes(forOp.getInitArgs()[iterArgNumber], visiting, assumedCapacities);
                auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
                if (failed(initSize) || !yield || iterArgNumber >= yield.getNumOperands())
                    return failure();

                // Prove the init capacity is a loop invariant. The current block
                // argument may occur recursively in the backedge expression, so use
                // the candidate capacity while checking that every yielded source
                // preserves it.
                assumedCapacities[blockArg] = *initSize;
                auto yieldSize = inferStaticAllocationSizeBytes(
                    yield.getOperand(iterArgNumber), std::move(visiting), std::move(assumedCapacities));
                if (succeeded(yieldSize) && *yieldSize == *initSize)
                    return *initSize;
            }
        }
        return failure();
    }

    Operation* def = address.getDefiningOp();
    if (!def)
        return failure();
    if (auto size = def->getAttrOfType<IntegerAttr>(kAllocSizeBytesMetadataAttrName))
        return size.getInt();
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
        if (cast.getNumOperands() == 1)
            return inferStaticAllocationSizeBytes(
                cast.getOperand(0), std::move(visiting), std::move(assumedCapacities));
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
        auto thenSize = inferStaticAllocationSizeBytes(thenYield.getOperand(resultNumber), visiting, assumedCapacities);
        auto elseSize = inferStaticAllocationSizeBytes(
            elseYield.getOperand(resultNumber), std::move(visiting), std::move(assumedCapacities));
        if (succeeded(thenSize) && succeeded(elseSize) && *thenSize == *elseSize)
            return *thenSize;
        return failure();
    }
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
        unsigned resultNumber = cast<OpResult>(address).getResultNumber();
        if (resultNumber >= forOp.getInitArgs().size())
            return failure();
        auto initSize = inferStaticAllocationSizeBytes(forOp.getInitArgs()[resultNumber], visiting, assumedCapacities);
        auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        if (failed(initSize) || !yield || resultNumber >= yield.getNumOperands())
            return failure();

        assumedCapacities[forOp.getRegionIterArg(resultNumber)] = *initSize;
        auto yieldSize = inferStaticAllocationSizeBytes(
            yield.getOperand(resultNumber), std::move(visiting), std::move(assumedCapacities));
        if (succeeded(yieldSize) && *initSize == *yieldSize)
            return *initSize;
    }
    return failure();
}

FailureOr<int64_t> getStaticAllocationElementCount(Value ptr)
{
    auto ptrType = dyn_cast<::tla::PtrType>(ptr.getType());
    auto intToPtr = ptr.getDefiningOp<::tla::IntToPtrOp>();
    if (!ptrType || !intToPtr)
        return failure();
    auto sizeBytes = inferStaticAllocationSizeBytes(intToPtr.getAddr(), {});
    int64_t elementBytes = getByteSizeOfFixedWidthScalarType(ptrType.getPointee());
    if (failed(sizeBytes) || *sizeBytes < 0 || elementBytes <= 0 || *sizeBytes % elementBytes != 0)
        return failure();
    return *sizeBytes / elementBytes;
}

mlir::FailureOr<TensorDescriptor> descriptorFromTensorDescOp(::tla::TensorDescOp descOp)
{
    auto info = decodeTileTypeInfo(descOp.getResult().getType());
    if (failed(info))
        return descOp->emitError() << "tla.tensor_desc result is not a structured tla.tensor", failure();
    TensorDescriptor d;
    d.base = descOp.getBase();
    if (auto memrefType = llvm::dyn_cast<MemRefType>(descOp.getBase().getType())) {
        d.bridgedBaseMemrefType = memrefType;
    } else if (llvm::isa<::tla::PtrType>(descOp.getBase().getType())) {
        // ptr-backed tile: a flat rank-1 memref of the backing element count, matching
        // the tla.make_tensor{,_like} derivation (buildHivmMemrefType). The count comes
        // from the inttoptr boundary's tla.alloc_size_bytes provenance (set by
        // tla-lower-ptr), falling back to origin_shape0 * origin_shape1.
        int64_t flatElemCount = ShapedType::kDynamic;
        if (auto n = getStaticAllocationElementCount(descOp.getBase()); succeeded(n) && *n > 0) {
            flatElemCount = *n;
        } else if (
            info->originShapeDims.size() >= 2 && info->originShapeDims[0] != ShapedType::kDynamic &&
            info->originShapeDims[1] != ShapedType::kDynamic && info->originShapeDims[0] > 0 &&
            info->originShapeDims[1] > 0) {
            flatElemCount = info->originShapeDims[0] * info->originShapeDims[1];
        }
        auto bridged =
            buildHivmMemrefType(descOp.getContext(), {flatElemCount}, info->mlirElementType, info->tlaAddressSpace);
        if (failed(bridged))
            return failure();
        d.bridgedBaseMemrefType = *bridged;
    } else {
        return descOp->emitError() << "base must be memref or !tla.ptr", failure();
    }
    d.rowOffset = descOp.getRowOffset();
    d.colOffset = descOp.getColOffset();
    d.stride0 = descOp.getStride0();
    d.stride1 = descOp.getStride1();
    d.shape0 = descOp.getShape0();
    d.shape1 = descOp.getShape1();
    d.originShape0 = descOp.getOriginShape0();
    d.originShape1 = descOp.getOriginShape1();
    d.absCoord0 = descOp.getRowOffset();
    d.absCoord1 = descOp.getColOffset();
    d.layoutTag = info->layoutTag;
    d.addrspace = info->addressSpace;
    d.elementType = info->elementType;
    d.rank = info->rank;
    auto packed = descOp.getPacked();
    if (packed.size() == 8) {
        d.packedShape.assign(packed.begin(), packed.begin() + 4);
        d.packedStride.assign(packed.begin() + 4, packed.end());
    }
    return d;
}

} // namespace tla
