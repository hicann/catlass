#include "TlaScratchAllocation.h"

#include "PassesCommon.h"

namespace tla {
namespace {

static FailureOr<uint64_t> alignUpCheckedU64(uint64_t value, uint64_t alignment)
{
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

} // namespace

const TlaScratchAllocation* TlaScratchAllocationPlan::lookup(mlir::Value root) const
{
    auto it = indexByRoot.find(root);
    if (it == indexByRoot.end() || it->second >= allocations.size())
        return nullptr;
    return &allocations[it->second];
}

FailureOr<TlaScratchAllocationPlan> planTlaScratchAllocations(ModuleOp module)
{
    TlaScratchAllocationPlan plan;
    llvm::StringMap<uint64_t> nextOffsetByAddrspace;
    SmallVector<::tla::AllocPtrOp, 8> allocs;
    module.walk([&](::tla::AllocPtrOp op) { allocs.push_back(op); });

    for (::tla::AllocPtrOp allocOp : allocs) {
        auto ptrTy = dyn_cast<::tla::PtrType>(allocOp.getResult().getType());
        int64_t signedSize = allocOp.getSizeBytesAttr().getInt();
        if (!ptrTy || signedSize <= 0 || ptrTy.getAlignment() == 0) {
            allocOp.emitError("failed to plan static scratch allocation");
            return failure();
        }

        uint64_t sizeBytes = static_cast<uint64_t>(signedSize);
        uint64_t alignment = ptrTy.getAlignment();
        std::string addressSpaceKey = ::stringifyAddressSpace(ptrTy.getAddrspace()).str();
        FailureOr<uint64_t> base = alignUpCheckedU64(nextOffsetByAddrspace[addressSpaceKey], alignment);
        FailureOr<uint64_t> alignedSize = alignUpCheckedU64(sizeBytes, alignment);
        if (failed(base) || failed(alignedSize) || *base > std::numeric_limits<uint64_t>::max() - sizeBytes ||
            *base > std::numeric_limits<uint64_t>::max() - *alignedSize) {
            allocOp.emitError("failed to assign a static scratch byte interval");
            return failure();
        }

        unsigned index = plan.allocations.size();
        plan.allocations.push_back(TlaScratchAllocation{
            allocOp, allocOp.getResult(), ptrTy.getAddrspace(), alignment, *base, sizeBytes, *base + sizeBytes});
        plan.indexByRoot[allocOp.getResult()] = index;
        plan.offsetByAllocResult[allocOp.getResult()] = *base;
        nextOffsetByAddrspace[addressSpaceKey] = *base + *alignedSize;
    }
    return plan;
}

} // namespace tla
