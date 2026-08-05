#pragma once

#include "Dialect/Tla/IR/TlaOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace tla {

// One statically assigned, whole-kernel scratch allocation. `base` and `end`
// are byte offsets in `addressSpace`; `end` is exclusive and describes the
// physical bytes requested by the allocation (padding before the next
// allocation is not part of the interval).
struct TlaScratchAllocation {
    ::tla::AllocPtrOp op;
    mlir::Value root;
    ::AddressSpace addressSpace;
    uint64_t alignment;
    uint64_t base;
    uint64_t sizeBytes;
    uint64_t end;
};

struct TlaScratchAllocationPlan {
    llvm::SmallVector<TlaScratchAllocation, 8> allocations;
    llvm::DenseMap<mlir::Value, unsigned> indexByRoot;
    llvm::DenseMap<mlir::Value, uint64_t> offsetByAllocResult;

    const TlaScratchAllocation* lookup(mlir::Value root) const;
};

// Compute the scratch byte-offset plan consumed by tla-lower-ptr. Keeping this
// plan shared is important: automatic synchronization IDs must describe
// the exact physical allocations whose byte offsets are emitted later.
mlir::FailureOr<TlaScratchAllocationPlan> planTlaScratchAllocations(mlir::ModuleOp module);

} // namespace tla
