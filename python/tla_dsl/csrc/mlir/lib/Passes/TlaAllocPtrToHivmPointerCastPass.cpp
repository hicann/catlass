#include "PassesCommon.h"
#include "PassesInternal.h"

namespace tla {
namespace {

static FailureOr<uint64_t> alignUpCheckedU64(uint64_t value, uint64_t alignment) {
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

struct TlaAllocPtrOffsetState {
  llvm::StringMap<uint64_t> nextOffsetByAddrspace;
  llvm::DenseMap<mlir::Value, uint64_t> offsetByAllocResult;
};

static FailureOr<uint64_t> assignOrGetAllocPtrOffsetForPass(::tla::AllocPtrOp allocOp,
                                                            TlaAllocPtrOffsetState &state) {
  auto ptrTy = dyn_cast<::tla::PtrType>(allocOp.getResult().getType());
  if (!ptrTy)
    return failure();

  auto cached = state.offsetByAllocResult.find(allocOp.getResult());
  if (cached != state.offsetByAllocResult.end())
    return cached->second;

  int64_t sizeBytes = allocOp.getSizeBytesAttr().getInt();
  if (sizeBytes < 0)
    return failure();
  uint64_t alignment = ptrTy.getAlignment();
  std::string addrspaceKey = ::stringifyAddressSpace(ptrTy.getAddrspace()).str();
  FailureOr<uint64_t> start =
      alignUpCheckedU64(state.nextOffsetByAddrspace[addrspaceKey], alignment);
  FailureOr<uint64_t> alignedSize = alignUpCheckedU64(static_cast<uint64_t>(sizeBytes), alignment);
  if (failed(start) || failed(alignedSize) ||
      *start > std::numeric_limits<uint64_t>::max() - *alignedSize) {
    return failure();
  }

  state.offsetByAllocResult[allocOp.getResult()] = *start;
  state.nextOffsetByAddrspace[addrspaceKey] = *start + *alignedSize;
  return *start;
}

static FailureOr<MemRefType> hivmMemref1D(MLIRContext *ctx, int64_t numElements, Type elementType,
                                          ::AddressSpace addr) {
  FailureOr<Attribute> memspace = mapTlaAddressSpaceToHivmMemspace(ctx, addr);
  if (failed(memspace))
    return failure();
  return MemRefType::get({numElements}, elementType, AffineMap(), *memspace);
}

static ::tla::AllocPtrOp traceToAllocForRecast(Value v) {
  if (auto alloc = v.getDefiningOp<::tla::AllocPtrOp>())
    return alloc;
  if (auto rec = v.getDefiningOp<::tla::RecastPtrOp>()) {
    if (rec->getNumOperands() == 0)
      return nullptr;
    return traceToAllocForRecast(rec.getSrc());
  }
  return nullptr;
}

class TlaAllocPtrToHivmPointerCastPass
    : public PassWrapper<TlaAllocPtrToHivmPointerCastPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaAllocPtrToHivmPointerCastPass)

  StringRef getArgument() const override { return "tla-alloc-ptr-to-hivm-pointer-cast"; }
  StringRef getName() const override { return "TlaAllocPtrToHivmPointerCastPass"; }
  StringRef getDescription() const override {
    return "Lower tla.alloc_ptr / tla.recast_ptr to hivm.hir.pointer_cast with 1D memref types, "
           "and bridge back to !tla.ptr via tla.hivm_memref_as_ptr.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect, arith::ArithDialect, func::FuncDialect, ::tla::TlaDialect,
                    ::mlir::memref::MemRefDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    TlaAllocPtrOffsetState st;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.getBody().empty())
        continue;
      {
        SmallVector<::tla::AllocPtrOp, 8> allocs;
        func.walk([&](::tla::AllocPtrOp a) { allocs.push_back(a); });
        for (::tla::AllocPtrOp a : allocs) {
          if (failed(assignOrGetAllocPtrOffsetForPass(a, st))) {
            a.emitError() << "alloc slot assignment failed in tla-alloc-ptr-to-hivm-pointer-cast";
            signalPassFailure();
            return;
          }
        }
      }

      SmallVector<::tla::RecastPtrOp, 8> recasts;
      func.walk([&](::tla::RecastPtrOp r) { recasts.push_back(r); });
      for (::tla::RecastPtrOp recast : recasts) {
        if (!recast->getBlock())
          continue;
        if (recast->getNumResults() != 1 || recast->getNumOperands() != 1)
          continue;
        auto dstPtrTy = dyn_cast<::tla::PtrType>(recast.getType());
        if (!dstPtrTy)
          continue;
        int64_t elemB = getByteSizeOfFixedWidthScalarType(dstPtrTy.getPointee());
        if (elemB <= 0) {
          recast.emitError() << "recast_ptr result pointee is not a fixed-width scalar type for "
                                "1D memref pointer_cast lowering";
          signalPassFailure();
          return;
        }
        auto alloc = traceToAllocForRecast(recast.getSrc());
        if (!alloc) {
          recast.emitError() << "recast_ptr for pointer_cast lowering must be traceable to a "
                                "dominating tla.alloc_ptr (possibly through other recast_ptr).";
          signalPassFailure();
          return;
        }
        int64_t sizeB = alloc.getSizeBytesAttr().getInt();
        if (sizeB % elemB != 0) {
          recast.emitError() << "alloc size_bytes is not a multiple of recast pointee type size";
          signalPassFailure();
          return;
        }
        int64_t numElems = sizeB / elemB;
        MLIRContext *ctx = recast->getContext();
        FailureOr<uint64_t> off = assignOrGetAllocPtrOffsetForPass(alloc, st);
        if (failed(off) || *off > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          recast.emitError()
              << "alloc offset resolution failed in tla-alloc-ptr-to-hivm-pointer-cast";
          signalPassFailure();
          return;
        }
        auto loc = recast.getLoc();
        FailureOr<MemRefType> mref =
            hivmMemref1D(ctx, numElems, dstPtrTy.getPointee(), dstPtrTy.getAddrspace());
        if (failed(mref)) {
          recast.emitError() << "failed to build HIVM memref type for recast_ptr lowering";
          signalPassFailure();
          return;
        }
        OpBuilder re(recast);
        auto c0 = re.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(*off), 64);
        auto pcast =
            re.create<hivm::PointerCastOp>(loc, *mref, ValueRange{c0}, ValueRange{}).getResult();
        auto bridge = re.create<::tla::HivmMemrefAsPtrOp>(loc, recast.getType(), pcast);
        recast.getResult().replaceAllUsesWith(bridge.getResult());
        recast.erase();
      }

      {
        SmallVector<::tla::AllocPtrOp, 8> allocs;
        func.walk([&](::tla::AllocPtrOp a) { allocs.push_back(a); });
        for (::tla::AllocPtrOp alloc : allocs) {
          if (!alloc->getBlock())
            continue;
          if (alloc->use_empty()) {
            alloc.erase();
            continue;
          }
          auto ptrTy = dyn_cast<::tla::PtrType>(alloc.getType());
          if (!ptrTy) {
            alloc->emitError() << "alloc_ptr not lowered: unexpected result type";
            signalPassFailure();
            return;
          }
          int64_t elemB = getByteSizeOfFixedWidthScalarType(ptrTy.getPointee());
          if (elemB <= 0) {
            alloc->emitError() << "alloc_ptr result pointee is not a fixed-width scalar type for "
                                  "1D memref pointer_cast lowering";
            signalPassFailure();
            return;
          }
          int64_t sizeB = alloc.getSizeBytesAttr().getInt();
          if (sizeB % elemB != 0) {
            alloc->emitError() << "alloc size_bytes is not a multiple of result pointee type size";
            signalPassFailure();
            return;
          }
          int64_t numElems = sizeB / elemB;
          MLIRContext *ctx = alloc->getContext();
          FailureOr<uint64_t> off = assignOrGetAllocPtrOffsetForPass(alloc, st);
          if (failed(off) || *off > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            alloc->emitError() << "alloc offset failed";
            signalPassFailure();
            return;
          }
          auto loc = alloc.getLoc();
          FailureOr<MemRefType> mref =
              hivmMemref1D(ctx, numElems, ptrTy.getPointee(), ptrTy.getAddrspace());
          if (failed(mref)) {
            alloc->emitError() << "failed HIVM memref for alloc_ptr lowering";
            signalPassFailure();
            return;
          }
          OpBuilder b(alloc);
          auto c0 = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(*off), 64);
          auto pcast =
              b.create<hivm::PointerCastOp>(loc, *mref, ValueRange{c0}, ValueRange{}).getResult();
          auto bridge = b.create<::tla::HivmMemrefAsPtrOp>(loc, alloc.getResult().getType(), pcast);
          alloc.getResult().replaceAllUsesWith(bridge.getResult());
          alloc.erase();
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTlaAllocPtrToHivmPointerCastPass() {
  return std::make_unique<TlaAllocPtrToHivmPointerCastPass>();
}

void registerTlaAllocPtrToHivmPointerCastPass() {
  PassRegistration<TlaAllocPtrToHivmPointerCastPass>();
}

} // namespace tla
