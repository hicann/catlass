#include "PassesCommon.h"
#include "PassesInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace tla {
namespace {

// Lower tla.scalar_load/store on bridged memref sources/dests before
// vector/cube region outlining.
//
// Placement: after tla-split-mixed-func, before tla-vector-region /
// tla-cube-region. Phase-1 ScalarSSA only covers GM kernel args already
// bridged by tla-lower-func (StridedLayout carries strides); leftover
// !tla.tensor sources are rejected here.
class TlaLowerScalarAccessPass
    : public PassWrapper<TlaLowerScalarAccessPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerScalarAccessPass)

  StringRef getArgument() const override { return "tla-lower-scalar-access"; }
  StringRef getName() const override { return "TlaLowerScalarAccessPass"; }
  StringRef getDescription() const override {
    return "Lower tla.scalar_load/store on bridged memrefs to memref.load/store "
           "before vector and cube region passes.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect,
                    ::tla::TlaDialect>();
  }

  void runOnOperation() override {
    if (failed(lowerScalarLoadOnMemrefs(getOperation())) ||
        failed(lowerScalarStoreOnMemrefs(getOperation())))
      signalPassFailure();
  }

private:
  static LogicalResult lowerScalarLoadOnMemrefs(ModuleOp module) {
    SmallVector<::tla::ScalarLoadOp, 16> ops;
    module.walk([&](::tla::ScalarLoadOp op) { ops.push_back(op); });
    for (::tla::ScalarLoadOp op : ops) {
      Value source = op.getSource();
      auto memrefTy = dyn_cast<MemRefType>(source.getType());
      if (!memrefTy) {
        return op.emitError(
                   "tla.scalar_load expects a bridged memref source after "
                   "tla-lower-func (Phase-1 GM kernel-arg indexing only)"),
               failure();
      }

      OpBuilder builder(op);
      SmallVector<Value, 2> indices(op.getIndices().begin(), op.getIndices().end());

      Value loaded;
      if (memrefTy.getRank() == 1) {
        if (indices.size() != 1)
          return op.emitError("rank-1 memref scalar_load expects exactly 1 index"),
                 failure();
        loaded = builder.create<mlir::memref::LoadOp>(op.getLoc(), source, indices[0]);
      } else if (memrefTy.getRank() == 2) {
        if (indices.size() != 2)
          return op.emitError("rank-2 memref scalar_load expects exactly 2 indices"),
                 failure();
        loaded = builder.create<mlir::memref::LoadOp>(op.getLoc(), source, indices);
      } else {
        return op.emitError("tla.scalar_load memref source must be rank-1 or rank-2"),
               failure();
      }
      op.getResult().replaceAllUsesWith(loaded);
      op.erase();
    }
    return success();
  }

  static LogicalResult lowerScalarStoreOnMemrefs(ModuleOp module) {
    SmallVector<::tla::ScalarStoreOp, 16> ops;
    module.walk([&](::tla::ScalarStoreOp op) { ops.push_back(op); });
    for (::tla::ScalarStoreOp op : ops) {
      Value dest = op.getDest();
      auto memrefTy = dyn_cast<MemRefType>(dest.getType());
      if (!memrefTy) {
        return op.emitError(
                   "tla.scalar_store expects a bridged memref dest after "
                   "tla-lower-func (Phase-1 GM kernel-arg indexing only)"),
               failure();
      }

      OpBuilder builder(op);
      SmallVector<Value, 2> indices(op.getIndices().begin(), op.getIndices().end());

      Value value = op.getValue();
      if (memrefTy.getRank() == 1) {
        if (indices.size() != 1)
          return op.emitError("rank-1 memref scalar_store expects exactly 1 index"),
                 failure();
        builder.create<mlir::memref::StoreOp>(op.getLoc(), value, dest, indices[0]);
      } else if (memrefTy.getRank() == 2) {
        if (indices.size() != 2)
          return op.emitError("rank-2 memref scalar_store expects exactly 2 indices"),
                 failure();
        builder.create<mlir::memref::StoreOp>(op.getLoc(), value, dest, indices);
      } else {
        return op.emitError("tla.scalar_store memref dest must be rank-1 or rank-2"),
               failure();
      }
      op.erase();
    }
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerScalarAccessPass() {
  return std::make_unique<TlaLowerScalarAccessPass>();
}

void registerTlaLowerScalarAccessPass() { PassRegistration<TlaLowerScalarAccessPass>(); }

} // namespace tla
