#pragma once

// Pieces of the tla.vector lowering that both the SIMD and the SIMT pass need.
// Everything here is mode-agnostic: it either decides which mode a
// `tla.vec.func` is in, or it handles scaffolding that exists regardless of
// mode (the `tla.vector` wrapper, `tla.local_mem_bar`, and the per-core arch
// ops that may not appear inside a vector region at all).

#include "Dialect/Tla/IR/TlaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace tla {
namespace vecregion {

/// Lower `tla.local_mem_bar` onto its HIVM barrier. Mode-agnostic: raw TLAIR may
/// place a local barrier directly in a vector wrapper, and a barrier-only
/// `tla.vec.func` is inlined rather than outlined by either mode.
mlir::LogicalResult lowerLocalMemBar(mlir::OpBuilder& b, ::tla::LocalMemBarOp op);

/// The `mode` attribute of a `tla.vec.func` ("simd" / "simt", either case).
/// SIMD is the historical default and is assumed when the attribute is absent.
mlir::StringRef getVecFuncMode(::tla::VecFuncOp vecFuncOp);
bool isSimtVecFunc(::tla::VecFuncOp vecFuncOp);
bool isSimdVecFunc(::tla::VecFuncOp vecFuncOp);

/// Inline the `tla.vector` wrapper regions of `funcOp`.
void inlineVectorRegionWrappers(mlir::func::FuncOp funcOp);

/// Per-core identity queries must be computed outside a `tla.vec.func`; emitting
/// them inside a vector region produces an op the vector backend cannot codegen.
bool isIllegalVecFuncArchOp(mlir::Operation* op, mlir::StringRef& dslName);
mlir::LogicalResult checkNoArchOpsInVecFunc(mlir::func::FuncOp funcOp);

/// Lower `tla.copy`. Mode-agnostic by construction: the route is chosen from the
/// source and destination address spaces, layout tags and element types -- never
/// from SIMD or SIMT -- and the cube pass owns the routes this one declines.
/// It lives here, and is registered by both passes, because a copy in a SIMT
/// kernel is the same copy as in a SIMD one. `loweredMemrefByValue` is the
/// registering pass's base-memref cache, so the copy and that pass's vec.func
/// lowering materialize a given descriptor base exactly once between them.
void populateTlaCopyPatterns(
    mlir::RewritePatternSet& patterns, mlir::DenseMap<mlir::Value, mlir::Value>& loweredMemrefByValue);

/// Pattern that inlines a `tla.vector` wrapper; both passes register it so
/// whichever runs first clears the scaffolding.
class InlineVectorRegionWrapperPattern : public mlir::OpRewritePattern<::tla::VectorOp> {
public:
    explicit InlineVectorRegionWrapperPattern(mlir::MLIRContext* context)
        : mlir::OpRewritePattern<::tla::VectorOp>(context, /*benefit=*/10)
    {}

    mlir::LogicalResult matchAndRewrite(::tla::VectorOp vectorOp, mlir::PatternRewriter& rewriter) const override
    {
        if (vectorOp->getNumRegions() == 0 || vectorOp.getBody().empty()) {
            rewriter.eraseOp(vectorOp);
            return mlir::success();
        }
        mlir::Block* body = &vectorOp.getBody().front();
        rewriter.inlineBlockBefore(body, vectorOp->getBlock(), vectorOp->getIterator());
        rewriter.eraseOp(vectorOp);
        return mlir::success();
    }
};

} // namespace vecregion
} // namespace tla
