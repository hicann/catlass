#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaTensorToMemref.h"

// tla-finalize-memref: finalize the tla.tensor -> memref lowering. Bridges the
// function ABI (func signatures + func.call surfaces tla.tensor -> memref,
// redirecting the tensor->memref arg casts from the region passes), then DCEs the
// dead scaffolding + unrealized casts left by the region passes. Last pass in the
// memref-lowering sequence.

namespace tla {
namespace {

  static bool isTlaTensorType(Type type) { return llvm::isa<::tla::TlaTensorType>(type); }

  static bool hasTlaTensorCallSurface(func::CallOp callOp) {
    return llvm::any_of(callOp.getOperandTypes(), isTlaTensorType) ||
           llvm::any_of(callOp.getResultTypes(), isTlaTensorType);
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

  static bool hasNoResultUses(Operation *op) {
    return llvm::all_of(op->getResults(), [](Value result) { return result.use_empty(); });
  }

  static bool isTlaTensorBridgeCast(UnrealizedConversionCastOp op) {
    return llvm::any_of(op->getOperandTypes(),
                        [](Type type) { return succeeded(decodeTileTypeInfo(type)); }) ||
           llvm::any_of(op->getResultTypes(),
                        [](Type type) { return succeeded(decodeTileTypeInfo(type)); });
  }

  static bool isDeadTensorBridgeCast(UnrealizedConversionCastOp op) {
    return hasNoResultUses(op.getOperation()) && isTlaTensorBridgeCast(op);
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
        if (llvm::isa<hivm::PointerCastOp>(op))
          toErase.push_back(op);
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

class TlaFinalizeMemrefPass : public PassWrapper<TlaFinalizeMemrefPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaFinalizeMemrefPass)

  StringRef getArgument() const override { return "tla-finalize-memref"; }
  StringRef getName() const override { return "TlaFinalizeMemrefPass"; }
  StringRef getDescription() const override {
    return "Finalize the tla.tensor -> memref lowering: DCE dead scaffolding and bridge casts.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect,
                    scf::SCFDialect, hivm::HIVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Bridge the tla.tensor function ABI to memref (signatures + call surfaces).
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
    // Assert the tensor / tile / region / compute ops this pass depends on being
    // gone are indeed gone. Pointer producers and transforms must already have
    // been eliminated by tla-lower-ptr; do not DCE them here, because that would
    // hide an upstream lowering failure. tla.tensor_desc is the exception: the
    // region passes materialize tile memrefs from it but may leave the op itself
    // dead, so it is DCE'd (not asserted). tla.mutex / tla.cross_* are
    // intentionally NOT asserted: they are lowered by
    // tla-lower-mutex-to-std / tla-lower-flag-barrier-to-hivm, which run AFTER
    // this pass, so they are still present when finalize runs.
    target.addIllegalOp<::tla::TileViewOp, ::tla::ScalarLoadOp, ::tla::ScalarStoreOp, ::tla::CopyOp, ::tla::MakeTensorLikeOp,
                        ::tla::MakeTensorOp, ::tla::LoadOp, ::tla::StoreOp, ::tla::FuncOp,
                        ::tla::ReturnOp, ::tla::CubeOp, ::tla::VectorOp, ::tla::MmadOp,
                        ::tla::AllocPtrOp, ::tla::RecastPtrOp, ::tla::TensorPtrOp,
                        ::tla::PtrAddOp>();
    target.addDynamicallyLegalOp<::tla::MakeShapeOp, ::tla::MakeCoordOp, ::tla::MakeStrideOp,
                                 ::tla::MakeLayoutOp, ::tla::IntToPtrOp, ::tla::TensorDescOp>(
        [](Operation *op) { return !hasNoResultUses(op); });
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](UnrealizedConversionCastOp op) { return !isDeadTensorBridgeCast(op); });
    target.addDynamicallyLegalOp<mlir::memref::SubViewOp>(
        [](mlir::memref::SubViewOp op) { return !hasNoResultUses(op.getOperation()); });
    target.addDynamicallyLegalOp<hivm::PointerCastOp>(
        [](hivm::PointerCastOp op) { return !hasNoResultUses(op.getOperation()); });

    for (int cleanupPass = 0; cleanupPass < 2; ++cleanupPass) {
      RewritePatternSet patterns(&getContext());
      patterns
          .add<EraseDeadTensorBridgeCastPattern, EraseDeadOpPattern<::tla::MakeShapeOp>,
               EraseDeadOpPattern<::tla::MakeCoordOp>, EraseDeadOpPattern<::tla::MakeStrideOp>,
               EraseDeadOpPattern<::tla::MakeLayoutOp>, EraseDeadOpPattern<::tla::IntToPtrOp>,
               EraseDeadOpPattern<mlir::memref::SubViewOp>, EraseDeadOpPattern<::tla::TensorDescOp>,
               EraseDeadOpPattern<hivm::PointerCastOp>>(
              &getContext());
      if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
    eraseDeadMaterializations(module);
    module.walk([](Operation *op) {
      op->removeAttr(kAllocSizeBytesMetadataAttrName);
    });
  }
};

} // namespace

std::unique_ptr<Pass> createTlaFinalizeMemrefPass() {
  return std::make_unique<TlaFinalizeMemrefPass>();
}

void registerTlaFinalizeMemrefPass() { PassRegistration<TlaFinalizeMemrefPass>(); }

} // namespace tla
