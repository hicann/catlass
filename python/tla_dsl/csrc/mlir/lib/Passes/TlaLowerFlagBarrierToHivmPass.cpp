#include "PassesCommon.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"

namespace tla {
namespace {

/// HIVM hardware exposes 8 event ids (0–7) per pipe pair for flag sync lowering.
static constexpr int64_t kMaxHivmPipePairEventIndex = 7;

class LocalMemBarOpRewrite : public OpRewritePattern<::tla::LocalMemBarOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::tla::LocalMemBarOp op,
                                PatternRewriter &rewriter) const override {
    auto barrierKind = op.getBarrierKind();
    if (barrierKind < 0 || barrierKind > 11) {
      return op.emitError() << "barrier_kind " << barrierKind
                            << " is out of range [0, 11]";
    }
    Value encodedVal = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), barrierKind, 32);
    rewriter.create<hivmave::VFMemBarOp>(op.getLoc(), encodedVal);
    rewriter.eraseOp(op);
    return success();
  }
};

class TlaLowerFlagBarrierToHivmPass : public PassWrapper<TlaLowerFlagBarrierToHivmPass, OperationPass<ModuleOp>> {
private:
  struct PipePair {
    int32_t src;
    int32_t dst;

    bool operator==(const PipePair &other) const { return src == other.src && dst == other.dst; }
  };

  struct PipePairInfo {
    static inline PipePair getEmptyKey() {
      return {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min()};
    }
    static inline PipePair getTombstoneKey() {
      return {std::numeric_limits<int32_t>::min() + 1, std::numeric_limits<int32_t>::min() + 1};
    }
    static unsigned getHashValue(const PipePair &key) {
      return llvm::hash_combine(key.src, key.dst);
    }
    static bool isEqual(const PipePair &lhs, const PipePair &rhs) { return lhs == rhs; }
  };

  struct FlagInfo {
    PipeAttr srcPipe;
    PipeAttr dstPipe;
    int64_t eventId = -1;
    bool hasSet = false;
    bool hasWait = false;
    Operation *firstWaitOp = nullptr;
  };

  struct CrossFlagInfo {
    int64_t id = -1;
    PipeAttr srcPipe;
    PipeAttr dstPipe;
    int64_t mode = 2;
    bool hasSet = false;
    bool hasWait = false;
    Operation *firstWaitOp = nullptr;
  };

  static FailureOr<hivm::PIPE> getHivmPipe(Pipe pipe) {
    switch (pipe) {
    case Pipe::scalar:
      return hivm::PIPE::PIPE_S;
    case Pipe::vector:
      return hivm::PIPE::PIPE_V;
    case Pipe::cube:
      return hivm::PIPE::PIPE_M;
    case Pipe::mte1:
      return hivm::PIPE::PIPE_MTE1;
    case Pipe::mte2:
      return hivm::PIPE::PIPE_MTE2;
    case Pipe::mte3:
      return hivm::PIPE::PIPE_MTE3;
    case Pipe::all:
      return hivm::PIPE::PIPE_ALL;
    case Pipe::fix:
      return hivm::PIPE::PIPE_FIX;
    default:
      return failure();
    }
  }

  static FailureOr<hivm::PipeAttr> getHivmPipeAttr(MLIRContext *ctx, PipeAttr attr) {
    FailureOr<hivm::PIPE> pipe = getHivmPipe(attr.getPipe());
    if (failed(pipe))
      return failure();
    return hivm::PipeAttr::get(ctx, *pipe);
  }

  static FailureOr<hivm::TCoreType> inferCrossCoreFlagCore(Operation *op) {
    // Derive the core from the enclosing function's hivm.func_core_type attribute
    // (set by tla-infer-func-core-type and carried onto the AIC/AIV fragments by
    // tla-split-mixed-func). This pass runs after tla-vector-region, so the
    // frontend tla.cube/tla.vector regions no longer exist to inspect.
    if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
      auto coreType =
          funcOp->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
      if (coreType && coreType.getFuncCoreType() == hivm::TFuncCoreType::AIC)
        return hivm::TCoreType::CUBE;
      if (coreType && coreType.getFuncCoreType() == hivm::TFuncCoreType::AIV)
        return hivm::TCoreType::VECTOR;
    }
    op->emitError() << "expected " << op->getName().getStringRef()
                    << " to be in a function with an AIC/AIV hivm.func_core_type "
                       "attribute (set by tla-infer-func-core-type)";
    return failure();
  }

  static FailureOr<StringRef> resolveFlagName(Operation *op, Value flagOperand) {
    if (auto flagAttr = op->getAttrOfType<StringAttr>("flag"))
      return flagAttr.getValue();
    if (flagOperand) {
      if (auto flagDef = flagOperand.getDefiningOp<::tla::FlagOp>()) {
        if (auto nameAttr = flagDef->getAttrOfType<StringAttr>("name"))
          return nameAttr.getValue();
      }
    }
    op->emitError() << "expected " << op->getName().getStringRef()
                    << " to have a 'flag' attribute or a tla.flag operand";
    return failure();
  }

  LogicalResult ensureEventId(Operation *op, FlagInfo &info,
                              DenseMap<PipePair, int64_t, PipePairInfo> &nextEventIdByPipe) const {
    if (info.eventId >= 0)
      return success();
    PipePair pair{static_cast<int32_t>(info.srcPipe.getPipe()),
                  static_cast<int32_t>(info.dstPipe.getPipe())};
    int64_t eventId = 0;
    auto pairIt = nextEventIdByPipe.find(pair);
    if (pairIt == nextEventIdByPipe.end()) {
      nextEventIdByPipe[pair] = 1;
    } else {
      eventId = pairIt->second;
      if (eventId > kMaxHivmPipePairEventIndex) {
        op->emitError() << "event id exhausted (max " << kMaxHivmPipePairEventIndex << ")";
        return failure();
      }
      pairIt->second = eventId + 1;
    }
    info.eventId = eventId;
    return success();
  }

  LogicalResult lowerSetOrWait(Operation *op, bool isSet, llvm::StringMap<FlagInfo> &flagInfoByName,
                               DenseMap<PipePair, int64_t, PipePairInfo> &nextEventIdByPipe,
                               SmallVectorImpl<Operation *> &toErase) const {
    if (op->getNumOperands() > 1) {
      op->emitError() << "expected " << op->getName().getStringRef()
                      << " to have at most 1 operand";
      return failure();
    }

    FailureOr<StringRef> flagName =
        resolveFlagName(op, op->getNumOperands() == 1 ? op->getOperand(0) : Value());
    if (failed(flagName))
      return failure();

    auto flagIt = flagInfoByName.find(*flagName);
    if (flagIt == flagInfoByName.end()) {
      op->emitError() << "unknown flag name '" << *flagName << "'";
      return failure();
    }

    FlagInfo &info = flagIt->second;
    if (isSet) {
      info.hasSet = true;
    } else {
      info.hasWait = true;
      if (!info.firstWaitOp)
        info.firstWaitOp = op;
    }
    if (failed(ensureEventId(op, info, nextEventIdByPipe)))
      return failure();

    PatternRewriter rewriter(op->getContext());
    rewriter.setInsertionPoint(op);
    MLIRContext *ctx = op->getContext();
    FailureOr<hivm::PipeAttr> setPipeAttr = getHivmPipeAttr(ctx, info.srcPipe);
    FailureOr<hivm::PipeAttr> waitPipeAttr = getHivmPipeAttr(ctx, info.dstPipe);
    if (failed(setPipeAttr) || failed(waitPipeAttr)) {
      op->emitError() << "unsupported pipe for HIVM sync lowering";
      return failure();
    }

    auto eventAttr = hivm::EventAttr::get(ctx, static_cast<hivm::EVENT>(info.eventId));
    if (isSet) {
      rewriter.create<hivm::SetFlagOp>(op->getLoc(), *setPipeAttr, *waitPipeAttr, eventAttr,
                                       Value{});
    } else {
      rewriter.create<hivm::WaitFlagOp>(op->getLoc(), *setPipeAttr, *waitPipeAttr, eventAttr,
                                        Value{});
    }
    toErase.push_back(op);
    return success();
  }

  LogicalResult lowerCrossCoreSetOrWaitFlag(Operation *op, bool isSet,
                                            llvm::StringMap<CrossFlagInfo> &crossFlagInfoByName,
                                            SmallVectorImpl<Operation *> &toErase) const {
    if (op->getNumOperands() != 1) {
      op->emitError() << "expected " << op->getName().getStringRef()
                      << " to have exactly 1 operand";
      return failure();
    }

    auto flagDef = op->getOperand(0).getDefiningOp<::tla::CrossFlagOp>();
    if (!flagDef || !flagDef->getAttrOfType<StringAttr>("name")) {
      op->emitError() << "expected " << op->getName().getStringRef()
                      << " to have a tla.cross_flag operand";
      return failure();
    }
    StringRef flagName = flagDef->getAttrOfType<StringAttr>("name").getValue();
    auto flagIt = crossFlagInfoByName.find(flagName);
    if (flagIt == crossFlagInfoByName.end() || flagIt->second.id < 0) {
      op->emitError() << "unknown cross flag name '" << flagName << "'";
      return failure();
    }

    CrossFlagInfo &info = flagIt->second;
    if (isSet) {
      info.hasSet = true;
    } else {
      info.hasWait = true;
      if (!info.firstWaitOp)
        info.firstWaitOp = op;
    }

    if (info.mode != 2) {
      op->emitError() << "unsupported cross_flag mode " << info.mode
                      << "; only mode 2 is currently supported";
      return failure();
    }

    auto core = inferCrossCoreFlagCore(op);
    auto setPipeAttr = getHivmPipeAttr(op->getContext(), info.srcPipe);
    auto waitPipeAttr = getHivmPipeAttr(op->getContext(), info.dstPipe);
    if (failed(core) || failed(setPipeAttr) || failed(waitPipeAttr)) {
      op->emitError() << "unsupported core or pipe for HIVM cross-core flag lowering";
      return failure();
    }

    PatternRewriter rewriter(op->getContext());
    rewriter.setInsertionPoint(op);
    auto coreTypeAttr = hivm::TCoreTypeAttr::get(op->getContext(), *core);
    OpFoldResult flag = IntegerAttr::get(IntegerType::get(op->getContext(), 64), info.id);
    if (isSet) {
      rewriter.create<hivm::SyncBlockSetOp>(op->getLoc(), coreTypeAttr, *setPipeAttr,
                                            *waitPipeAttr, flag);
    } else {
      rewriter.create<hivm::SyncBlockWaitOp>(op->getLoc(), coreTypeAttr, *setPipeAttr,
                                             *waitPipeAttr, flag);
    }
    toErase.push_back(op);
    return success();
  }

  LogicalResult lowerPipeBarrier(::tla::PipeBarrierOp op) const {
    auto pipeAttr = op->getAttrOfType<PipeAttr>("pipe");
    if (!pipeAttr) {
      op.emitError() << "expected tla.pipe_barrier to have a 'pipe' attribute";
      return failure();
    }

    PatternRewriter rewriter(op->getContext());
    rewriter.setInsertionPoint(op);
    FailureOr<hivm::PipeAttr> hivmPipeAttr = getHivmPipeAttr(op.getContext(), pipeAttr);
    if (failed(hivmPipeAttr)) {
      op.emitError() << "unsupported pipe for HIVM pipe_barrier lowering";
      return failure();
    }

    rewriter.create<hivm::PipeBarrierOp>(op.getLoc(), *hivmPipeAttr);
    rewriter.eraseOp(op);
    return success();
  }

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerFlagBarrierToHivmPass)

  StringRef getArgument() const override { return "tla-lower-flag-barrier-to-hivm"; }
  StringRef getName() const override { return "TlaLowerFlagBarrierToHivmPass"; }
  StringRef getDescription() const override {
    return "Lower Tla pipe synchronization flags to HIVM synchronization ops.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::tla::TlaDialect, hivm::HIVMDialect, arith::ArithDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<::tla::FlagOp, 8> tlaFlagOps;
    SmallVector<::tla::CrossFlagOp, 8> tlaCrossFlagOps;
    SmallVector<Operation *, 8> flagUseOps;
    SmallVector<Operation *, 8> crossCoreUseOps;
    SmallVector<::tla::PipeBarrierOp, 8> pipeBarrierOps;
    DenseMap<PipePair, int64_t, PipePairInfo> nextEventIdByPipe;
    llvm::StringMap<FlagInfo> flagInfoByName;
    llvm::StringMap<CrossFlagInfo> crossFlagInfoByName;
    int64_t nextCrossFlagId = 0;
    SmallVector<Operation *, 8> toErase;

    module.walk([&](::tla::FlagOp op) { tlaFlagOps.push_back(op); });
    for (::tla::FlagOp flagOp : tlaFlagOps) {
      if (flagOp->getNumResults() != 1) {
        flagOp->emitError() << "expected tla.flag to have exactly 1 result";
        signalPassFailure();
        return;
      }
      auto nameAttr = flagOp->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        flagOp->emitError() << "expected tla.flag to have a 'name' attribute";
        signalPassFailure();
        return;
      }
      auto srcPipeAttr = flagOp->getAttrOfType<PipeAttr>("src_pipe");
      auto dstPipeAttr = flagOp->getAttrOfType<PipeAttr>("dst_pipe");
      if (!srcPipeAttr || !dstPipeAttr) {
        flagOp->emitError() << "expected tla.flag to have src_pipe and dst_pipe";
        signalPassFailure();
        return;
      }
      auto insert =
          flagInfoByName.try_emplace(nameAttr.getValue(), FlagInfo{srcPipeAttr, dstPipeAttr, -1});
      if (!insert.second) {
        flagOp->emitError() << "duplicate tla.flag name '" << nameAttr.getValue() << "'";
        signalPassFailure();
        return;
      }
      for (auto &use : flagOp->getResult(0).getUses()) {
        Operation *userOp = use.getOwner();
        if (llvm::isa<::tla::SetFlagOp, ::tla::WaitFlagOp>(userOp))
          continue;
        flagOp->emitError() << "tla.flag result is used by unsupported op '"
                            << userOp->getName().getStringRef() << "'";
        signalPassFailure();
        return;
      }
    }

    module.walk([&](Operation *op) {
      if (llvm::isa<::tla::SetFlagOp, ::tla::WaitFlagOp>(op))
        flagUseOps.push_back(op);
    });
    for (Operation *op : flagUseOps) {
      if (!op->getBlock())
        continue;
      bool isSet = llvm::isa<::tla::SetFlagOp>(op);
      if (failed(lowerSetOrWait(op, isSet, flagInfoByName, nextEventIdByPipe, toErase))) {
        signalPassFailure();
        return;
      }
    }

    for (auto &entry : flagInfoByName) {
      FlagInfo &info = entry.getValue();
      if (!info.hasWait || info.hasSet)
        continue;
      if (info.firstWaitOp) {
        info.firstWaitOp->emitError()
            << "wait_flag used without set_flag for '" << entry.getKey() << "'";
      } else {
        module.emitError() << "wait_flag used without set_flag for '" << entry.getKey() << "'";
      }
      signalPassFailure();
      return;
    }

    module.walk([&](::tla::CrossFlagOp op) { tlaCrossFlagOps.push_back(op); });
    for (::tla::CrossFlagOp flagOp : tlaCrossFlagOps) {
      if (flagOp->getNumResults() != 1) {
        flagOp->emitError() << "expected tla.cross_flag to have exactly 1 result";
        signalPassFailure();
        return;
      }
      auto nameAttr = flagOp->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        flagOp->emitError() << "expected tla.cross_flag to have a 'name' attribute";
        signalPassFailure();
        return;
      }
      auto srcPipeAttr = flagOp->getAttrOfType<PipeAttr>("src_pipe");
      auto dstPipeAttr = flagOp->getAttrOfType<PipeAttr>("dst_pipe");
      auto modeAttr = flagOp->getAttrOfType<IntegerAttr>("mode");
      int64_t mode = modeAttr ? modeAttr.getInt() : 2;
      if (!srcPipeAttr || !dstPipeAttr) {
        flagOp->emitError() << "expected tla.cross_flag to have src_pipe and dst_pipe";
        signalPassFailure();
        return;
      }
      auto insert = crossFlagInfoByName.try_emplace(nameAttr.getValue(), CrossFlagInfo{});
      if (!insert.second) {
        CrossFlagInfo &existing = insert.first->second;
        if ((existing.srcPipe && existing.srcPipe != srcPipeAttr) ||
            (existing.dstPipe && existing.dstPipe != dstPipeAttr) || existing.mode != mode) {
          flagOp->emitError() << "conflicting tla.cross_flag definition for name '"
                              << nameAttr.getValue() << "'";
          signalPassFailure();
          return;
        }
      }
      CrossFlagInfo &info = insert.first->second;
      if (info.id < 0) {
        if (nextCrossFlagId > 10) {
          flagOp->emitError() << "cross flag id exhausted (max id 10, 11 flags)";
          signalPassFailure();
          return;
        }
        info.id = nextCrossFlagId++;
      }
      if (!info.srcPipe)
        info.srcPipe = srcPipeAttr;
      if (!info.dstPipe)
        info.dstPipe = dstPipeAttr;
      info.mode = mode;
      for (auto &use : flagOp->getResult(0).getUses()) {
        Operation *userOp = use.getOwner();
        if (llvm::isa<::tla::CrossCoreSetFlagOp, ::tla::CrossCoreWaitFlagOp>(userOp))
          continue;
        flagOp->emitError() << "tla.cross_flag result is used by unsupported op '"
                            << userOp->getName().getStringRef() << "'";
        signalPassFailure();
        return;
      }
    }

    module.walk([&](Operation *op) {
      if (llvm::isa<::tla::CrossCoreSetFlagOp, ::tla::CrossCoreWaitFlagOp>(op))
        crossCoreUseOps.push_back(op);
    });
    for (Operation *op : crossCoreUseOps) {
      if (!op->getBlock())
        continue;
      bool isSet = llvm::isa<::tla::CrossCoreSetFlagOp>(op);
      if (failed(lowerCrossCoreSetOrWaitFlag(op, isSet, crossFlagInfoByName, toErase))) {
        signalPassFailure();
        return;
      }
    }

    for (auto &entry : crossFlagInfoByName) {
      CrossFlagInfo &info = entry.getValue();
      if (!info.hasWait || info.hasSet)
        continue;
      if (info.firstWaitOp) {
        info.firstWaitOp->emitError()
            << "cross_core_wait_flag used without cross_core_set_flag for '" << entry.getKey() << "'";
      } else {
        module.emitError() << "cross_core_wait_flag used without cross_core_set_flag for '"
                           << entry.getKey() << "'";
      }
      signalPassFailure();
      return;
    }

    for (::tla::FlagOp flagOp : tlaFlagOps) {
      Operation *flagRaw = flagOp.getOperation();
      if (!flagRaw->getBlock() || flagRaw->getNumResults() != 1)
        continue;
      if (flagRaw->getResult(0).use_empty()) {
        toErase.push_back(flagRaw);
        continue;
      }
      bool allUsersErased = true;
      for (auto &use : flagRaw->getResult(0).getUses()) {
        if (!llvm::is_contained(toErase, use.getOwner())) {
          allUsersErased = false;
          break;
        }
      }
      if (allUsersErased)
        toErase.push_back(flagRaw);
    }

    for (::tla::CrossFlagOp flagOp : tlaCrossFlagOps) {
      Operation *flagRaw = flagOp.getOperation();
      if (flagRaw && flagRaw->getBlock())
        toErase.push_back(flagRaw);
    }

    for (Operation *op : toErase) {
      if (op && op->getBlock())
        op->erase();
    }

    module.walk([&](::tla::PipeBarrierOp op) { pipeBarrierOps.push_back(op); });
    for (::tla::PipeBarrierOp op : pipeBarrierOps) {
      if (!op->getBlock())
        continue;
      if (failed(lowerPipeBarrier(op))) {
        signalPassFailure();
        return;
      }
    }

    RewritePatternSet localBarrierPatterns(&getContext());
    localBarrierPatterns.add<LocalMemBarOpRewrite>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(localBarrierPatterns)))) {
      signalPassFailure();
      return;
    }

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<::tla::FlagOp, ::tla::SetFlagOp, ::tla::WaitFlagOp,
                        ::tla::CrossFlagOp, ::tla::CrossCoreSetFlagOp,
                        ::tla::CrossCoreWaitFlagOp, ::tla::PipeBarrierOp>();

    RewritePatternSet patterns(&getContext());
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerFlagBarrierToHivmPass() { return std::make_unique<TlaLowerFlagBarrierToHivmPass>(); }

void registerTlaLowerFlagBarrierToHivmPass() { PassRegistration<TlaLowerFlagBarrierToHivmPass>(); }

} // namespace tla
