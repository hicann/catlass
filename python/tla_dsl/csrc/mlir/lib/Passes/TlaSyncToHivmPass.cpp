#include "PassesCommon.h"

namespace tla {
namespace {

/// HIVM hardware exposes 8 event ids (0–7) per pipe pair for flag sync lowering.
static constexpr int64_t kMaxHivmPipePairEventIndex = 7;

class TlaSyncToHivmPass : public PassWrapper<TlaSyncToHivmPass, OperationPass<ModuleOp>> {
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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaSyncToHivmPass)

  StringRef getArgument() const override { return "tla-sync-to-hivm"; }
  StringRef getName() const override { return "TlaSyncToHivmPass"; }
  StringRef getDescription() const override {
    return "Lower Tla pipe synchronization flags to HIVM synchronization ops.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::tla::TlaDialect, hivm::HIVMDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<::tla::FlagOp, 8> tlaFlagOps;
    SmallVector<Operation *, 8> flagUseOps;
    SmallVector<::tla::PipeBarrierOp, 8> pipeBarrierOps;
    DenseMap<PipePair, int64_t, PipePairInfo> nextEventIdByPipe;
    llvm::StringMap<FlagInfo> flagInfoByName;
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

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<::tla::FlagOp, ::tla::SetFlagOp, ::tla::WaitFlagOp, ::tla::PipeBarrierOp>();

    RewritePatternSet patterns(&getContext());
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTlaSyncToHivmPass() { return std::make_unique<TlaSyncToHivmPass>(); }

void registerTlaSyncToHivmPass() { PassRegistration<TlaSyncToHivmPass>(); }

} // namespace tla
