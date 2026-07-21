#include "PassesCommon.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"

namespace tla {
namespace {

/// HIVM hardware exposes 8 event ids (0–7) per pipe pair for flag sync lowering.
static constexpr int64_t kMaxHivmPipePairEventIndex = 7;
static constexpr int64_t kMaxCrossFlagId = 15;

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
    int64_t mode = -1;
    Operation *firstDecl = nullptr;
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
    // (set by tla-lower-func and carried onto the AIC/AIV fragments by
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
                       "attribute (set by tla-lower-func or tla-split-mixed-func)";
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

  class CrossFlagOpConversion : public OpConversionPattern<::tla::CrossFlagOp> {
  public:
    CrossFlagOpConversion(TypeConverter &converter, MLIRContext *ctx,
                          const llvm::StringMap<CrossFlagInfo> &flags)
        : OpConversionPattern(converter, ctx), flags(flags) {}

    LogicalResult matchAndRewrite(::tla::CrossFlagOp op, OpAdaptor,
                                  ConversionPatternRewriter &rewriter) const override {
      auto name = op->getAttrOfType<StringAttr>("name");
      auto it = name ? flags.find(name.getValue()) : flags.end();
      if (it == flags.end() || it->second.id < 0)
        return rewriter.notifyMatchFailure(op, "missing assigned cross flag id");
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, it->second.id, 64);
      return success();
    }

  private:
    const llvm::StringMap<CrossFlagInfo> &flags;
  };

  class CrossUseOpConversion : public ConversionPattern {
  public:
    CrossUseOpConversion(StringRef opName, TypeConverter &converter, MLIRContext *ctx, bool isSet)
        : ConversionPattern(converter, opName, 1, ctx), isSet(isSet) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
      if (operands.size() != 1)
        return rewriter.notifyMatchFailure(op, "expected one cross flag operand");
      auto flagType = dyn_cast<::tla::CrossFlagType>(op->getOperand(0).getType());
      if (!flagType)
        return rewriter.notifyMatchFailure(op, "expected a mode-parameterized cross flag type");
      int64_t mode = flagType.getMode();
      auto core = inferCrossCoreFlagCore(op);
      auto pipeAttr = op->getAttrOfType<PipeAttr>("pipe");
      FailureOr<hivm::PIPE> pipe =
          pipeAttr ? getHivmPipe(pipeAttr.getPipe()) : FailureOr<hivm::PIPE>(failure());
      if (failed(core) || failed(pipe))
        return rewriter.notifyMatchFailure(op, "unsupported core or pipe");
      uint64_t pipeValue = static_cast<uint64_t>(*pipe);
      Value flagId = operands.front();
      IntegerAttr staticIdAttr;
      if (auto constant = flagId.getDefiningOp<arith::ConstantIntOp>())
        staticIdAttr = rewriter.getI64IntegerAttr(constant.value());

      auto emitIntraBlock = [&](OpBuilder &builder, Location loc, Value id,
                                IntegerAttr immediateId) {
        if (immediateId) {
          if (isSet)
            builder.create<hivm::SetIntraBlockImmInstrOp>(loc, pipeValue, immediateId.getInt());
          else
            builder.create<hivm::WaitIntraBlockImmInstrOp>(loc, pipeValue, immediateId.getInt());
        } else if (isSet) {
          builder.create<hivm::SetIntraBlockRegInstrOp>(loc, pipeValue, id);
        } else {
          builder.create<hivm::WaitIntraBlockRegInstrOp>(loc, pipeValue, id);
        }
      };

      if (mode == 0 || mode == 1) {
        if (isSet) {
          Value config;
          if (staticIdAttr) {
            int64_t encoded = 1 | (mode << 4) | (staticIdAttr.getInt() << 8);
            config = rewriter.create<arith::ConstantIntOp>(op->getLoc(), encoded, 64);
          } else {
            Value shift = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 8, 64);
            Value shifted = rewriter.create<arith::ShLIOp>(op->getLoc(), flagId, shift);
            Value base = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 1 | (mode << 4), 64);
            config = rewriter.create<arith::OrIOp>(op->getLoc(), shifted, base);
          }
          rewriter.create<hivm::SetCrossCoreInstrOp>(op->getLoc(), pipeValue, config);
        } else if (staticIdAttr) {
          rewriter.create<hivm::WaitFlagDevPipeImmInstrOp>(op->getLoc(), pipeValue,
                                                           staticIdAttr.getInt());
        } else {
          rewriter.create<hivm::WaitFlagDevPipeRegInstrOp>(op->getLoc(), pipeValue, flagId);
        }
        rewriter.eraseOp(op);
        return success();
      }

      if (mode != 2 && mode != 4)
        return op->emitError() << "unsupported cross_flag mode " << mode;

      auto offsetId = [&](int64_t offset) -> std::pair<Value, IntegerAttr> {
        if (staticIdAttr)
          return {Value(), rewriter.getI64IntegerAttr(staticIdAttr.getInt() + offset)};
        if (offset == 0)
          return {flagId, IntegerAttr()};
        Value addend = rewriter.create<arith::ConstantIntOp>(op->getLoc(), offset, 64);
        return {rewriter.create<arith::AddIOp>(op->getLoc(), flagId, addend), IntegerAttr()};
      };

      if (mode == 2) {
        auto base = offsetId(0);
        emitIntraBlock(rewriter, op->getLoc(), base.first, base.second);
        if (*core == hivm::TCoreType::CUBE) {
          auto upper = offsetId(16);
          emitIntraBlock(rewriter, op->getLoc(), upper.first, upper.second);
        }
        rewriter.eraseOp(op);
        return success();
      }

      auto aivIdAttr = op->getAttrOfType<IntegerAttr>("aiv_id");
      if (!aivIdAttr || (aivIdAttr.getInt() != 0 && aivIdAttr.getInt() != 1))
        return op->emitError("mode 4 requires aiv_id to be 0 or 1");
      if (*core == hivm::TCoreType::CUBE) {
        auto selected = offsetId(16 * aivIdAttr.getInt());
        emitIntraBlock(rewriter, op->getLoc(), selected.first, selected.second);
      } else {
        Value subBlock =
            rewriter.create<hivm::GetSubBlockIdxOp>(op->getLoc(), rewriter.getI64Type());
        Value expected =
            rewriter.create<arith::ConstantIntOp>(op->getLoc(), aivIdAttr.getInt(), 64);
        Value condition = rewriter.create<arith::CmpIOp>(op->getLoc(), arith::CmpIPredicate::eq,
                                                         subBlock, expected);
        auto guard = rewriter.create<scf::IfOp>(op->getLoc(), condition, false);
        OpBuilder::InsertionGuard insertionGuard(rewriter);
        rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
        auto base = offsetId(0);
        emitIntraBlock(rewriter, op->getLoc(), base.first, base.second);
      }
      rewriter.eraseOp(op);
      return success();
    }

  private:
    bool isSet;
  };

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerFlagBarrierToHivmPass)

  StringRef getArgument() const override { return "tla-lower-flag-barrier-to-hivm"; }
  StringRef getName() const override { return "TlaLowerFlagBarrierToHivmPass"; }
  StringRef getDescription() const override {
    return "Lower Tla pipe synchronization flags to HIVM synchronization ops.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::tla::TlaDialect, hivm::HIVMDialect, arith::ArithDialect, func::FuncDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<::tla::FlagOp, 8> tlaFlagOps;
    SmallVector<::tla::CrossFlagOp, 8> tlaCrossFlagOps;
    SmallVector<Operation *, 8> flagUseOps;
    SmallVector<::tla::PipeBarrierOp, 8> pipeBarrierOps;
    DenseMap<PipePair, int64_t, PipePairInfo> nextEventIdByPipe;
    llvm::StringMap<FlagInfo> flagInfoByName;
    llvm::StringMap<CrossFlagInfo> crossFlagInfoByName;
    SmallVector<Operation *, 8> toErase;

    module.walk([&](Operation *op) {
      if (auto flagOp = dyn_cast<::tla::FlagOp>(op))
        tlaFlagOps.push_back(flagOp);
      else if (auto crossFlagOp = dyn_cast<::tla::CrossFlagOp>(op))
        tlaCrossFlagOps.push_back(crossFlagOp);
      else if (llvm::isa<::tla::SetFlagOp, ::tla::WaitFlagOp>(op))
        flagUseOps.push_back(op);
    });
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
      auto flagType = dyn_cast<::tla::CrossFlagType>(flagOp.getFlag().getType());
      if (!flagType) {
        flagOp->emitError() << "expected tla.cross_flag to produce !tla.cross_flag<mode>";
        signalPassFailure();
        return;
      }
      int64_t mode = flagType.getMode();
      if (mode != 0 && mode != 1 && mode != 2 && mode != 4) {
        flagOp->emitError() << "unsupported cross_flag mode " << mode
                            << "; supported modes are 0, 1, 2, and 4";
        signalPassFailure();
        return;
      }
      auto insert = crossFlagInfoByName.try_emplace(nameAttr.getValue(), CrossFlagInfo{});
      CrossFlagInfo &info = insert.first->second;
      if (!insert.second) {
        if (info.mode != mode) {
          flagOp->emitError() << "conflicting tla.cross_flag definition for name '"
                              << nameAttr.getValue() << "': mode " << info.mode << " vs " << mode;
          signalPassFailure();
          return;
        }
        if (info.firstDecl->getParentOfType<func::FuncOp>() ==
            flagOp->getParentOfType<func::FuncOp>()) {
          flagOp->emitError() << "duplicate tla.cross_flag name '" << nameAttr.getValue() << "'";
          signalPassFailure();
          return;
        }
      } else {
        info.firstDecl = flagOp;
        info.mode = mode;
      }
    }

    SmallVector<StringRef> crossFlagNames;
    crossFlagNames.reserve(crossFlagInfoByName.size());
    for (auto &entry : crossFlagInfoByName)
      crossFlagNames.push_back(entry.getKey());
    llvm::sort(crossFlagNames);
    if (crossFlagNames.size() > static_cast<size_t>(kMaxCrossFlagId + 1)) {
      tlaCrossFlagOps.back()->emitError()
          << "cross flag id exhausted (legal range 0-" << kMaxCrossFlagId
          << "; maximum " << (kMaxCrossFlagId + 1) << " flags)";
      signalPassFailure();
      return;
    }
    for (auto [id, name] : llvm::enumerate(crossFlagNames))
      crossFlagInfoByName.find(name)->second.id = id;

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

    TypeConverter crossFlagTypeConverter;
    crossFlagTypeConverter.addConversion([](Type type) { return type; });
    crossFlagTypeConverter.addConversion(
        [&](::tla::CrossFlagType) -> Type { return IntegerType::get(&getContext(), 64); });

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<::tla::FlagOp, ::tla::SetFlagOp, ::tla::WaitFlagOp,
                        ::tla::CrossFlagOp, ::tla::CrossCoreSetFlagOp,
                        ::tla::CrossCoreWaitFlagOp, ::tla::PipeBarrierOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<CrossFlagOpConversion>(crossFlagTypeConverter, &getContext(), crossFlagInfoByName);
    patterns.add<CrossUseOpConversion>(::tla::CrossCoreSetFlagOp::getOperationName(),
                                       crossFlagTypeConverter, &getContext(), true);
    patterns.add<CrossUseOpConversion>(::tla::CrossCoreWaitFlagOp::getOperationName(),
                                       crossFlagTypeConverter, &getContext(), false);
    scf::populateSCFStructuralTypeConversionsAndLegality(crossFlagTypeConverter, patterns, target);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerFlagBarrierToHivmPass() { return std::make_unique<TlaLowerFlagBarrierToHivmPass>(); }

void registerTlaLowerFlagBarrierToHivmPass() { PassRegistration<TlaLowerFlagBarrierToHivmPass>(); }

} // namespace tla
