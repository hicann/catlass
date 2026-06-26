#include "PassesCommon.h"
#include "PassesInternal.h"
#include "mlir/IR/IRMapping.h"

namespace tla {
namespace {

using ConstantMaterializer =
    std::function<Value(Operation *, int64_t, unsigned)>;

static func::FuncOp getOrCreateRuntimeCall(ModuleOp module, StringRef name,
                                           ArrayRef<Type> operandTypes,
                                           ArrayRef<Type> resultTypes = {}) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;

  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = builder.getFunctionType(operandTypes, resultTypes);
  auto func = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
  func.setPrivate();
  return func;
}

static FailureOr<std::string> getMutexPipeSuffix(PipeAttr pipeAttr) {
  switch (pipeAttr.getPipe()) {
  case Pipe::vector:
    return std::string("v");
  case Pipe::cube:
    return std::string("m");
  case Pipe::mte1:
    return std::string("mte1");
  case Pipe::mte2:
    return std::string("mte2");
  case Pipe::mte3:
    return std::string("mte3");
  case Pipe::fix:
    return std::string("fix");
  default:
    return failure();
  }
}

static bool isTlaMutexType(Type type) { return isa<::tla::MutexType>(type); }

static bool isTransparentMutexForward(Value value, Value source,
                                      DenseSet<Value> &visited) {
  if (value == source)
    return true;
  if (!value || !source || !isTlaMutexType(value.getType()) ||
      !isTlaMutexType(source.getType()))
    return false;
  if (!visited.insert(value).second)
    return false;

  if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return false;
    unsigned resultIndex = result.getResultNumber();
    scf::YieldOp thenYield = ifOp.thenYield();
    scf::YieldOp elseYield = ifOp.elseYield();
    if (!thenYield || !elseYield || resultIndex >= thenYield.getNumOperands() ||
        resultIndex >= elseYield.getNumOperands())
      return false;
    DenseSet<Value> thenVisited = visited;
    DenseSet<Value> elseVisited = visited;
    return isTransparentMutexForward(thenYield.getOperand(resultIndex), source,
                                     thenVisited) &&
           isTransparentMutexForward(elseYield.getOperand(resultIndex), source,
                                     elseVisited);
  }

  if (auto forOp = value.getDefiningOp<scf::ForOp>()) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return false;
    unsigned resultIndex = result.getResultNumber();
    if (resultIndex >= forOp.getInitArgs().size())
      return false;
    auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yieldOp || resultIndex >= yieldOp.getNumOperands())
      return false;
    DenseSet<Value> initVisited = visited;
    DenseSet<Value> yieldVisited = visited;
    return isTransparentMutexForward(forOp.getInitArgs()[resultIndex], source,
                                     initVisited) &&
           isTransparentMutexForward(yieldOp.getOperand(resultIndex),
                                     forOp.getRegionIterArg(resultIndex),
                                     yieldVisited);
  }

  return false;
}

static bool isTransparentMutexForward(Value value, Value source) {
  DenseSet<Value> visited;
  return isTransparentMutexForward(value, source, visited);
}

static bool sameMutexRoot(::tla::MutexOp lhs, ::tla::MutexOp rhs) {
  return lhs && rhs && lhs.getOperation() == rhs.getOperation();
}

static void emitMutexResolveError(Operation *diagnosticOp,
                                  bool *emittedDiagnostic,
                                  const Twine &message) {
  if (!diagnosticOp)
    return;
  diagnosticOp->emitError() << message;
  if (emittedDiagnostic)
    *emittedDiagnostic = true;
}

static FailureOr<::tla::MutexOp> resolveMutexRootImpl(Value value,
                                                      DenseSet<Value> &visited,
                                                      Operation *diagnosticOp,
                                                      bool *emittedDiagnostic,
                                                      bool verifyForBackedge) {
  if (!value || !isTlaMutexType(value.getType()))
    return failure();
  if (!visited.insert(value).second)
    return failure();

  if (auto mutexOp = value.getDefiningOp<::tla::MutexOp>())
    return mutexOp;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *owner = blockArg.getOwner();
    auto forOp = owner ? dyn_cast_or_null<scf::ForOp>(owner->getParentOp())
                       : scf::ForOp();
    if (!forOp || owner != forOp.getBody())
      return failure();
    unsigned argNumber = blockArg.getArgNumber();
    if (argNumber < forOp.getNumInductionVars())
      return failure();
    unsigned iterIndex = argNumber - forOp.getNumInductionVars();
    if (iterIndex >= forOp.getInitArgs().size())
      return failure();

    FailureOr<::tla::MutexOp> initRoot = resolveMutexRootImpl(
        forOp.getInitArgs()[iterIndex], visited, diagnosticOp,
        emittedDiagnostic, verifyForBackedge);
    if (failed(initRoot))
      return failure();
    if (!verifyForBackedge)
      return *initRoot;

    auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yieldOp || iterIndex >= yieldOp.getNumOperands())
      return failure();
    Value yielded = yieldOp.getOperand(iterIndex);
    if (yielded == value)
      return *initRoot;

    DenseSet<Value> yieldVisited = visited;
    FailureOr<::tla::MutexOp> yieldRoot = resolveMutexRootImpl(
        yielded, yieldVisited, diagnosticOp, emittedDiagnostic, false);
    if (failed(yieldRoot))
      return failure();
    if (!sameMutexRoot(*initRoot, *yieldRoot)) {
      emitMutexResolveError(diagnosticOp, emittedDiagnostic,
                            "cannot lower tla.mutex through scf.for iter_arg "
                            "with different init/yield mutex operands");
      return failure();
    }
    return *initRoot;
  }

  if (auto forOp = value.getDefiningOp<scf::ForOp>()) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return failure();
    unsigned resultIndex = result.getResultNumber();
    if (resultIndex >= forOp.getInitArgs().size())
      return failure();
    auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yieldOp || resultIndex >= yieldOp.getNumOperands())
      return failure();

    DenseSet<Value> initVisited = visited;
    FailureOr<::tla::MutexOp> initRoot =
        resolveMutexRootImpl(forOp.getInitArgs()[resultIndex], initVisited,
                             diagnosticOp, emittedDiagnostic, true);
    DenseSet<Value> yieldVisited = visited;
    FailureOr<::tla::MutexOp> yieldRoot =
        resolveMutexRootImpl(yieldOp.getOperand(resultIndex), yieldVisited,
                             diagnosticOp, emittedDiagnostic, false);
    if (failed(initRoot) || failed(yieldRoot))
      return failure();
    if (!sameMutexRoot(*initRoot, *yieldRoot)) {
      emitMutexResolveError(diagnosticOp, emittedDiagnostic,
                            "cannot lower tla.mutex through scf.for result "
                            "with different init/yield mutex operands");
      return failure();
    }
    return *initRoot;
  }

  if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return failure();
    unsigned resultIndex = result.getResultNumber();
    scf::YieldOp thenYield = ifOp.thenYield();
    scf::YieldOp elseYield = ifOp.elseYield();
    if (!thenYield || !elseYield || resultIndex >= thenYield.getNumOperands() ||
        resultIndex >= elseYield.getNumOperands())
      return failure();

    DenseSet<Value> thenVisited = visited;
    FailureOr<::tla::MutexOp> thenRoot =
        resolveMutexRootImpl(thenYield.getOperand(resultIndex), thenVisited,
                             diagnosticOp, emittedDiagnostic, true);
    DenseSet<Value> elseVisited = visited;
    FailureOr<::tla::MutexOp> elseRoot =
        resolveMutexRootImpl(elseYield.getOperand(resultIndex), elseVisited,
                             diagnosticOp, emittedDiagnostic, true);
    if (failed(thenRoot) || failed(elseRoot))
      return failure();
    if (!sameMutexRoot(*thenRoot, *elseRoot)) {
      emitMutexResolveError(diagnosticOp, emittedDiagnostic,
                            "cannot lower tla.mutex through scf.if result with "
                            "different branch mutex operands");
      return failure();
    }
    return *thenRoot;
  }

  return failure();
}

static FailureOr<::tla::MutexOp> resolveMutexRootFromInit(Value value) {
  DenseSet<Value> visited;
  return resolveMutexRootImpl(value, visited, nullptr, nullptr, false);
}

static void eraseTrailingScfYield(Block *block) {
  if (!block || block->empty())
    return;
  if (auto yieldOp = dyn_cast<scf::YieldOp>(&block->back()))
    yieldOp.erase();
}

static bool hasNonScfYieldOps(Block *block) {
  if (!block)
    return false;
  return llvm::any_of(*block,
                      [](Operation &op) { return !isa<scf::YieldOp>(op); });
}

static FailureOr<Value> materializeMutexIdImpl(
    PatternRewriter &rewriter, Value value, Operation *diagnosticOp,
    ConstantMaterializer getOrCreateConstant, DenseSet<Value> &visited) {
  if (!value || !isTlaMutexType(value.getType()))
    return failure();
  if (!visited.insert(value).second)
    return failure();

  auto emitStaticMutexId = [&](::tla::MutexOp mutexOp) -> FailureOr<Value> {
    int64_t mutexId = mutexOp.getIdAttr().getInt();
    if (mutexId < 0) {
      diagnosticOp->emitError() << "mutex id auto allocation is not "
                                   "implemented for bitcode call lowering";
      return failure();
    }
    if (mutexId > 255) {
      diagnosticOp->emitError()
          << "mutex id must be in range 0..255 for bitcode call lowering";
      return failure();
    }
    return getOrCreateConstant(diagnosticOp, mutexId, 8);
  };

  if (auto mutexOp = value.getDefiningOp<::tla::MutexOp>())
    return emitStaticMutexId(mutexOp);

  if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return failure();
    unsigned resultIndex = result.getResultNumber();
    scf::YieldOp thenYield = ifOp.thenYield();
    scf::YieldOp elseYield = ifOp.elseYield();
    if (!thenYield || !elseYield || resultIndex >= thenYield.getNumOperands() ||
        resultIndex >= elseYield.getNumOperands())
      return failure();

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(ifOp);
    auto newIfOp = rewriter.create<scf::IfOp>(
        ifOp.getLoc(), rewriter.getI8Type(), ifOp.getCondition(), true);
    eraseTrailingScfYield(newIfOp.thenBlock());
    eraseTrailingScfYield(newIfOp.elseBlock());

    {
      PatternRewriter::InsertionGuard thenGuard(rewriter);
      rewriter.setInsertionPointToEnd(newIfOp.thenBlock());
      DenseSet<Value> thenVisited = visited;
      FailureOr<Value> thenId = materializeMutexIdImpl(
          rewriter, thenYield.getOperand(resultIndex), diagnosticOp,
          getOrCreateConstant, thenVisited);
      if (failed(thenId))
        return failure();
      rewriter.create<scf::YieldOp>(thenYield.getLoc(), *thenId);
    }
    {
      PatternRewriter::InsertionGuard elseGuard(rewriter);
      rewriter.setInsertionPointToEnd(newIfOp.elseBlock());
      DenseSet<Value> elseVisited = visited;
      FailureOr<Value> elseId = materializeMutexIdImpl(
          rewriter, elseYield.getOperand(resultIndex), diagnosticOp,
          getOrCreateConstant, elseVisited);
      if (failed(elseId))
        return failure();
      rewriter.create<scf::YieldOp>(elseYield.getLoc(), *elseId);
    }
    return newIfOp.getResult(0);
  }

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *owner = blockArg.getOwner();
    auto forOp = owner ? dyn_cast_or_null<scf::ForOp>(owner->getParentOp())
                       : scf::ForOp();
    if (!forOp || owner != forOp.getBody())
      return failure();
    unsigned argNumber = blockArg.getArgNumber();
    if (argNumber < forOp.getNumInductionVars())
      return failure();
    unsigned iterIndex = argNumber - forOp.getNumInductionVars();
    if (iterIndex >= forOp.getInitArgs().size())
      return failure();
    auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yieldOp || iterIndex >= yieldOp.getNumOperands() ||
        !isTransparentMutexForward(yieldOp.getOperand(iterIndex), value))
      return failure();
    return materializeMutexIdImpl(rewriter, forOp.getInitArgs()[iterIndex],
                                  diagnosticOp, getOrCreateConstant, visited);
  }

  if (auto forOp = value.getDefiningOp<scf::ForOp>()) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return failure();
    unsigned resultIndex = result.getResultNumber();
    if (resultIndex >= forOp.getInitArgs().size())
      return failure();
    auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yieldOp || resultIndex >= yieldOp.getNumOperands())
      return failure();
    Value regionIterArg = forOp.getRegionIterArg(resultIndex);
    if (!isTransparentMutexForward(yieldOp.getOperand(resultIndex),
                                   regionIterArg))
      return failure();
    return materializeMutexIdImpl(rewriter, forOp.getInitArgs()[resultIndex],
                                  diagnosticOp, getOrCreateConstant, visited);
  }

  bool emittedDiagnostic = false;
  DenseSet<Value> rootVisited;
  FailureOr<::tla::MutexOp> mutexRoot = resolveMutexRootImpl(
      value, rootVisited, diagnosticOp, &emittedDiagnostic, true);
  if (failed(mutexRoot))
    return failure();
  return emitStaticMutexId(*mutexRoot);
}

static FailureOr<Value>
materializeMutexId(PatternRewriter &rewriter, Value value,
                   Operation *diagnosticOp,
                   ConstantMaterializer getOrCreateConstant) {
  DenseSet<Value> visited;
  return materializeMutexIdImpl(rewriter, value, diagnosticOp,
                                getOrCreateConstant, visited);
}

static bool
shouldDropMutexForResult(scf::ForOp forOp, unsigned resultIndex,
                         SmallVectorImpl<Value> &replacementValues) {
  if (!isTlaMutexType(forOp.getResult(resultIndex).getType()))
    return false;
  auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yieldOp || resultIndex >= forOp.getInitArgs().size() ||
      resultIndex >= yieldOp.getNumOperands())
    return false;
  Value regionIterArg = forOp.getRegionIterArg(resultIndex);
  if (!isTransparentMutexForward(yieldOp.getOperand(resultIndex),
                                 regionIterArg))
    return false;
  replacementValues[resultIndex] = forOp.getInitArgs()[resultIndex];
  return true;
}

static bool cleanupMutexForOp(scf::ForOp forOp) {
  if (forOp.getNumResults() == 0)
    return false;

  SmallVector<Value> replacementValues(forOp.getNumResults());
  SmallVector<bool> dropResult(forOp.getNumResults(), false);
  bool changed = false;
  for (unsigned i = 0, e = forOp.getNumResults(); i < e; ++i) {
    dropResult[i] = shouldDropMutexForResult(forOp, i, replacementValues);
    changed |= dropResult[i];
  }
  if (!changed)
    return false;

  SmallVector<Value> keptInitArgs;
  SmallVector<unsigned> keptIndices;
  SmallVector<Value> keptYieldOperands;
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (unsigned i = 0, e = forOp.getNumResults(); i < e; ++i) {
    if (dropResult[i])
      continue;
    keptIndices.push_back(i);
    keptInitArgs.push_back(forOp.getInitArgs()[i]);
    keptYieldOperands.push_back(yieldOp.getOperand(i));
  }

  OpBuilder builder(forOp);
  IRMapping mapping;
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), keptInitArgs);
  newForOp->setAttrs(forOp->getAttrs());
  Block *newBody = newForOp.getBody();
  eraseTrailingScfYield(newBody);
  Block *oldBody = forOp.getBody();
  mapping.map(oldBody->getArgument(0), newBody->getArgument(0));
  for (Value replacement : replacementValues) {
    if (replacement)
      mapping.map(replacement, replacement);
  }
  for (auto [newArgIndex, oldResultIndex] : llvm::enumerate(keptIndices)) {
    mapping.map(oldBody->getArgument(oldResultIndex + 1),
                newBody->getArgument(newArgIndex + 1));
  }
  for (unsigned i = 0, e = forOp.getNumResults(); i < e; ++i) {
    if (dropResult[i])
      mapping.map(oldBody->getArgument(i + 1), replacementValues[i]);
  }

  builder.setInsertionPointToEnd(newBody);
  for (Operation &op : oldBody->without_terminator())
    builder.clone(op, mapping);

  SmallVector<Value> mappedYieldOperands;
  for (Value operand : keptYieldOperands)
    mappedYieldOperands.push_back(mapping.lookupOrDefault(operand));
  builder.create<scf::YieldOp>(yieldOp.getLoc(), mappedYieldOperands);

  for (unsigned i = 0, e = forOp.getNumResults(); i < e; ++i) {
    if (dropResult[i] && replacementValues[i])
      forOp.getResult(i).replaceAllUsesWith(replacementValues[i]);
  }
  for (auto [newResultIndex, oldResultIndex] : llvm::enumerate(keptIndices))
    forOp.getResult(oldResultIndex)
        .replaceAllUsesWith(newForOp.getResult(newResultIndex));
  forOp.erase();
  return true;
}

static bool shouldDropMutexIfResult(scf::IfOp ifOp, unsigned resultIndex,
                                    SmallVectorImpl<Value> &replacementValues) {
  if (!isTlaMutexType(ifOp.getResult(resultIndex).getType()))
    return false;
  if (ifOp.getResult(resultIndex).use_empty())
    return true;

  scf::YieldOp thenYield = ifOp.thenYield();
  scf::YieldOp elseYield = ifOp.elseYield();
  if (!thenYield || !elseYield || resultIndex >= thenYield.getNumOperands() ||
      resultIndex >= elseYield.getNumOperands())
    return false;

  FailureOr<::tla::MutexOp> thenRoot =
      resolveMutexRootFromInit(thenYield.getOperand(resultIndex));
  FailureOr<::tla::MutexOp> elseRoot =
      resolveMutexRootFromInit(elseYield.getOperand(resultIndex));
  if (failed(thenRoot) || failed(elseRoot) ||
      !sameMutexRoot(*thenRoot, *elseRoot))
    return false;
  replacementValues[resultIndex] = thenRoot->getResult();
  return true;
}

static bool cleanupMutexIfOp(scf::IfOp ifOp) {
  if (ifOp.getNumResults() == 0)
    return false;

  SmallVector<Value> replacementValues(ifOp.getNumResults());
  SmallVector<bool> dropResult(ifOp.getNumResults(), false);
  bool changed = false;
  for (unsigned i = 0, e = ifOp.getNumResults(); i < e; ++i) {
    dropResult[i] = shouldDropMutexIfResult(ifOp, i, replacementValues);
    changed |= dropResult[i];
  }
  if (!changed)
    return false;

  SmallVector<Type> keptResultTypes;
  SmallVector<unsigned> keptIndices;
  SmallVector<Value> keptThenOperands;
  SmallVector<Value> keptElseOperands;
  scf::YieldOp thenYield = ifOp.thenYield();
  scf::YieldOp elseYield = ifOp.elseYield();
  for (unsigned i = 0, e = ifOp.getNumResults(); i < e; ++i) {
    if (dropResult[i])
      continue;
    keptIndices.push_back(i);
    keptResultTypes.push_back(ifOp.getResult(i).getType());
    keptThenOperands.push_back(thenYield.getOperand(i));
    keptElseOperands.push_back(elseYield.getOperand(i));
  }

  if (keptResultTypes.empty() && !hasNonScfYieldOps(ifOp.thenBlock()) &&
      !hasNonScfYieldOps(ifOp.elseBlock())) {
    for (unsigned i = 0, e = ifOp.getNumResults(); i < e; ++i) {
      if (dropResult[i] && replacementValues[i])
        ifOp.getResult(i).replaceAllUsesWith(replacementValues[i]);
    }
    ifOp.erase();
    return true;
  }

  OpBuilder builder(ifOp);
  auto newIfOp = builder.create<scf::IfOp>(ifOp.getLoc(), keptResultTypes,
                                           ifOp.getCondition(), true);
  newIfOp->setAttrs(ifOp->getAttrs());
  eraseTrailingScfYield(newIfOp.thenBlock());
  eraseTrailingScfYield(newIfOp.elseBlock());

  IRMapping thenMapping;
  builder.setInsertionPointToEnd(newIfOp.thenBlock());
  for (Operation &op : ifOp.thenBlock()->without_terminator())
    builder.clone(op, thenMapping);
  SmallVector<Value> mappedThenOperands;
  for (Value operand : keptThenOperands)
    mappedThenOperands.push_back(thenMapping.lookupOrDefault(operand));
  builder.create<scf::YieldOp>(thenYield.getLoc(), mappedThenOperands);

  IRMapping elseMapping;
  builder.setInsertionPointToEnd(newIfOp.elseBlock());
  for (Operation &op : ifOp.elseBlock()->without_terminator())
    builder.clone(op, elseMapping);
  SmallVector<Value> mappedElseOperands;
  for (Value operand : keptElseOperands)
    mappedElseOperands.push_back(elseMapping.lookupOrDefault(operand));
  builder.create<scf::YieldOp>(elseYield.getLoc(), mappedElseOperands);

  for (unsigned i = 0, e = ifOp.getNumResults(); i < e; ++i) {
    if (dropResult[i] && replacementValues[i])
      ifOp.getResult(i).replaceAllUsesWith(replacementValues[i]);
  }
  for (auto [newResultIndex, oldResultIndex] : llvm::enumerate(keptIndices))
    ifOp.getResult(oldResultIndex)
        .replaceAllUsesWith(newIfOp.getResult(newResultIndex));
  ifOp.erase();
  return true;
}

static void cleanupMutexControlFlow(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *, 8> controlFlowOps;
    module.walk([&](Operation *op) {
      if (llvm::isa<scf::ForOp, scf::IfOp>(op))
        controlFlowOps.push_back(op);
    });
    for (Operation *op : llvm::reverse(controlFlowOps)) {
      if (!op || !op->getBlock())
        continue;
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        changed |= cleanupMutexForOp(forOp);
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(op))
        changed |= cleanupMutexIfOp(ifOp);
    }
  }
}

template <typename MutexOpT>
struct LowerTlaMutexAccessPattern : public OpRewritePattern<MutexOpT> {
  LowerTlaMutexAccessPattern(MLIRContext *ctx, ModuleOp module,
                             StringRef calleePrefix,
                             ConstantMaterializer getOrCreateConstant)
      : OpRewritePattern<MutexOpT>(ctx), module(module),
        calleePrefix(calleePrefix.str()),
        getOrCreateConstant(getOrCreateConstant) {}

  LogicalResult matchAndRewrite(MutexOpT op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1) {
      op.emitError() << "expected " << op->getName().getStringRef()
                     << " to have exactly 1 operand";
      return failure();
    }

    FailureOr<Value> mutexId = materializeMutexId(
        rewriter, op.getMutex(), op.getOperation(), getOrCreateConstant);
    if (failed(mutexId)) {
      op.emitError() << "expected " << op->getName().getStringRef()
                     << " to have a lowerable tla.mutex operand";
      return failure();
    }

    FailureOr<std::string> pipeSuffix = getMutexPipeSuffix(op.getPipe());
    if (failed(pipeSuffix)) {
      op.emitError() << "unsupported pipe for mutex bitcode call lowering";
      return failure();
    }

    auto i8Type = rewriter.getI8Type();
    SmallVector<Type, 1> operandTypes = {i8Type};
    auto callee = getOrCreateRuntimeCall(
        module, calleePrefix + "_" + *pipeSuffix, operandTypes);
    rewriter.create<func::CallOp>(op.getLoc(), callee, ValueRange{*mutexId});
    return success();
  }

private:
  ModuleOp module;
  std::string calleePrefix;
  ConstantMaterializer getOrCreateConstant;
};

struct ConstantKey {
  int64_t value;
  unsigned bits;

  bool operator==(const ConstantKey &other) const {
    return value == other.value && bits == other.bits;
  }
};

struct ConstantKeyInfo {
  static inline ConstantKey getEmptyKey() {
    return {std::numeric_limits<int64_t>::min(), 0};
  }
  static inline ConstantKey getTombstoneKey() {
    return {std::numeric_limits<int64_t>::min() + 1, 0};
  }
  static unsigned getHashValue(const ConstantKey &key) {
    return llvm::hash_combine(key.value, key.bits);
  }
  static bool isEqual(const ConstantKey &lhs, const ConstantKey &rhs) {
    return lhs == rhs;
  }
};

class TlaLowerMutexToStdPass
    : public PassWrapper<TlaLowerMutexToStdPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerMutexToStdPass)

  StringRef getArgument() const override { return "tla-lower-mutex-to-std"; }
  StringRef getName() const override { return "TlaLowerMutexToStdPass"; }
  StringRef getDescription() const override {
    return "Lower Tla mutex ops to standard MLIR runtime calls.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    DenseMap<Block *, DenseMap<ConstantKey, Value, ConstantKeyInfo>>
        constantByScope;
    auto getOrCreateConstant = [&](Operation *anchor, int64_t value,
                                   unsigned bits) -> Value {
      ConstantKey key{value, bits};
      Block *scopeBlock = nullptr;
      if (auto tlaFunc = anchor->getParentOfType<::tla::FuncOp>()) {
        scopeBlock = &tlaFunc.getBody().front();
      } else if (auto func = anchor->getParentOfType<mlir::func::FuncOp>()) {
        scopeBlock = &func.getBody().front();
      } else if (auto module = anchor->getParentOfType<ModuleOp>()) {
        scopeBlock = &module.getBodyRegion().front();
      } else {
        scopeBlock = anchor->getBlock();
      }
      auto &cache = constantByScope[scopeBlock];
      auto it = cache.find(key);
      if (it != cache.end())
        return it->second;

      OpBuilder builder(scopeBlock, scopeBlock->begin());
      Value constant;
      if (bits == 0) {
        constant =
            builder.create<arith::ConstantIndexOp>(anchor->getLoc(), value);
      } else {
        auto intType = builder.getIntegerType(bits);
        auto intAttr = builder.getIntegerAttr(intType, value);
        constant = builder.create<arith::ConstantOp>(anchor->getLoc(), intType,
                                                     intAttr);
      }
      cache[key] = constant;
      return constant;
    };

    if (failed(lowerTlaMutexToStd(getOperation(), getOrCreateConstant)))
      signalPassFailure();
  }
};

} // namespace

LogicalResult lowerTlaMutexToStd(ModuleOp module,
                                 ConstantMaterializer getOrCreateConstant) {
  LowerTlaMutexAccessPattern<::tla::MutexLockOp> lowerMutexLock(
      module.getContext(), module, "get_buf", getOrCreateConstant);
  LowerTlaMutexAccessPattern<::tla::MutexUnlockOp> lowerMutexUnlock(
      module.getContext(), module, "rls_buf", getOrCreateConstant);
  SmallVector<Operation *, 8> mutexUseOps;
  module.walk([&](Operation *op) {
    if (llvm::isa<::tla::MutexLockOp, ::tla::MutexUnlockOp>(op))
      mutexUseOps.push_back(op);
  });

  for (Operation *op : mutexUseOps) {
    if (!op->getBlock())
      continue;
    PatternRewriter rewriter(op->getContext());
    rewriter.setInsertionPoint(op);
    LogicalResult lowered = success();
    if (auto lockOp = llvm::dyn_cast<::tla::MutexLockOp>(op)) {
      lowered = lowerMutexLock.matchAndRewrite(lockOp, rewriter);
    } else if (auto unlockOp = llvm::dyn_cast<::tla::MutexUnlockOp>(op)) {
      lowered = lowerMutexUnlock.matchAndRewrite(unlockOp, rewriter);
    }
    if (failed(lowered))
      return failure();
  }

  for (Operation *op : mutexUseOps) {
    if (op && op->getBlock())
      op->erase();
  }

  cleanupMutexControlFlow(module);

  SmallVector<::tla::MutexOp, 8> tlaMutexOps;
  module.walk([&](::tla::MutexOp op) { tlaMutexOps.push_back(op); });
  for (::tla::MutexOp mutexOp : tlaMutexOps) {
    Operation *mutexRaw = mutexOp.getOperation();
    if (!mutexRaw->getBlock())
      continue;
    if (!mutexOp.getResult().use_empty())
      continue;
    mutexRaw->erase();
  }

  return success();
}

std::unique_ptr<Pass> createTlaLowerMutexToStdPass() {
  return std::make_unique<TlaLowerMutexToStdPass>();
}

void registerTlaLowerMutexToStdPass() {
  PassRegistration<TlaLowerMutexToStdPass>();
}

} // namespace tla
