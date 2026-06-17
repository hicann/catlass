// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include "PassesCommon.h"
#include "PassesInternal.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace tla {
namespace {

static constexpr StringLiteral kSplitMixedExecUnitsAttrValue = "cube_vector";
static constexpr StringLiteral kCubeExecUnitsAttrValue = "cube";
static constexpr StringLiteral kVectorExecUnitsAttrValue = "vector";
static bool isMixedSplitCandidate(func::FuncOp funcOp) {
  if (funcOp.isDeclaration() || isPrivateSymbol(funcOp))
    return false;
  if (!funcOp->hasAttr(kTlaMixedSplitAttrName))
    return false;
  auto execUnits = funcOp->getAttrOfType<StringAttr>(kTlaExecUnitsAttrName);
  auto coreType = funcOp->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
  return execUnits && execUnits.getValue() == kSplitMixedExecUnitsAttrValue && coreType &&
         coreType.getFuncCoreType() == hivm::TFuncCoreType::MIX;
}

struct MixedRegionTopology {
  SmallVector<Operation *, 8> aicRoots;
  SmallVector<Operation *, 8> aivRoots;
};

enum class SplitSide { AIC, AIV };

static constexpr std::array<StringLiteral, 8> kAllowedLeafOpsOutsideMixedRegions = {
    "tla.alloc_ptr",          "tla.recast_ptr", "tla.arch.block_idx", "tla.arch.block_dim",
    "tla.arch.sub_block_idx", "tla.flag",       "tla.cross_flag",     "arith.constant"};

static bool containsCubeRegion(Operation *op) {
  if (isa<::tla::CubeOp>(op))
    return true;
  return op->walk([&](::tla::CubeOp) { return WalkResult::interrupt(); }).wasInterrupted();
}

static bool containsVectorRegion(Operation *op) {
  if (isa<::tla::VectorOp>(op))
    return true;
  return op->walk([&](::tla::VectorOp) { return WalkResult::interrupt(); }).wasInterrupted();
}

static bool isAllowedPipeBarrierOutsideMixedRegion(Operation *op) {
  auto barrier = dyn_cast<::tla::PipeBarrierOp>(op);
  return barrier && stringifyPipe(barrier.getPipe().getPipe()) == "all";
}

static bool isAllowedSpecialCaseOutsideMixedRegion(Operation *op) {
  return isAllowedPipeBarrierOutsideMixedRegion(op);
}

static bool isAllowedLeafOutsideMixedRegion(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (llvm::is_contained(kAllowedLeafOpsOutsideMixedRegions, name))
    return true;
  return isAllowedSpecialCaseOutsideMixedRegion(op);
}

static LogicalResult validateAllowedOpsOutsideMixedRegions(Block &block);

static LogicalResult validateAllowedOpOutsideMixedRegions(Operation *op) {
  if (isa<::tla::CubeOp, ::tla::VectorOp>(op))
    return success();
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return validateAllowedOpsOutsideMixedRegions(*forOp.getBody());
  if (isAllowedLeafOutsideMixedRegion(op))
    return success();

  return op->emitOpError() << "is not allowed outside tla.cube or tla.vector in a mixed kernel";
}

static LogicalResult validateAllowedOpsOutsideMixedRegions(Block &block) {
  for (Operation &op : block.without_terminator()) {
    if (failed(validateAllowedOpOutsideMixedRegions(&op)))
      return failure();
  }
  return success();
}

static FailureOr<MixedRegionTopology> analyzeMixedRegionTopology(func::FuncOp funcOp) {
  if (funcOp.getBody().empty())
    return failure();

  Block &entry = funcOp.getBody().front();
  SmallVector<Operation *, 8> aicRoots;
  SmallVector<Operation *, 8> aivRoots;

  for (Operation &op : entry.without_terminator()) {
    if (containsCubeRegion(&op))
      aicRoots.push_back(&op);
    if (containsVectorRegion(&op))
      aivRoots.push_back(&op);
  }

  if (aicRoots.empty() || aivRoots.empty())
    return failure();

  return MixedRegionTopology{aicRoots, aivRoots};
}

static void collectReferencedValues(Operation *op, SmallVectorImpl<Value> &values) {
  values.append(op->operand_begin(), op->operand_end());
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block)
        collectReferencedValues(&nested, values);
    }
  }
}

static FailureOr<SmallVector<Operation *, 16>>
computeOrderedTopLevelSlice(Block &entry, ArrayRef<Operation *> roots) {
  SmallPtrSet<Operation *, 16> required;
  SmallVector<Operation *, 16> worklist;
  for (Operation *root : roots) {
    if (required.insert(root).second)
      worklist.push_back(root);
  }

  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    SmallVector<Value> referencedValues;
    collectReferencedValues(current, referencedValues);
    for (Value value : referencedValues) {
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        if (blockArg.getOwner() == &entry)
          continue;
        continue;
      }

      Operation *def = value.getDefiningOp();
      if (!def)
        continue;

      Operation *topLevel = def;
      while (topLevel && topLevel->getBlock() != &entry)
        topLevel = topLevel->getParentOp();
      if (!topLevel || topLevel->getBlock() != &entry)
        return failure();
      if (required.insert(topLevel).second)
        worklist.push_back(topLevel);
    }
  }

  SmallVector<Operation *, 16> ordered;
  for (Operation &op : entry.without_terminator()) {
    if (required.contains(&op))
      ordered.push_back(&op);
  }
  return ordered;
}

static bool containsRegionForSide(Operation *op, SplitSide side) {
  return side == SplitSide::AIC ? containsCubeRegion(op) : containsVectorRegion(op);
}

static bool isRegionWrapperForSide(Operation *op, SplitSide side) {
  return side == SplitSide::AIC ? isa<::tla::CubeOp>(op) : isa<::tla::VectorOp>(op);
}

static FailureOr<SmallVector<Operation *, 16>>
computeOrderedBlockSlice(Block &block, ArrayRef<Operation *> roots) {
  SmallPtrSet<Operation *, 16> required;
  SmallVector<Operation *, 16> worklist;
  for (Operation *root : roots) {
    if (required.insert(root).second)
      worklist.push_back(root);
  }

  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    SmallVector<Value> referencedValues;
    collectReferencedValues(current, referencedValues);
    for (Value value : referencedValues) {
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        if (blockArg.getOwner() == &block)
          continue;
        continue;
      }

      Operation *def = value.getDefiningOp();
      if (!def)
        continue;

      Operation *blockLocal = def;
      while (blockLocal && blockLocal->getBlock() != &block)
        blockLocal = blockLocal->getParentOp();
      if (!blockLocal || blockLocal->getBlock() != &block)
        continue;
      if (required.insert(blockLocal).second)
        worklist.push_back(blockLocal);
    }
  }

  SmallVector<Operation *, 16> ordered;
  for (Operation &op : block.without_terminator()) {
    if (required.contains(&op))
      ordered.push_back(&op);
  }
  return ordered;
}

static LogicalResult filterBlockForSide(Block &block, SplitSide side);

static LogicalResult filterOperationRegionsForSide(Operation *op, SplitSide side) {
  if (isRegionWrapperForSide(op, side))
    return success();

  for (Region &region : op->getRegions()) {
    for (Block &nestedBlock : region) {
      if (failed(filterBlockForSide(nestedBlock, side)))
        return failure();
    }
  }
  return success();
}

static LogicalResult filterBlockForSide(Block &block, SplitSide side) {
  SmallVector<Operation *, 16> roots;
  for (Operation &op : block.without_terminator()) {
    if (containsRegionForSide(&op, side))
      roots.push_back(&op);
  }
  if (roots.empty())
    return success();

  FailureOr<SmallVector<Operation *, 16>> ordered = computeOrderedBlockSlice(block, roots);
  if (failed(ordered))
    return failure();

  SmallPtrSet<Operation *, 16> keep(ordered->begin(), ordered->end());
  SmallVector<Operation *, 16> eraseList;
  for (Operation &op : block.without_terminator()) {
    if (!keep.contains(&op))
      eraseList.push_back(&op);
  }
  for (Operation *op : llvm::reverse(eraseList))
    op->erase();

  for (Operation *op : *ordered) {
    if (!op->getBlock())
      continue;
    if (failed(filterOperationRegionsForSide(op, side)))
      return failure();
  }
  return success();
}

static bool isDroppableUnusedSplitHelper(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "tla.flag" || name == "tla.cross_flag" || name == "tla.alloc_ptr" ||
         name == "tla.recast_ptr";
}

static void eraseUnusedSplitHelpers(Block &entry) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : llvm::make_early_inc_range(entry.without_terminator())) {
      if (op.use_empty() && isDroppableUnusedSplitHelper(&op)) {
        op.erase();
        changed = true;
      }
    }
  }
}

static bool isFrontendVectorRegionWrapper(::tla::VectorOp vectorOp) {
  auto role = vectorOp->getAttrOfType<StringAttr>("tla.vector_role");
  return role && role.getValue() == "region";
}

static void inlineFrontendVectorRegionWrappers(func::FuncOp funcOp) {
  SmallVector<::tla::VectorOp, 4> wrappers;
  funcOp.walk([&](::tla::VectorOp vectorOp) {
    if (isFrontendVectorRegionWrapper(vectorOp))
      wrappers.push_back(vectorOp);
  });

  IRRewriter rewriter(funcOp.getContext());
  for (::tla::VectorOp vectorOp : wrappers) {
    if (!vectorOp || vectorOp.getBody().empty())
      continue;
    Block *body = &vectorOp.getBody().front();
    rewriter.inlineBlockBefore(body, vectorOp->getBlock(), vectorOp->getIterator());
    rewriter.eraseOp(vectorOp);
  }
}

static FailureOr<func::FuncOp> createSplitFunctionLike(func::FuncOp source, StringRef newName,
                                                       ArrayRef<Operation *> orderedOps,
                                                       StringRef execUnits, HivmCoreKind coreKind) {
  MLIRContext *ctx = source.getContext();
  auto newType = source.getFunctionType();
  auto newFunc = func::FuncOp::create(source.getLoc(), newName, newType);
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name == SymbolTable::getSymbolAttrName() || name == "function_type" ||
        name == kTlaMixedSplitAttrName)
      continue;
    newFunc->setAttr(attr.getName(), attr.getValue());
  }

  newFunc->setAttr(kTlaExecUnitsAttrName, StringAttr::get(ctx, execUnits));
  setRequiredHaccEntryAttrs(newFunc, ctx);
  setC310RegbaseTargetAttr(newFunc, ctx);
  newFunc->setAttr(hivm::TFuncCoreTypeAttr::name,
                   hivm::TFuncCoreTypeAttr::get(ctx, toFuncCoreType(coreKind)));
  newFunc->setAttr(kMixModeAttrName, StringAttr::get(ctx, "mix"));
  newFunc->setAttr(kParallelModeAttrName, StringAttr::get(ctx, "simd"));
  newFunc->setAttr(hivm::TPartOfMixAttr::name, UnitAttr::get(ctx));
  newFunc->setAttr(hivm::VFModeAttr::name, hivm::VFModeAttr::get(ctx, hivm::VFMode::SIMD));

  Block *newEntry = newFunc.addEntryBlock();
  IRMapping mapper;
  OpBuilder builder(newEntry, newEntry->begin());
  for (auto [oldArg, newArg] : llvm::zip_equal(source.getArguments(), newEntry->getArguments()))
    mapper.map(oldArg, newArg);
  SplitSide side = coreKind == HivmCoreKind::AIC ? SplitSide::AIC : SplitSide::AIV;
  for (Operation *op : orderedOps) {
    Operation *cloned = builder.clone(*op, mapper);
    if (failed(filterOperationRegionsForSide(cloned, side))) {
      newFunc.emitError()
          << "frontend-declared mixed split could not isolate nested region ownership";
      return failure();
    }
  }
  if (side == SplitSide::AIV)
    inlineFrontendVectorRegionWrappers(newFunc);
  eraseUnusedSplitHelpers(*newEntry);
  builder.create<func::ReturnOp>(source.getLoc());
  return newFunc;
}

class TlaSplitMixedFuncPass : public PassWrapper<TlaSplitMixedFuncPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaSplitMixedFuncPass)

  StringRef getArgument() const override { return "tla-split-mixed-func"; }
  StringRef getName() const override { return "TlaSplitMixedFuncPass"; }
  StringRef getDescription() const override {
    return "Materialize frontend-declared mixed Tla functions into AIC and AIV functions.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, scf::SCFDialect, ::tla::TlaDialect, hacc::HACCDialect,
                    hivm::HIVMDialect, hivm_regbaseintrins::HIVMRegbaseIntrinsDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp, 4> candidates;
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (isMixedSplitCandidate(funcOp))
        candidates.push_back(funcOp);
    }

    for (func::FuncOp funcOp : candidates) {
      if (failed(validateAllowedOpsOutsideMixedRegions(funcOp.getBody().front()))) {
        signalPassFailure();
        return;
      }

      FailureOr<MixedRegionTopology> topology = analyzeMixedRegionTopology(funcOp);
      if (failed(topology)) {
        funcOp.emitError()
            << "frontend-declared mixed split requires at least one tla.cube and one tla.vector "
               "region";
        signalPassFailure();
        return;
      }
      FailureOr<SmallVector<Operation *, 16>> aicOps =
          computeOrderedTopLevelSlice(funcOp.getBody().front(), topology->aicRoots);
      FailureOr<SmallVector<Operation *, 16>> aivOps =
          computeOrderedTopLevelSlice(funcOp.getBody().front(), topology->aivRoots);
      if (failed(aicOps) || failed(aivOps)) {
        funcOp.emitError()
            << "frontend-declared mixed split could not compute a valid top-level slice for split "
               "functions";
        signalPassFailure();
        return;
      }

      OpBuilder builder(funcOp);
      FailureOr<func::FuncOp> aicFunc =
          createSplitFunctionLike(funcOp, (funcOp.getSymName() + "_mix_aic").str(), *aicOps,
                                  kCubeExecUnitsAttrValue, HivmCoreKind::AIC);
      FailureOr<func::FuncOp> aivFunc =
          createSplitFunctionLike(funcOp, (funcOp.getSymName() + "_mix_aiv").str(), *aivOps,
                                  kVectorExecUnitsAttrValue, HivmCoreKind::AIV);
      if (failed(aicFunc) || failed(aivFunc)) {
        signalPassFailure();
        return;
      }
      builder.insert(*aicFunc);
      builder.insert(*aivFunc);
      funcOp.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTlaSplitMixedFuncPass() {
  return std::make_unique<TlaSplitMixedFuncPass>();
}

void registerTlaSplitMixedFuncPass() { PassRegistration<TlaSplitMixedFuncPass>(); }

} // namespace tla
