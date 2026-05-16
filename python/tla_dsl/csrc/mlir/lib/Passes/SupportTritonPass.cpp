#include "PassesCommon.h"

namespace tla {
namespace {

class SupportTritonPass : public PassWrapper<SupportTritonPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SupportTritonPass)

  StringRef getArgument() const override { return "enable-triton"; }
  StringRef getName() const override { return "SupportTritonPass"; }
  StringRef getDescription() const override {
    return "Add Triton-style hidden entry arguments to the head function only.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, hacc::HACCDialect, hivm::HIVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr(kTlaHasVectorRegionAttrName))
      return;
    SmallVector<func::FuncOp, 4> candidates;
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.isDeclaration())
        continue;
      if (funcOp.isPrivate())
        continue;
      if (!funcOp->hasAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY)))
        continue;
      candidates.push_back(funcOp);
    }

    for (func::FuncOp funcOp : candidates) {
      if (failed(rewriteEntrySignature(funcOp))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  // Triton-adapted Ascend entry kernels are expected to start with three hidden
  // HACC/HIVM ABI arguments before the user-visible tensor parameters:
  //   1. `ffts_base_address` (`kFFTSBaseAddr`, i64): base address consumed by
  //      HIVM sync/block infrastructure such as `hivm.set_ffts_base_addr`.
  //   2. `sync_block_lock` (`kSyncBlockLock`, memref<?xi8, gm>): backing storage
  //      for cross-block synchronization/lock state.
  //   3. `workspace` (`kWorkspace`, memref<?xi8, gm>): global scratch buffer
  //      used by downstream workspace alloc/binding passes.
  //
  // These names and roles come from AscendNPU-IR's HACC kernel arg kinds.
  static bool hasTritonLeadingArgs(func::FuncOp funcOp) {
    if (funcOp.getNumArguments() < 3)
      return false;
    auto ffts = funcOp.getArgAttrOfType<hacc::KernelArgTypeAttr>(0, hacc::KernelArgTypeAttr::name);
    auto sync = funcOp.getArgAttrOfType<hacc::KernelArgTypeAttr>(1, hacc::KernelArgTypeAttr::name);
    auto workspace =
        funcOp.getArgAttrOfType<hacc::KernelArgTypeAttr>(2, hacc::KernelArgTypeAttr::name);
    return ffts && sync && workspace && ffts.getArgType() == hacc::KernelArgType::kFFTSBaseAddr &&
           sync.getArgType() == hacc::KernelArgType::kSyncBlockLock &&
           workspace.getArgType() == hacc::KernelArgType::kWorkspace;
  }

  static DenseElementsAttr buildDynMemrefMaskAttr(MLIRContext *ctx, int64_t count) {
    auto i1Ty = IntegerType::get(ctx, 1);
    auto vecTy = VectorType::get({count}, i1Ty);
    SmallVector<bool, 16> bits(count, false);
    if (count > 1)
      bits[1] = true;
    if (count > 2)
      bits[2] = true;
    return DenseElementsAttr::get(vecTy, ArrayRef<bool>(bits));
  }

  static FailureOr<MemRefType> buildDynamicByteMemref(MLIRContext *ctx) {
    auto memspace = mapTlaAddressSpaceToHivmMemspace(ctx, AddressSpace::gm);
    if (failed(memspace))
      return failure();
    return MemRefType::get({ShapedType::kDynamic}, IntegerType::get(ctx, 8), AffineMap(),
                           *memspace);
  }

  static LogicalResult rewriteEntrySignature(func::FuncOp funcOp) {
    if (hasTritonLeadingArgs(funcOp))
      return success();

    MLIRContext *ctx = funcOp.getContext();
    auto dynByteMemref = buildDynamicByteMemref(ctx);
    if (failed(dynByteMemref))
      return failure();

    SmallVector<Type, 8> newInputs;
    newInputs.reserve(funcOp.getNumArguments() + 3);
    // Prepend the hidden Triton-support ABI arguments in the same order as the
    // AscendNPU-IR HACC kernel arg kinds: FFTS base, sync-block lock, workspace.
    newInputs.push_back(IntegerType::get(ctx, 64));
    newInputs.push_back(*dynByteMemref);
    newInputs.push_back(*dynByteMemref);
    llvm::append_range(newInputs, funcOp.getArgumentTypes());

    funcOp.setType(FunctionType::get(ctx, newInputs, funcOp.getResultTypes()));
    Block &entry = funcOp.getBody().front();
    Location loc = funcOp.getLoc();
    entry.insertArgument(static_cast<unsigned>(0), newInputs[0], loc);
    entry.insertArgument(static_cast<unsigned>(1), newInputs[1], loc);
    entry.insertArgument(static_cast<unsigned>(2), newInputs[2], loc);

    funcOp.setArgAttr(0, hacc::KernelArgTypeAttr::name,
                      hacc::KernelArgTypeAttr::get(ctx, hacc::KernelArgType::kFFTSBaseAddr));
    funcOp.setArgAttr(1, hacc::KernelArgTypeAttr::name,
                      hacc::KernelArgTypeAttr::get(ctx, hacc::KernelArgType::kSyncBlockLock));
    funcOp.setArgAttr(2, hacc::KernelArgTypeAttr::name,
                      hacc::KernelArgTypeAttr::get(ctx, hacc::KernelArgType::kWorkspace));
    funcOp->setAttr(hivm::HIVMFuncDynMemrefArgsAttr::getMnemonic(),
                    buildDynMemrefMaskAttr(ctx, newInputs.size()));
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createSupportTritonPass() { return std::make_unique<SupportTritonPass>(); }

void registerSupportTritonPass() { PassRegistration<SupportTritonPass>(); }

} // namespace tla
