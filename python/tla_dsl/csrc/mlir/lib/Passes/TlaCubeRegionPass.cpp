#include "Dialect/Tla/IR/TlaAttrs.h"
#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaTensorToMemref.h"

// tla-cube-region: lowers the cube (AIC) compute ops (tla.copy / tla.mmad) via
// the shared !tla.tensor->memref lowering (TlaTensorMemrefLowering), and flattens
// the tla.cube region. Runs after tla-vector-region, before tla-finalize-memref.
//
// Ordering mirrors the previous in-lower-to-std flow: derive descriptors, lower
// tla.copy (descriptor + payload driven), lower tile producers to memref.subview
// (so tla.mmad operands are memref-backed), flatten cube regions, lower tla.mmad,
// then the late type-driven tla.copy cleanup. The dead scaffolding + unrealized
// casts that remain are DCE'd by the downstream cleanup pass (tla-finalize-memref).

namespace tla {
namespace {

  static FailureOr<hivm::AddressSpace> resolveHivmAddressSpace(MLIRContext *ctx,
                                                               StringRef addressSpace) {
    auto tlaAddressSpace = symbolizeAddressSpace(addressSpace);
    if (!tlaAddressSpace)
      return failure();
    FailureOr<Attribute> memorySpaceOr = mapTlaAddressSpaceToHivmMemspace(ctx, *tlaAddressSpace);
    if (failed(memorySpaceOr))
      return failure();
    auto memorySpaceAttr = dyn_cast<hivm::AddressSpaceAttr>(*memorySpaceOr);
    if (!memorySpaceAttr)
      return failure();
    return memorySpaceAttr.getAddressSpace();
  }

  static StringRef copyRuntimeElemSuffix(StringRef elementType) {
    if (elementType == "f32")
      return "float";
    if (elementType == "f16")
      return "half";
    if (elementType == "bf16")
      return "bf16";
    return {};
  }

  static std::string getCopyRouteCallee(MLIRContext *ctx, StringRef srcAddrspace,
                                        StringRef dstAddrspace, TensorLayoutTag srcLayout,
                                        TensorLayoutTag dstLayout, StringRef srcElementType,
                                        StringRef dstElementType, StringRef extraDesc="") {
    FailureOr<hivm::AddressSpace> srcSpace = resolveHivmAddressSpace(ctx, srcAddrspace);
    FailureOr<hivm::AddressSpace> dstSpace = resolveHivmAddressSpace(ctx, dstAddrspace);
    if (failed(srcSpace) || failed(dstSpace))
      return {};
    StringRef dstElem = dstElementType.empty() ? srcElementType : dstElementType;

    // Copy routing is keyed by explicit (addrspace, layout-tag) pairs.
    // Runtime symbol names encode both endpoint layout tags so future layout
    // variants can be added as new explicit routes instead of overloading
    // addrspace-only names.
    // Current TLA MLIR carries explicit layout tags through tla.make_tensor_like and
    // optionally on tla.tile_view.
    if (*srcSpace == hivm::AddressSpace::UB && *dstSpace == hivm::AddressSpace::L1 &&
        srcLayout == TensorLayoutTag::RowMajor && dstLayout == TensorLayoutTag::zN) {
      if (srcElementType != dstElem)
        return {};
      StringRef suffix = copyRuntimeElemSuffix(srcElementType);
      if (suffix.empty())
        return {};
      return Twine("copy_ubuf_row_major_to_cbuf_zN_").concat(suffix).str();
    }
    if (*srcSpace == hivm::AddressSpace::GM && *dstSpace == hivm::AddressSpace::L1 &&
        srcLayout == TensorLayoutTag::RowMajor && dstLayout == TensorLayoutTag::zN) {
      if (srcElementType != dstElem)
        return {};
      StringRef suffix = copyRuntimeElemSuffix(srcElementType);
      if (suffix.empty())
        return {};
      return Twine("copy_gm_row_major_to_cbuf_zN_").concat(suffix).str();
    }
    if (*srcSpace == hivm::AddressSpace::GM && *dstSpace == hivm::AddressSpace::L1 &&
        srcLayout == TensorLayoutTag::ColumnMajor && dstLayout == TensorLayoutTag::nZ) {
      if (srcElementType != dstElem)
        return {};
      StringRef suffix = copyRuntimeElemSuffix(srcElementType);
      if (suffix.empty())
        return {};
      return Twine("copy_gm_column_major_to_cbuf_nZ_").concat(suffix).str();
    }
    if (*srcSpace == hivm::AddressSpace::L1 && *dstSpace == hivm::AddressSpace::L0A &&
        srcLayout == TensorLayoutTag::zN && dstLayout == TensorLayoutTag::zN) {
      if (srcElementType != dstElem)
        return {};
      StringRef suffix = copyRuntimeElemSuffix(srcElementType);
      if (suffix.empty())
        return {};
      return Twine("copy_cbuf_zN_to_ca_zN_").concat(suffix).str();
    }
    if (*srcSpace == hivm::AddressSpace::L1 && *dstSpace == hivm::AddressSpace::L0A &&
        srcLayout == TensorLayoutTag::nZ && dstLayout == TensorLayoutTag::zN) {
      if (srcElementType != dstElem)
        return {};
      StringRef suffix = copyRuntimeElemSuffix(srcElementType);
      if (suffix.empty())
        return {};
      return Twine("copy_cbuf_nZ_to_ca_zN_").concat(suffix).str();
    }
    if (*srcSpace == hivm::AddressSpace::L1 && *dstSpace == hivm::AddressSpace::L0B &&
        srcLayout == TensorLayoutTag::zN && dstLayout == TensorLayoutTag::nZ) {
      if (srcElementType != dstElem)
        return {};
      StringRef suffix = copyRuntimeElemSuffix(srcElementType);
      if (suffix.empty())
        return {};
      return Twine("copy_cbuf_zN_to_cb_nZ_").concat(suffix).str();
    }
    if (*srcSpace == hivm::AddressSpace::L1 && *dstSpace == hivm::AddressSpace::L0B &&
        srcLayout == TensorLayoutTag::nZ && dstLayout == TensorLayoutTag::nZ) {
      if (srcElementType != dstElem)
        return {};
      StringRef suffix = copyRuntimeElemSuffix(srcElementType);
      if (suffix.empty())
        return {};
      return Twine("copy_cbuf_nZ_to_cb_nZ_").concat(suffix).str();
    }
    // L0C (fp32 MMAD acc) -> GM row-major: dst may be f32 / f16 / bf16 (narrowing on fixpipe).
    if (*srcSpace == hivm::AddressSpace::L0C && *dstSpace == hivm::AddressSpace::GM &&
        srcLayout == TensorLayoutTag::L0C && dstLayout == TensorLayoutTag::RowMajor) {
      if (srcElementType != "f32")
        return {};
      StringRef suffix = copyRuntimeElemSuffix(dstElem);
      if (suffix.empty())
        return {};
      return Twine("copy_cc_to_gm_row_major_").concat(suffix).str();
    }
    // L0C (fp32 MMAD acc) -> UB row-major: dst may be f32 / f16 / bf16 (narrowing on fixpipe).
    if (*srcSpace == hivm::AddressSpace::L0C && *dstSpace == hivm::AddressSpace::UB &&
        srcLayout == TensorLayoutTag::L0C && dstLayout == TensorLayoutTag::RowMajor) {
      if (srcElementType != "f32")
        return {};
      StringRef suffix = copyRuntimeElemSuffix(dstElem);
      if (suffix.empty())
        return {};
      return Twine("copy_cc_to_ubuf_row_major_").concat(extraDesc).concat("_").concat(suffix).str();
    }
    return {};
  }

  static SmallVector<Value, 8> buildRowMajorCopyPayload(OpBuilder &builder, Location loc,
                                                        const TensorDescriptor &desc) {
    // Row-major runtime payload carries both tile extent and origin layout.
    // dma.cpp builds a TLA tensor from origin shape/stride plus abs coord so
    // nested tile_view copies preserve the root row stride.
    return {
        castValueToI64(builder, loc, desc.shape0),
        castValueToI64(builder, loc, desc.shape1),
        castValueToI64(builder, loc, desc.stride0),
        castValueToI64(builder, loc, desc.stride1),
        castValueToI64(builder, loc, desc.absCoord0),
        castValueToI64(builder, loc, desc.absCoord1),
        castValueToI64(builder, loc, desc.originShape0),
        castValueToI64(builder, loc, desc.originShape1),
    };
  }

  static SmallVector<Value, 12> buildPackedCopyPayload(OpBuilder &builder, Location loc,
                                                       const TensorDescriptor &desc) {
    return {
        castValueToI64(builder, loc, desc.packedShape[0]),
        castValueToI64(builder, loc, desc.packedShape[1]),
        castValueToI64(builder, loc, desc.packedShape[2]),
        castValueToI64(builder, loc, desc.packedShape[3]),
        castValueToI64(builder, loc, desc.packedStride[0]),
        castValueToI64(builder, loc, desc.packedStride[1]),
        castValueToI64(builder, loc, desc.packedStride[2]),
        castValueToI64(builder, loc, desc.packedStride[3]),
        castValueToI64(builder, loc, desc.rowOffset),
        castValueToI64(builder, loc, desc.colOffset),
        castValueToI64(builder, loc, desc.originShape0),
        castValueToI64(builder, loc, desc.originShape1),
    };
  }

  static SmallVector<Value, 20> buildCopyPayloadForRoute(OpBuilder &builder, Location loc,
                                                         const TensorDescriptor &srcDesc,
                                                         const TensorDescriptor &dstDesc) {
    SmallVector<Value, 20> payload;
    auto append = [&](ArrayRef<Value> values) { payload.append(values.begin(), values.end()); };
    if (isLinearLayout(srcDesc.layoutTag))
      append(buildRowMajorCopyPayload(builder, loc, srcDesc));
    else
      append(buildPackedCopyPayload(builder, loc, srcDesc));

    if (isLinearLayout(dstDesc.layoutTag))
      append(buildRowMajorCopyPayload(builder, loc, dstDesc));
    else
      append(buildPackedCopyPayload(builder, loc, dstDesc));
    return payload;
  }

  static bool isAicTemplateRuntimeCall(StringRef name) {
    if (name == "mmad_float_float_float" || name == "mmad_half_half_float" ||
        name == "mmad_bf16_bf16_float")
      return true;
    if (!(name.starts_with("copy_")))
      return false;
    if (!(name.ends_with("_float") || name.ends_with("_half") || name.ends_with("_bf16")))
      return false;
    return name.starts_with("copy_gm_row_major_to_cbuf_zN_") ||
           name.starts_with("copy_gm_column_major_to_cbuf_nZ_") ||
           name.starts_with("copy_cbuf_zN_to_ca_zN_") ||
           name.starts_with("copy_cbuf_nZ_to_ca_zN_") ||
           name.starts_with("copy_cbuf_zN_to_cb_nZ_") ||
           name.starts_with("copy_cbuf_nZ_to_cb_nZ_") ||
           name.starts_with("copy_cc_to_ubuf_row_major_") ||
           name.starts_with("copy_cc_to_gm_row_major_");
  }

  static bool isAivTemplateRuntimeCall(StringRef name) {
    return name == "copy_gm_to_ubuf_1d_float" ||
           name == "copy_ubuf_to_gm_1d_float" ||
           name.starts_with("copy_ubuf_row_major_to_cbuf_zN_");
  }

  static void annotateAicTemplateRuntimeCall(func::FuncOp func) {
    MLIRContext *ctx = func.getContext();
    func->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE),
                  UnitAttr::get(ctx));
    func->setAttr(hivm::TFuncCoreTypeAttr::name,
                  hivm::TFuncCoreTypeAttr::get(ctx, toFuncCoreType(HivmCoreKind::AIC)));
    func->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
  }

  static void annotateAivTemplateRuntimeCall(func::FuncOp func) {
    MLIRContext *ctx = func.getContext();
    func->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE),
                  UnitAttr::get(ctx));
    func->setAttr(hivm::TFuncCoreTypeAttr::name,
                  hivm::TFuncCoreTypeAttr::get(ctx, toFuncCoreType(HivmCoreKind::AIV)));
    func->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
  }

  static func::FuncOp getOrCreateRuntimeCall(ModuleOp module, StringRef name,
                                             ArrayRef<Type> operandTypes,
                                             ArrayRef<Type> resultTypes = {}) {
    if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
      if (isAicTemplateRuntimeCall(name))
        annotateAicTemplateRuntimeCall(existing);
      if (isAivTemplateRuntimeCall(name))
        annotateAivTemplateRuntimeCall(existing);
      return existing;
    }
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());
    auto funcType = builder.getFunctionType(operandTypes, resultTypes);
    auto func = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
    func.setPrivate();
    if (isAicTemplateRuntimeCall(name))
      annotateAicTemplateRuntimeCall(func);
    if (isAivTemplateRuntimeCall(name))
      annotateAivTemplateRuntimeCall(func);
    return func;
  }

  struct LowerTlaMmadPattern : public OpRewritePattern<::tla::MmadOp> {
    LowerTlaMmadPattern(MLIRContext *ctx, ModuleOp module,
                        DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue,
                        SmallVectorImpl<Operation *> &toErase)
        : OpRewritePattern<::tla::MmadOp>(ctx), module(module),
          tensorDescriptorByValue(tensorDescriptorByValue), toErase(toErase) {}

    LogicalResult matchAndRewrite(::tla::MmadOp op, PatternRewriter &rewriter) const override {
      if (op->getNumOperands() < 3)
        return success();

      Value acc = op->getOperand(0);
      Value lhs = op->getOperand(1);
      Value rhs = op->getOperand(2);
      Type accType = acc.getType();
      Type lhsType = lhs.getType();
      Type rhsType = rhs.getType();

      Value initC = op->getOperand(3);
      Value unitFlag = op->getOperand(4);

      auto i1Type = rewriter.getI1Type();
      auto i64Type = rewriter.getI64Type();
      auto i8Type = rewriter.getI8Type();

      Value initCVal = initC;
      Value unitFlagVal = rewriter.create<arith::TruncIOp>(op.getLoc(), i8Type, unitFlag);

      auto lhsInfo = ::tla::decodeTileTypeInfo(lhsType);
      auto rhsInfo = ::tla::decodeTileTypeInfo(rhsType);
      auto accInfo = ::tla::decodeTileTypeInfo(accType);
      if (failed(lhsInfo) || failed(rhsInfo) || failed(accInfo)) {
        op.emitError() << "tla.mmad currently requires structured tla.tensor operand types";
        return failure();
      }
      if (lhsInfo->rank != 2 || rhsInfo->rank != 2 || accInfo->rank != 2) {
        op.emitError() << "tla.mmad currently supports rank-2 tiles only";
        return failure();
      }
      if (lhsInfo->addressSpace != "l0a" || rhsInfo->addressSpace != "l0b" ||
          accInfo->addressSpace != "l0c") {
        op.emitError()
            << "unsupported tla.mmad tile addrspaces; expected acc in l0c, lhs in l0a, rhs in l0b";
        return failure();
      }
      bool supportedF16Route = lhsInfo->elementType == "f16" && rhsInfo->elementType == "f16" &&
                               accInfo->elementType == "f32";
      bool supportedBf16Route = lhsInfo->elementType == "bf16" && rhsInfo->elementType == "bf16" &&
                                accInfo->elementType == "f32";
      bool supportedF32Route = lhsInfo->elementType == "f32" && rhsInfo->elementType == "f32" &&
                               accInfo->elementType == "f32";
      if (!supportedF16Route && !supportedBf16Route && !supportedF32Route) {
        op.emitError() << "unsupported tla.mmad element types; expected f16,f16 -> f32, bf16,bf16 "
                          "-> f32, or f32,f32 -> f32 (L0C accumulator is fp32)";
        return failure();
      }

      auto maybeStaticShapeCheck = [&](int64_t lhsM, int64_t lhsK, int64_t rhsK, int64_t rhsN,
                                       int64_t accM, int64_t accN) -> LogicalResult {
        if (lhsM == ShapedType::kDynamic || lhsK == ShapedType::kDynamic ||
            rhsK == ShapedType::kDynamic || rhsN == ShapedType::kDynamic ||
            accM == ShapedType::kDynamic || accN == ShapedType::kDynamic) {
          return success();
        }
        if (lhsK != rhsK || lhsM != accM || rhsN != accN) {
          op.emitError() << "unsupported tla.mmad tile shape contract; expected lhs(MxK), "
                            "rhs(KxN), acc(MxN)";
          return failure();
        }
        return success();
      };
      if (failed(maybeStaticShapeCheck(lhsInfo->originShapeDims[0], lhsInfo->originShapeDims[1],
                                       rhsInfo->originShapeDims[0], rhsInfo->originShapeDims[1],
                                       accInfo->originShapeDims[0], accInfo->originShapeDims[1])))
        return failure();
      if (accInfo->layoutTag != TensorLayoutTag::L0C || lhsInfo->layoutTag != TensorLayoutTag::zN ||
          rhsInfo->layoutTag != TensorLayoutTag::nZ) {
        op.emitError()
            << "unsupported tla.mmad operand layout; expected acc L0Clayout, lhs zN, rhs nZ";
        return failure();
      }

      auto materializeTensorOperand = [&](Value tensor, Type tensorType) -> FailureOr<Value> {
        return ::tla::materializeTensorOperandAsMemref(rewriter, op.getLoc(), tensor, tensorType,
                                                       toErase);
      };

      FailureOr<Value> lhsMemref = materializeTensorOperand(lhs, lhsType);
      FailureOr<Value> rhsMemref = materializeTensorOperand(rhs, rhsType);
      FailureOr<Value> accMemref = materializeTensorOperand(acc, accType);
      if (failed(lhsMemref) || failed(rhsMemref) || failed(accMemref)) {
        op.emitError() << "failed to bridge tla.mmad operands to memref values";
        return failure();
      }

      // Match tla.copy runtime ABI: pass dynamic strided memrefs to the C stub
      // (same as buildRuntimeMemref in LowerTlaCopyPattern).
      auto toRuntimeMemref = [&](Value v) -> FailureOr<Value> {
        auto baseType = dyn_cast<MemRefType>(v.getType());
        if (!baseType) {
          op.emitError() << "tla.mmad memref operand must have memref type";
          return failure();
        }
        MemRefType runtimeType = ::tla::getDynamicStridedMemrefType(baseType);
        return ::tla::castMemrefToType(rewriter, op.getLoc(), v, runtimeType);
      };
      FailureOr<Value> lhsRuntime = toRuntimeMemref(*lhsMemref);
      FailureOr<Value> rhsRuntime = toRuntimeMemref(*rhsMemref);
      FailureOr<Value> accRuntime = toRuntimeMemref(*accMemref);
      if (failed(lhsRuntime) || failed(rhsRuntime) || failed(accRuntime))
        return failure();

      auto materializeIndexDim = [&](Value tensor, int64_t staticOriginDim, StringRef fieldName,
                                     bool takeSecondDim) -> FailureOr<Value> {
        auto it = tensorDescriptorByValue.find(tensor);
        if (it != tensorDescriptorByValue.end()) {
          Value dim = takeSecondDim ? it->second.originShape1 : it->second.originShape0;
          if (dim && dim.getType().isIndex())
            return dim;
        }
        if (staticOriginDim == ShapedType::kDynamic) {
          op.emitError() << "tla.mmad requires " << fieldName
                         << " from tensor descriptor SSA when type origin_shape is dynamic";
          return failure();
        }
        return rewriter.create<arith::ConstantIndexOp>(op.getLoc(), staticOriginDim).getResult();
      };
      FailureOr<Value> mIndex = materializeIndexDim(lhs, lhsInfo->originShapeDims[0], "M", false);
      FailureOr<Value> kIndex = materializeIndexDim(lhs, lhsInfo->originShapeDims[1], "K", true);
      FailureOr<Value> nIndex = materializeIndexDim(rhs, rhsInfo->originShapeDims[1], "N", true);
      if (failed(mIndex) || failed(kIndex) || failed(nIndex))
        return failure();

      auto castIndexToI64 = [&](Value v) -> Value {
        return rewriter.create<arith::IndexCastOp>(op.getLoc(), i64Type, v).getResult();
      };
      Value mI64 = castIndexToI64(*mIndex);
      Value kI64 = castIndexToI64(*kIndex);
      Value nI64 = castIndexToI64(*nIndex);

      SmallVector<Type, 8> operandTypes = {(*lhsRuntime).getType(),
                                           (*rhsRuntime).getType(),
                                           (*accRuntime).getType(),
                                           i64Type,
                                           i64Type,
                                           i64Type,
                                           i1Type,
                                           i8Type};
      StringRef calleeName = supportedF16Route    ? "mmad_half_half_float"
                             : supportedBf16Route ? "mmad_bf16_bf16_float"
                                                  : "mmad_float_float_float";
      auto callee = getOrCreateRuntimeCall(module, calleeName, operandTypes);
      SmallVector<Value, 8> operands = {
          *lhsRuntime, *rhsRuntime, *accRuntime,           mI64,
          nI64,        kI64,        initCVal,              unitFlagVal
      };
      rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
      toErase.push_back(op.getOperation());
      return success();
    }

  private:
    ModuleOp module;
    DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue;
    SmallVectorImpl<Operation *> &toErase;
  };

  struct LowerTlaCopyPattern : public OpRewritePattern<::tla::CopyOp> {
    LowerTlaCopyPattern(MLIRContext *ctx, ModuleOp module,
                        DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue,
                        SmallVectorImpl<Operation *> &toErase, AllocatorOffsetState *allocatorState,
                        DenseMap<Value, Value> &loweredMemrefByValue)
        : OpRewritePattern<::tla::CopyOp>(ctx), module(module),
          tensorDescriptorByValue(tensorDescriptorByValue), toErase(toErase),
          allocatorState(allocatorState), loweredMemrefByValue(loweredMemrefByValue) {}

    LogicalResult matchAndRewrite(::tla::CopyOp op, PatternRewriter &rewriter) const override {
      if ((op->getNumOperands() != 2 && op->getNumOperands() != 3) || op->getNumResults() != 0) {
        op.emitError() << "expected tla.copy to have 2 or 3 operands and 0 results";
        return failure();
      }

      Value dstTile = op->getOperand(0);
      Value srcTile = op->getOperand(1);
      auto dstIt = tensorDescriptorByValue.find(dstTile);
      auto srcIt = tensorDescriptorByValue.find(srcTile);
      if (dstIt == tensorDescriptorByValue.end()) {
        op.emitError() << "missing descriptor for tla.copy dst tile; expected tile produced by "
                          "tla.tile_view/tla.make_tensor_like in this pass";
        return failure();
      }
      if (srcIt == tensorDescriptorByValue.end()) {
        op.emitError() << "missing descriptor for tla.copy src tile; expected tile produced by "
                          "tla.tile_view/tla.make_tensor_like in this pass";
        return failure();
      }

      const TensorDescriptor &dstDesc = dstIt->second;
      const TensorDescriptor &srcDesc = srcIt->second;
      if (!::tla::validateTensorDescriptorV1(
              op, dstDesc, "malformed descriptor for tla.copy dst tile operand",
              /*requireShapeOperands=*/true)) {
        return failure();
      }
      if (!::tla::validateTensorDescriptorV1(
              op, srcDesc, "malformed descriptor for tla.copy src tile operand",
              /*requireShapeOperands=*/true)) {
        return failure();
      }
      StringRef srcAddrspace = srcDesc.addrspace;
      StringRef dstAddrspace = dstDesc.addrspace;
      std::string src2Dst = std::string(srcDesc.addrspace) + "2" + std::string(dstAddrspace);
      if (srcAddrspace == "l0c") {
        if (op->getNumOperands() != 3) {
          op.emitError() << "expected tla.copy " << src2Dst << " has 3 operands";
          return failure();
        }
      } else if (op->getNumOperands() != 2) {
        op.emitError() << "expected tla.copy " << src2Dst << " has 2 operands";
        return failure();
      }
      bool rankOk = dstDesc.rank == srcDesc.rank;
      bool sameElem = dstDesc.elementType == srcDesc.elementType;
      auto buildRuntimeMemref = [&](const TensorDescriptor &desc) -> FailureOr<Value> {
        FailureOr<Value> baseMemref = ::tla::getOrMaterializeDescriptorBaseMemref(
            rewriter, op.getLoc(), desc, allocatorState, op.getOperation(), loweredMemrefByValue);
        if (failed(baseMemref))
          return failure();
        auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
        if (!baseType)
          return failure();
        MemRefType runtimeType = ::tla::getDynamicStridedMemrefType(baseType);
        return ::tla::castMemrefToType(rewriter, op.getLoc(), *baseMemref,
                                                   runtimeType);
      };

      StringRef extraDesc = "";
      struct L0C2DstInfo {
        uint8_t unitFlag = 0;
        bool relu_enable = false;
        QuantMode quantMode = QuantMode::NO_QUANT;
        L0C2UBMode l0c2UbMode = L0C2UBMode::NO_SPLIT_VEC_0;
        uint8_t subBlockId = 0;
      } l0c2DstInfo;
      if (srcAddrspace == "l0c") {
        auto params = op->getOperand(2);
        auto l0c2DstParamsOp = dyn_cast<::tla::CopyL0C2DstParamsOp>(params.getDefiningOp());
        if (!l0c2DstParamsOp) {
          op.emitError() << "expected tla.CopyL0C2DstParams as third operand";
          return failure();
        }
        l0c2DstInfo.unitFlag = static_cast<uint8_t>(l0c2DstParamsOp.getUnitFlag());
        l0c2DstInfo.relu_enable = l0c2DstParamsOp.getReluEnable();
        l0c2DstInfo.quantMode = l0c2DstParamsOp.getQuantMode().getQuantMode();
        if (dstAddrspace == "ub") {
          l0c2DstInfo.l0c2UbMode = l0c2DstParamsOp.getL0c2ubMode().getL0c2ubMode();
          StringRef splitMode = "nosplit";
          switch (l0c2DstInfo.l0c2UbMode) {
            case L0C2UBMode::NO_SPLIT_VEC_0:
              break;
            case L0C2UBMode::NO_SPLIT_VEC_1:
              l0c2DstInfo.subBlockId = 1;
              splitMode = "nosplit";
              break;
            case L0C2UBMode::SPLIT_M:
              splitMode = "splitm";
              break;
            case L0C2UBMode::SPLIT_N:
              splitMode = "splitn";
              break;
          }
          if ((l0c2DstInfo.l0c2UbMode==L0C2UBMode::SPLIT_M || l0c2DstInfo.l0c2UbMode==L0C2UBMode::SPLIT_N)
              && (srcDesc.elementType != dstDesc.elementType)) {
              op->emitError("When copy l0c to ub with split mode, src and dst type must be same");
              return failure();
          }
          extraDesc = splitMode;
        }
      }

      std::string calleeName = getCopyRouteCallee(
          op.getContext(), srcAddrspace, dstAddrspace, srcDesc.layoutTag, dstDesc.layoutTag,
          srcDesc.elementType, dstDesc.elementType, extraDesc);
      if (!calleeName.empty()) {
        bool l0c2DstNarrow = srcAddrspace == "l0c" && (dstAddrspace == "gm" || dstAddrspace == "ub") &&
                           srcDesc.layoutTag == TensorLayoutTag::L0C &&
                           dstDesc.layoutTag == TensorLayoutTag::RowMajor &&
                           srcDesc.elementType == "f32" &&
                           (dstDesc.elementType == "f16" || dstDesc.elementType == "bf16");
        if (!rankOk || (!sameElem && !l0c2DstNarrow)) {
          op.emitError() << "tla.copy supported route has src/dst descriptor metadata mismatch "
                            "(rank/element type)";
          return failure();
        }

        FailureOr<Value> dstRuntimeMemref = buildRuntimeMemref(dstDesc);
        FailureOr<Value> srcRuntimeMemref = buildRuntimeMemref(srcDesc);
        if (failed(dstRuntimeMemref) || failed(srcRuntimeMemref))
          return failure();
        SmallVector<Value, 20> payload =
            buildCopyPayloadForRoute(rewriter, op.getLoc(), srcDesc, dstDesc);
        SmallVector<Type, 22> operandTypes = {(*srcRuntimeMemref).getType(),
                                              (*dstRuntimeMemref).getType()};
        operandTypes.reserve(2 + payload.size());
        for (Value payloadValue : payload)
          operandTypes.push_back(payloadValue.getType());
        SmallVector<Value, 22> operands = {*srcRuntimeMemref, *dstRuntimeMemref};
        operands.append(payload.begin(), payload.end());
        if (srcAddrspace == "l0c") {
          auto i8Type = rewriter.getI8Type();
          auto unitFlagVal = rewriter.create<arith::ConstantIntOp>(op.getLoc(), l0c2DstInfo.unitFlag, 8);
          operandTypes.push_back(i8Type);
          operands.push_back(unitFlagVal);
          if (dstAddrspace == "ub") {
            auto subBlockIdVal = rewriter.create<arith::ConstantIntOp>(op.getLoc(), l0c2DstInfo.subBlockId, 8);
            operandTypes.push_back(i8Type);
            operands.push_back(subBlockIdVal);
          }
        }
        auto callee = getOrCreateRuntimeCall(module, calleeName, operandTypes);
        rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
        toErase.push_back(op.getOperation());
        return success();
      }

      if (rankOk && sameElem && srcDesc.layoutTag == TensorLayoutTag::RowMajor &&
          dstDesc.layoutTag == TensorLayoutTag::RowMajor) {
        FailureOr<Value> dstRuntimeMemref = buildRuntimeMemref(dstDesc);
        FailureOr<Value> srcRuntimeMemref = buildRuntimeMemref(srcDesc);
        if (failed(dstRuntimeMemref) || failed(srcRuntimeMemref))
          return failure();

        if (srcAddrspace == "gm" && dstAddrspace == "ub") {
          auto srcType = dyn_cast<MemRefType>((*srcRuntimeMemref).getType());
          if (!srcType)
            return failure();
          FailureOr<Value> zeroValue =
              ::tla::makeZeroValue(rewriter, op.getLoc(), srcType.getElementType());
          if (failed(zeroValue))
            return failure();
          auto padModeAttr = rewriter.getAttr<hivm::PadModeAttr>(hivm::PadMode::PadValue);
          Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
          auto load = rewriter.create<hivm::LoadOp>(op.getLoc(), TypeRange{}, *srcRuntimeMemref,
                                                    *dstRuntimeMemref, padModeAttr, *zeroValue,
                                                    zeroIndex);
          load->removeAttr("init_out_buffer");
          load->removeAttr("may_implicit_transpose_with_last_axis");
          toErase.push_back(op.getOperation());
          return success();
        }

        if (srcAddrspace == "ub" && dstAddrspace == "gm") {
          rewriter.create<hivm::StoreOp>(op.getLoc(), TypeRange{}, *srcRuntimeMemref,
                                         *dstRuntimeMemref);
          toErase.push_back(op.getOperation());
          return success();
        }
      }

      op.emitError() << "tla.copy descriptor/layout combination is unsupported: " << srcAddrspace
                     << "(" << ::tla::stringifyTensorLayoutTag(srcDesc.layoutTag)
                     << ") -> " << dstAddrspace << "("
                     << ::tla::stringifyTensorLayoutTag(dstDesc.layoutTag) << ")";
      return failure();
    }

  private:
    ModuleOp module;
    DenseMap<Value, TensorDescriptor> &tensorDescriptorByValue;
    SmallVectorImpl<Operation *> &toErase;
    AllocatorOffsetState *allocatorState;
    DenseMap<Value, Value> &loweredMemrefByValue;
  };

  struct LowerTlaCopyLatePattern : public OpRewritePattern<::tla::CopyOp> {
    LowerTlaCopyLatePattern(MLIRContext *ctx, ModuleOp module,
                            SmallVectorImpl<Operation *> &toErase)
        : OpRewritePattern<::tla::CopyOp>(ctx), module(module), toErase(toErase) {}

    LogicalResult matchAndRewrite(::tla::CopyOp op, PatternRewriter &rewriter) const override {
      if (!op->getBlock() || llvm::is_contained(toErase, op.getOperation()))
        return success();
      if (op->getNumOperands() != 2 || op->getNumResults() != 0)
        return success();

      auto dstInfo = ::tla::decodeTileTypeInfo(op->getOperand(0).getType());
      auto srcInfo = ::tla::decodeTileTypeInfo(op->getOperand(1).getType());
      if (failed(dstInfo) || failed(srcInfo)) {
        op.emitError()
            << "tla.copy late cleanup requires structured tla.tensor metadata on both operands";
        return failure();
      }

      std::string calleeName = getCopyRouteCallee(
          op.getContext(), srcInfo->addressSpace, dstInfo->addressSpace, srcInfo->layoutTag,
          dstInfo->layoutTag, srcInfo->elementType, dstInfo->elementType);
      if (calleeName.empty()) {
        op.emitError() << "tla.copy descriptor/layout combination is unsupported: "
                       << srcInfo->addressSpace << "("
                       << ::tla::stringifyTensorLayoutTag(srcInfo->layoutTag) << ") -> "
                       << dstInfo->addressSpace << "("
                       << ::tla::stringifyTensorLayoutTag(dstInfo->layoutTag) << ")";
        return failure();
      }
      bool sameElem = srcInfo->elementType == dstInfo->elementType;
      bool l0cGmNarrow = srcInfo->addressSpace == "l0c" && dstInfo->addressSpace == "gm" &&
                         srcInfo->layoutTag == TensorLayoutTag::L0C &&
                         dstInfo->layoutTag == TensorLayoutTag::RowMajor &&
                         srcInfo->elementType == "f32" &&
                         (dstInfo->elementType == "f16" || dstInfo->elementType == "bf16");
      if (srcInfo->rank != dstInfo->rank || (!sameElem && !l0cGmNarrow)) {
        op.emitError() << "tla.copy late cleanup found src/dst descriptor metadata mismatch "
                          "(rank/element type)";
        return failure();
      }

      SmallVector<Type, 2> operandTypes = {op->getOperand(1).getType(),
                                           op->getOperand(0).getType()};
      auto callee = getOrCreateRuntimeCall(module, calleeName, operandTypes);
      SmallVector<Value, 2> operands = {op->getOperand(1), op->getOperand(0)};
      rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
      toErase.push_back(op.getOperation());
      return success();
    }

  private:
    ModuleOp module;
    SmallVectorImpl<Operation *> &toErase;
  };

// Flatten a tla.cube region: splice its body into the parent block, erase wrapper.
struct LowerTlaCubePattern : public OpRewritePattern<::tla::CubeOp> {
  using OpRewritePattern<::tla::CubeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::tla::CubeOp op, PatternRewriter &rewriter) const override {
    if (op->getNumRegions() == 0) {
      rewriter.eraseOp(op);
      return success();
    }
    Region &region = op->getRegion(0);
    if (region.empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    Block &body = region.front();
    Block *parentBlock = op->getBlock();
    parentBlock->getOperations().splice(op->getIterator(), body.getOperations(), body.begin(),
                                        body.end());
    rewriter.eraseOp(op);
    return success();
  }
};

class TlaCubeRegionPass : public PassWrapper<TlaCubeRegionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaCubeRegionPass)

  StringRef getArgument() const override { return "tla-cube-region"; }
  StringRef getName() const override { return "TlaCubeRegionPass"; }
  StringRef getDescription() const override {
    return "Lower tla.cube compute ops (tla.copy / tla.mmad) and flatten the cube region.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect,
                    scf::SCFDialect, hivm::HIVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *, 8> toErase;
    ::tla::AllocatorOffsetState allocatorState;
    allocatorState.toErase = &toErase;
    ::tla::TlaTensorMemrefLowering lowering;
    auto &tensorDescriptorByValue = lowering.descriptorByValue;

    // Stage 0/1: derive tensor descriptors.
    if (failed(lowering.deriveDescriptors(module))) {
      signalPassFailure();
      return;
    }

    // Stage 2: descriptor-driven tla.copy lowering (supported v1 routes -> runtime
    // calls; unsupported combinations stay as tla.copy and fail legalization later).
    LowerTlaCopyPattern lowerCopy(&getContext(), module, tensorDescriptorByValue, toErase,
                                  &allocatorState, lowering.loweredMemrefByValue);
    SmallVector<::tla::CopyOp, 16> copyOps;
    module.walk([&](::tla::CopyOp op) { copyOps.push_back(op); });
    bool copyLoweringFailed = false;
    for (::tla::CopyOp op : copyOps) {
      if (!op || !op->getBlock())
        continue;
      PatternRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      if (failed(lowerCopy.matchAndRewrite(op, rewriter)))
        copyLoweringFailed = true;
    }
    if (copyLoweringFailed)
      signalPassFailure();

    // Stage 2.6: lower tile producers to memref.subview (shared) so tla.mmad
    // operands are memref-backed.
    if (failed(lowering.lowerTileProducers(module, &allocatorState, toErase))) {
      signalPassFailure();
      return;
    }

    // Stage 6A: flatten tla.cube region wrappers into their parent block.
    LowerTlaCubePattern lowerCube(&getContext());
    SmallVector<Operation *, 16> cubeOps;
    module.walk<WalkOrder::PostOrder>([&](::tla::CubeOp op) { cubeOps.push_back(op); });
    for (Operation *op : cubeOps) {
      if (!op || !op->getBlock())
        continue;
      PatternRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);
      if (failed(lowerCube.matchAndRewrite(llvm::cast<::tla::CubeOp>(op), rewriter))) {
        signalPassFailure();
        return;
      }
    }

    // Stage 8B: lower tla.mmad.
    LowerTlaMmadPattern lowerMmad(&getContext(), module, tensorDescriptorByValue, toErase);
    SmallVector<Operation *, 16> mmadOps;
    module.walk([&](Operation *op) {
      if (llvm::isa<::tla::MmadOp>(op))
        mmadOps.push_back(op);
    });
    for (Operation *op : mmadOps) {
      if (!op->getBlock())
        continue;
      PatternRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);
      if (auto mmadOp = llvm::dyn_cast<::tla::MmadOp>(op)) {
        if (failed(lowerMmad.matchAndRewrite(mmadOp, rewriter))) {
          signalPassFailure();
          return;
        }
      }
    }

    // Stage 8C: late type-driven tla.copy cleanup (loop-cloned copies).
    LowerTlaCopyLatePattern lowerCopyLate(&getContext(), module, toErase);
    SmallVector<::tla::CopyOp, 16> lateCopyOps;
    module.walk([&](::tla::CopyOp op) { lateCopyOps.push_back(op); });
    bool lateCopyLoweringFailed = false;
    for (::tla::CopyOp op : lateCopyOps) {
      if (!op || !op->getBlock() || llvm::is_contained(toErase, op.getOperation()))
        continue;
      PatternRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      if (failed(lowerCopyLate.matchAndRewrite(op, rewriter)))
        lateCopyLoweringFailed = true;
    }
    if (lateCopyLoweringFailed)
      signalPassFailure();

    // Flush staged erases (lowered copy/mmad ops + consumed ptr bridges + dead
    // tile handles). Remaining dead scaffolding is DCE'd by the cleanup pass.
    DenseSet<Operation *> pendingErase;
    for (Operation *op : toErase)
      if (op && op->getBlock())
        pendingErase.insert(op);
    bool progress = true;
    while (progress && !pendingErase.empty()) {
      progress = false;
      for (Operation *op : toErase) {
        if (!op || !pendingErase.contains(op) || !op->getBlock())
          continue;
        bool hasLiveResultUses = false;
        for (Value result : op->getResults())
          if (!result.use_empty()) {
            hasLiveResultUses = true;
            break;
          }
        if (hasLiveResultUses)
          continue;
        pendingErase.erase(op);
        op->erase();
        progress = true;
      }
    }
    if (!pendingErase.empty()) {
      for (Operation *op : pendingErase)
        op->emitError() << "staged erase failed for '" << op->getName().getStringRef()
                        << "' in tla-cube-region: operation still has live result users";
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTlaCubeRegionPass() { return std::make_unique<TlaCubeRegionPass>(); }

void registerTlaCubeRegionPass() { PassRegistration<TlaCubeRegionPass>(); }

} // namespace tla
