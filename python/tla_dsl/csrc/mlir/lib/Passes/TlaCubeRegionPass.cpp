#include "Dialect/Tla/IR/TlaAttrs.h"
#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaTensorToMemref.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Dominance.h"

// tla-cube-region: lowers the cube (AIC) compute ops (tla.copy / tla.mmad) via
// the shared !tla.tensor->memref lowering (TlaTensorMemrefLowering), then flattens
// the tla.cube region. Runs after tla-vector-region, before tla-finalize-memref.
//
// Flow: collect materialized descriptors, then lower tla.copy (descriptor +
// payload driven) and tla.mmad, both reconstructing tile memrefs directly from
// tla.tensor_desc. Descriptors are available for every copy, so copy lowering is
// a single descriptor-driven path. Finalize DCEs dead scaffolding and unrealized
// casts.

namespace tla {
namespace {

// tla.mmad_mx is only meaningful when its L0 operands were produced by an MX
// load. The hardware's mad_mx takes no scale operand: the e8m0 block is attached
// by the L1->L0 copy and read from a side buffer addressed off the L0 tile. That
// makes MX-ness a property of how the tile was *written*, which the element type
// and layout tag cannot express -- an ordinary fp8 tla.copy leaves an operand
// that looks identical to an MX-loaded one.
//
// The frontend has both ops and the user picks; nothing here infers one from the
// other.

// The bc symbol suffix is always the C++ type the wrapper instantiates. The DSL
// spells the packed fp4 formats !tla.f4e2m1 / !tla.f4e1m2; the C++ types are
// float4_e2m1x2_t / float4_e1m2x2_t. Kept here so every caller that builds an
// fp4 callee name uses the same mapping.
//
// The encoding is the element type -- there is no separate attribute to consult
// and therefore nothing that can disagree with it. A tile is !tla.f4e2m1 or
// !tla.f4e1m2 from the moment it is allocated, so the load and the matmul read
// the same answer by construction; ::tla::isPackedFp4Type is the shared test.
static StringRef fp4CppType(Type elementType)
{
    return ::llvm::isa<::tla::Float4E1M2Type>(elementType) ? "float4_e1m2x2_t" : "float4_e2m1x2_t";
}

// CTRL[51] selects the mmad M/N compute-direction priority
static constexpr unsigned int ComputeOrderBit = 51;

// CTRL[46] enables the mmad HF32 rounding mode
static constexpr unsigned int HF32ModeBit = 46;

// CTRL[47] selects the mmad HF32 rounding mode:
// 0 = NEAREST_EVEN (hardware default), 1 = NEAREST_ZERO
static constexpr unsigned int HF32TransModeBit = 47;

// Shared by tla.mmad and tla.mmad_mx: the two ops differ only in whether the
// L0 operands carry an attached e8m0 scale block, which changes the callee's
// L0 element types (mx_fp8_*) and therefore selects mad_mx over mad. The
// operand list, shape contract and descriptor handling are identical.
template <typename MmadOpTy>
struct LowerTlaMmadPatternImpl : public OpRewritePattern<MmadOpTy> {
    static constexpr bool kIsMx = std::is_same_v<MmadOpTy, ::tla::MmadMxOp>;

    LowerTlaMmadPatternImpl(
        MLIRContext* ctx, DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue,
        SmallVectorImpl<Operation*>& toErase, DenseMap<Value, Value>& loweredMemrefByValue,
        bool funcLevelComputeOrderSet, bool funcLevelHF32Set)
        : OpRewritePattern<MmadOpTy>(ctx),
          tensorDescriptorByValue(tensorDescriptorByValue),
          toErase(toErase),
          loweredMemrefByValue(loweredMemrefByValue),
          funcLevelComputeOrderSet(funcLevelComputeOrderSet),
          funcLevelHF32Set(funcLevelHF32Set)
    {}

    LogicalResult matchAndRewrite(MmadOpTy op, PatternRewriter& rewriter) const override
    {
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

        auto lhsInfo = ::tla::decodeTensorTypeInfo(lhsType);
        auto rhsInfo = ::tla::decodeTensorTypeInfo(rhsType);
        auto accInfo = ::tla::decodeTensorTypeInfo(accType);
        if (failed(lhsInfo) || failed(rhsInfo) || failed(accInfo)) {
            op.emitError() << "tla.mmad currently requires structured tla.tensor operand types";
            return failure();
        }
        if (lhsInfo->rank != 2 || rhsInfo->rank != 2 || accInfo->rank != 2) {
            op.emitError() << "tla.mmad currently supports rank-2 tiles only";
            return failure();
        }
        if (lhsInfo->addressSpace != "l0a" || rhsInfo->addressSpace != "l0b" || accInfo->addressSpace != "l0c") {
            op.emitError() << "unsupported tla.mmad tile addrspaces; expected acc in l0c, lhs in l0a, rhs in l0b";
            return failure();
        }
        bool supportedF16Route =
            lhsInfo->elementType.isF16() && rhsInfo->elementType.isF16() && accInfo->elementType.isF32();
        bool supportedBf16Route =
            lhsInfo->elementType.isBF16() && rhsInfo->elementType.isBF16() && accInfo->elementType.isF32();
        bool supportedF32Route =
            lhsInfo->elementType.isF32() && rhsInfo->elementType.isF32() && accInfo->elementType.isF32();
        // Integer route: unlike the float routes, the L0C accumulator is i32.
        bool supportedI8Route = lhsInfo->elementType.isSignlessInteger(8) &&
                                rhsInfo->elementType.isSignlessInteger(8) && accInfo->elementType.isSignlessInteger(32);
        // fp8 routes support both MLIR f8E4M3FN and f8E5M2 types. The two formats
        // mix freely on the cube, so all four operand pairings are legal; each
        // accumulates into fp32.
        constexpr bool isMx = kIsMx;
        auto isFp8 = [](Type elem) { return elem.isFloat8E4M3FN() || elem.isFloat8E5M2(); };
        bool bothFp8 = isFp8(lhsInfo->elementType) && isFp8(rhsInfo->elementType) && accInfo->elementType.isF32();
        // Plain fp8 (tla.mmad): the L0 tiles carry no scale block, so this lands on
        // the same `mad` intrinsic as the float routes.
        bool supportedFp8Route = !isMx && bothFp8;
        // MX fp8 (tla.mmad_mx): the L0 tiles were loaded with their scale block
        // attached (tla.copy with a scale operand), so the callee instantiates the
        // mx_fp8_* L0 types and reaches mad_mx.
        bool isMxFp8 = isMx && bothFp8;
        // MX fp4 tiles state both width and encoding in the element type
        // (!tla.f4e2m1 / !tla.f4e1m2). They use the ordinary zN / nZ layouts like
        // every other operand, and the two sides need not share an encoding -- the
        // cube has a mad_mx for every pairing.
        bool isMxFp4 = isMx && ::tla::isPackedFp4Type(lhsInfo->elementType) &&
                       ::tla::isPackedFp4Type(rhsInfo->elementType) && accInfo->elementType.isF32();
        if (isMx && !isMxFp8 && !isMxFp4) {
            op.emitError() << "tla.mmad_mx requires f8E4M3FN/f8E5M2 operands or fp4-tagged "
                              "tiles, and an f32 accumulator";
            return failure();
        }
        if (!isMx && !supportedF16Route && !supportedBf16Route && !supportedF32Route && !supportedI8Route &&
            !supportedFp8Route) {
            // A packed fp4 tile reaching a plain tla.mmad means it was loaded
            // without a scale: there is no non-microscaling fp4 route on the cube,
            // so say that rather than listing the element types it is not.
            if (::tla::isPackedFp4Type(lhsInfo->elementType) || ::tla::isPackedFp4Type(rhsInfo->elementType)) {
                op.emitError() << "packed fp4 tla.mmad operands must be loaded by tla.copy(..., scale=...); "
                                  "the cube has no non-microscaling fp4 route";
                return failure();
            }
            op.emitError() << "unsupported tla.mmad element types; expected f16,f16 -> f32, bf16,bf16 "
                              "-> f32, f32,f32 -> f32, any f8E4M3FN/f8E5M2 pair -> f32 (fp32 L0C "
                              "accumulator), or i8,i8 -> i32 (i32 L0C accumulator)";
            return failure();
        }

        auto maybeStaticShapeCheck = [&](int64_t lhsM, int64_t lhsK, int64_t rhsK, int64_t rhsN, int64_t accM,
                                         int64_t accN) -> LogicalResult {
            if (lhsM == ShapedType::kDynamic || lhsK == ShapedType::kDynamic || rhsK == ShapedType::kDynamic ||
                rhsN == ShapedType::kDynamic || accM == ShapedType::kDynamic || accN == ShapedType::kDynamic) {
                return success();
            }
            if (lhsK != rhsK || lhsM != accM || rhsN != accN) {
                op.emitError() << "unsupported tla.mmad tile shape contract; expected lhs(MxK), "
                                  "rhs(KxN), acc(MxN)";
                return failure();
            }
            return success();
        };
        if (failed(maybeStaticShapeCheck(
                lhsInfo->originShapeDims[0], lhsInfo->originShapeDims[1], rhsInfo->originShapeDims[0],
                rhsInfo->originShapeDims[1], accInfo->originShapeDims[0], accInfo->originShapeDims[1])))
            return failure();
        TensorLayoutTag expectedLhsTag = TensorLayoutTag::zN;
        TensorLayoutTag expectedRhsTag = TensorLayoutTag::nZ;
        if (accInfo->layoutTag != TensorLayoutTag::L0C || lhsInfo->layoutTag != expectedLhsTag ||
            rhsInfo->layoutTag != expectedRhsTag) {
            op.emitError() << "unsupported tla.mmad operand layout; expected acc L0Clayout, lhs zN, rhs nZ";
            return failure();
        }

        // Materialize each tile operand's memref directly from its tla.tensor_desc
        // descriptor (shared with the tla.copy path).
        auto materializeTensorOperand = [&](Value tensor) -> FailureOr<Value> {
            auto it = tensorDescriptorByValue.find(tensor);
            if (it == tensorDescriptorByValue.end()) {
                op.emitError() << "missing descriptor for tla.mmad tile operand";
                return failure();
            }
            return ::tla::materializeTileMemrefFromDescriptor(
                rewriter, op.getLoc(), it->second, op.getOperation(), loweredMemrefByValue);
        };

        FailureOr<Value> lhsMemref = materializeTensorOperand(lhs);
        FailureOr<Value> rhsMemref = materializeTensorOperand(rhs);
        FailureOr<Value> accMemref = materializeTensorOperand(acc);
        if (failed(lhsMemref) || failed(rhsMemref) || failed(accMemref)) {
            op.emitError() << "failed to bridge tla.mmad operands to memref values";
            return failure();
        }

        // Match the tla.copy runtime ABI: pass dynamic strided memrefs to the C stub
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
                Value dim = it->second.originShape[takeSecondDim ? 1 : 0];
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
        // FP8 needs a per-operand name because the two formats can be mixed.
        // Two spellings, because the two families name their BC symbols
        // differently and each is forced.
        //
        // Plain fp8 symbols are named after the C++ element type the wrapper
        // instantiates -- the CANN builtin fp8_e4m3fn_t / fp8_e5m2_t -- which is a
        // valid identifier and so can be pasted straight into the symbol.
        //
        // The MX operand types cannot: AscendC::mx_fp8_e4m3_t contains a `::`.
        // mmad.cpp therefore aliases them, and the alias follows the MLIR element
        // type lowercased (f8E4M3FN -> f8e4m3fn) with an `mx` prefix, so the MX
        // symbol is built as "mx" + this tag. fp4 uses the same rule via its own
        // element type (!tla.f4e2m1 / !tla.f4e1m2).
        auto fp8Tag = [](Type elem) -> StringRef { return elem.isFloat8E4M3FN() ? "fp8_e4m3fn_t" : "fp8_e5m2_t"; };
        auto mxFp8Tag = [](Type elem) -> StringRef {
            return elem.isFloat8E4M3FN() ? "mx_fp8_e4m3_t" : "mx_fp8_e5m2_t";
        };
        std::string fp8CalleeStorage;
        if (isMxFp4) {
            // Per operand, not once: the two sides carry their own encodings and
            // the bc registers every pairing.
            fp8CalleeStorage =
                ("mmad_" + fp4CppType(lhsInfo->elementType) + "_" + fp4CppType(rhsInfo->elementType) + "_float").str();
        } else if (isMxFp8) {
            fp8CalleeStorage =
                ("mmad_" + mxFp8Tag(lhsInfo->elementType) + "_" + mxFp8Tag(rhsInfo->elementType) + "_float").str();
        } else if (supportedFp8Route) {
            fp8CalleeStorage =
                ("mmad_" + fp8Tag(lhsInfo->elementType) + "_" + fp8Tag(rhsInfo->elementType) + "_float").str();
        }
        StringRef calleeName = supportedF16Route                         ? "mmad_half_half_float" :
                               supportedBf16Route                        ? "mmad_bf16_bf16_float" :
                               supportedI8Route                          ? "mmad_int8_int8_int32" :
                               (isMxFp8 || isMxFp4 || supportedFp8Route) ? StringRef(fp8CalleeStorage) :
                                                                           "mmad_float_float_float";
        auto callee = ::tla::getOrCreateRuntimeCall(op->template getParentOfType<ModuleOp>(), calleeName, operandTypes);
        SmallVector<Value, 8> operands = {*lhsRuntime, *rhsRuntime, *accRuntime, mI64,
                                          nI64,        kI64,        initCVal,    unitFlagVal};
        if (!funcLevelComputeOrderSet) {
            auto computeOrderAttr = op->template getAttrOfType<::tla::ComputeOrderAttr>("compute_order");
            bool isNFirst = computeOrderAttr.getValue() == ComputeOrder::N_FIRST;
            rewriter.create<hivm::SetCtrlOp>(op.getLoc(), isNFirst, ComputeOrderBit);
        }
        // HF32 is an fp32 rounding mode, so it exists only on tla.mmad; tla.mmad_mx
        // carries no hf32_mode attribute and this is compiled out for it.
        if constexpr (!kIsMx) {
            if (!funcLevelHF32Set) {
                auto modeAttr = op->template getAttrOfType<::tla::HF32ModeAttr>("hf32_mode");
                HF32Mode mode = modeAttr.getValue();
                bool enableHF32 = mode != HF32Mode::HF32_DISABLE;
                bool nearestZero = mode == HF32Mode::HF32_NEAREST_ZERO;
                rewriter.create<hivm::SetCtrlOp>(op.getLoc(), enableHF32, HF32ModeBit);
                rewriter.create<hivm::SetCtrlOp>(op.getLoc(), nearestZero, HF32TransModeBit);
            }
        }
        rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
        toErase.push_back(op.getOperation());
        return success();
    }

private:
    DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue;
    SmallVectorImpl<Operation*>& toErase;
    DenseMap<Value, Value>& loweredMemrefByValue;
    bool funcLevelComputeOrderSet;
    bool funcLevelHF32Set;
};

using LowerTlaMmadPattern = LowerTlaMmadPatternImpl<::tla::MmadOp>;
using LowerTlaMmadMxPattern = LowerTlaMmadPatternImpl<::tla::MmadMxOp>;

// tla.copy_mx: L1 -> L0A/L0B with the operand's e8m0 scale block attached. The
// scale is consumed by this load, which is also what selects the mx_fp8_* L0
// element type and therefore the later mad_mx.
struct LowerTlaCopyMxPattern : public OpRewritePattern<::tla::CopyMxOp> {
    LowerTlaCopyMxPattern(
        MLIRContext* ctx, DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue,
        SmallVectorImpl<Operation*>& toErase, DenseMap<Value, Value>& loweredMemrefByValue)
        : OpRewritePattern<::tla::CopyMxOp>(ctx),
          tensorDescriptorByValue(tensorDescriptorByValue),
          toErase(toErase),
          loweredMemrefByValue(loweredMemrefByValue)
    {}

    LogicalResult matchAndRewrite(::tla::CopyMxOp op, PatternRewriter& rewriter) const override
    {
        Value dstTile = op.getDst();
        Value srcTile = op.getSrc();
        Value scaleTile = op.getScale();

        auto lookup = [&](Value v, StringRef what) -> const TensorDescriptor* {
            auto it = tensorDescriptorByValue.find(v);
            if (it == tensorDescriptorByValue.end()) {
                op.emitError() << "missing descriptor for tla.copy_mx " << what << " tile";
                return nullptr;
            }
            return &it->second;
        };
        const TensorDescriptor* dstDesc = lookup(dstTile, "dst");
        const TensorDescriptor* srcDesc = lookup(srcTile, "src");
        const TensorDescriptor* scaleDesc = lookup(scaleTile, "scale");
        if (!dstDesc || !srcDesc || !scaleDesc)
            return failure();

        auto buildRuntimeMemref = [&](const TensorDescriptor& desc) -> FailureOr<Value> {
            FailureOr<Value> base = ::tla::materializeTileMemrefFromDescriptor(
                rewriter, op.getLoc(), desc, op.getOperation(), loweredMemrefByValue);
            if (failed(base))
                return failure();
            auto baseType = dyn_cast<MemRefType>((*base).getType());
            if (!baseType)
                return failure();
            return ::tla::castMemrefToType(rewriter, op.getLoc(), *base, ::tla::getDynamicStridedMemrefType(baseType));
        };

        FailureOr<Value> srcMemref = buildRuntimeMemref(*srcDesc);
        FailureOr<Value> dstMemref = buildRuntimeMemref(*dstDesc);
        FailureOr<Value> scaleMemref = buildRuntimeMemref(*scaleDesc);
        if (failed(srcMemref) || failed(dstMemref) || failed(scaleMemref))
            return failure();

        bool toL0A = dstDesc->addrspace == "l0a";
        // The element type states both the 4-bit width and the encoding, so the
        // callee suffix comes straight off the tile.
        bool isFp4 = ::tla::isPackedFp4Type(srcDesc->elementType);
        std::string calleeName;
        if (isFp4) {
            calleeName = (Twine("copy_mx_l1_") + (srcDesc->layoutTag == TensorLayoutTag::nZ ? "nZ" : "zN") +
                          (toL0A ? "_to_l0a_zN_" : "_to_l0b_nZ_") + fp4CppType(srcDesc->elementType))
                             .str();
        } else {
            // The suffix is the *L1* element type the wrapper instantiates; the
            // mx_fp8_* type appears only on the L0 side of that same wrapper.
            StringRef fmt = srcDesc->elementType.isFloat8E4M3FN() ? "fp8_e4m3fn_t" :
                            srcDesc->elementType.isFloat8E5M2()   ? "fp8_e5m2_t" :
                                                                    "";
            if (fmt.empty()) {
                op.emitError() << "unsupported tla.copy_mx element type " << srcDesc->elementType
                               << "; expected f8E4M3FN, f8E5M2, f4e2m1, or f4e1m2";
                return failure();
            }
            // Either source layout, on either side: the L1 tile keeps whatever
            // orientation its GM operand had, and the transposing pairing is a
            // real Catlass specialization.
            calleeName = (Twine("copy_mx_l1_") + (srcDesc->layoutTag == TensorLayoutTag::nZ ? "nZ" : "zN") +
                          (toL0A ? "_to_l0a_zN_" : "_to_l0b_nZ_") + fmt)
                             .str();
        }

        SmallVector<Value, 40> payload;
        auto appendDesc = [&](const TensorDescriptor& desc) {
            SmallVector<Value, 12> fields = ::tla::buildCopyPayloadForDescriptor(rewriter, op.getLoc(), desc);
            payload.append(fields.begin(), fields.end());
        };
        appendDesc(*srcDesc);
        appendDesc(*dstDesc);
        appendDesc(*scaleDesc);

        SmallVector<Type, 40> operandTypes = {(*srcMemref).getType(), (*dstMemref).getType(), (*scaleMemref).getType()};
        for (Value v : payload)
            operandTypes.push_back(v.getType());
        SmallVector<Value, 40> operands = {*srcMemref, *dstMemref, *scaleMemref};
        operands.append(payload.begin(), payload.end());

        auto callee = ::tla::getOrCreateRuntimeCall(op->getParentOfType<ModuleOp>(), calleeName, operandTypes);
        rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
        toErase.push_back(op.getOperation());
        return success();
    }

private:
    DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue;
    SmallVectorImpl<Operation*>& toErase;
    DenseMap<Value, Value>& loweredMemrefByValue;
};

struct LowerTlaCopyPattern : public OpRewritePattern<::tla::CopyOp> {
    LowerTlaCopyPattern(
        MLIRContext* ctx, DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue,
        SmallVectorImpl<Operation*>& toErase, DenseMap<Value, Value>& loweredMemrefByValue)
        : OpRewritePattern<::tla::CopyOp>(ctx),
          tensorDescriptorByValue(tensorDescriptorByValue),
          toErase(toErase),
          loweredMemrefByValue(loweredMemrefByValue)
    {}

    LogicalResult matchAndRewrite(::tla::CopyOp op, PatternRewriter& rewriter) const override
    {
        if ((op->getNumOperands() != 2 && op->getNumOperands() != 3) || op->getNumResults() != 0) {
            op.emitError() << "expected tla.copy to have 2 or 3 operands and 0 results";
            return failure();
        }

        Value dstTile = op->getOperand(0);
        Value srcTile = op->getOperand(1);
        auto dstIt = tensorDescriptorByValue.find(dstTile);
        auto srcIt = tensorDescriptorByValue.find(srcTile);
        if (dstIt == tensorDescriptorByValue.end()) {
            op.emitError() << "missing descriptor for tla.copy dst tile; expected a tla.tensor_desc "
                              "operand materialized by tla-lower-tensor-desc";
            return failure();
        }
        if (srcIt == tensorDescriptorByValue.end()) {
            op.emitError() << "missing descriptor for tla.copy src tile; expected a tla.tensor_desc "
                              "operand materialized by tla-lower-tensor-desc";
            return failure();
        }

        const TensorDescriptor& dstDesc = dstIt->second;
        const TensorDescriptor& srcDesc = srcIt->second;
        if (!::tla::validateTensorDescriptor(op, dstDesc, "malformed descriptor for tla.copy dst tile operand")) {
            return failure();
        }
        if (!::tla::validateTensorDescriptor(op, srcDesc, "malformed descriptor for tla.copy src tile operand")) {
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
        auto buildRuntimeMemref = [&](const TensorDescriptor& desc) -> FailureOr<Value> {
            FailureOr<Value> baseMemref = ::tla::getOrMaterializeDescriptorBaseMemref(
                rewriter, op.getLoc(), desc, op.getOperation(), loweredMemrefByValue);
            if (failed(baseMemref))
                return failure();
            auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
            if (!baseType)
                return failure();
            MemRefType runtimeType = ::tla::getDynamicStridedMemrefType(baseType);
            return ::tla::castMemrefToType(rewriter, op.getLoc(), *baseMemref, runtimeType);
        };

        std::string extraDesc;
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
                if ((l0c2DstInfo.l0c2UbMode == L0C2UBMode::SPLIT_M || l0c2DstInfo.l0c2UbMode == L0C2UBMode::SPLIT_N) &&
                    (srcDesc.elementType != dstDesc.elementType)) {
                    op->emitError("When copy l0c to ub with split mode, src and dst type must be same");
                    return failure();
                }
                extraDesc = splitMode;
            }
        }

        std::string calleeName = ::tla::getCopyRouteCallee(
            op.getContext(), srcAddrspace, dstAddrspace, srcDesc.layoutTag, dstDesc.layoutTag, srcDesc.elementType,
            dstDesc.elementType, extraDesc);
        if (!calleeName.empty()) {
            FailureOr<Value> dstRuntimeMemref = buildRuntimeMemref(dstDesc);
            FailureOr<Value> srcRuntimeMemref = buildRuntimeMemref(srcDesc);
            if (failed(dstRuntimeMemref) || failed(srcRuntimeMemref))
                return failure();
            SmallVector<Value, 24> payload = ::tla::buildCopyPayloadForRoute(rewriter, op.getLoc(), srcDesc, dstDesc);
            SmallVector<Type, 22> operandTypes = {(*srcRuntimeMemref).getType(), (*dstRuntimeMemref).getType()};
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
            auto callee = ::tla::getOrCreateRuntimeCall(op->getParentOfType<ModuleOp>(), calleeName, operandTypes);

            // Enclose `copy` with atomic add and atomic none
            auto getAtomicKind = [](AtomicMode mode) -> hivm::AtomicKind {
                switch (mode) {
                    case AtomicMode::add:
                        return hivm::AtomicKind::ADD;
                    // For further extension, add other atomic mode case
                    default:
                        return hivm::AtomicKind::NONE;
                }
            };

            auto atomicModeAttr = op->getAttrOfType<::tla::AtomicModeAttr>("atomic_mode");
            Type dstType = cast<MemRefType>((*dstRuntimeMemref).getType()).getElementType();
            bool _enable_atomic = atomicModeAttr && atomicModeAttr.getAtomicMode() != AtomicMode::none;
            if (_enable_atomic) {
                if (atomicModeAttr.getAtomicMode() != AtomicMode::add) {
                    op.emitError() << "currently only atomic add is supported";
                    return failure();
                }
                auto modeAttr =
                    hivm::AtomicKindAttr::get(rewriter.getContext(), getAtomicKind(atomicModeAttr.getAtomicMode()));
                rewriter.create<hivm::SetAtomicOp>(op.getLoc(), modeAttr, mlir::TypeAttr::get(dstType));
            }
            rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
            if (_enable_atomic) {
                auto modeAttr = hivm::AtomicKindAttr::get(rewriter.getContext(), hivm::AtomicKind::NONE);
                rewriter.create<hivm::SetAtomicOp>(op.getLoc(), modeAttr, mlir::TypeAttr::get(dstType));
            }
            toErase.push_back(op.getOperation());
            return success();
        }

        // GM<->UB row-major copies are owned by tla-vector-region's LowerCopyPattern
        // (UB is vector-core memory); the cube pass only lowers the cube-side routes
        // above (L1 / L0A / L0B / L0C).

        op.emitError() << "tla.copy descriptor/layout combination is unsupported: " << srcAddrspace << "("
                       << ::tla::stringifyTensorLayoutTag(srcDesc.layoutTag) << ") -> " << dstAddrspace << "("
                       << ::tla::stringifyTensorLayoutTag(dstDesc.layoutTag) << ")";
        return failure();
    }

private:
    DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue;
    SmallVectorImpl<Operation*>& toErase;
    DenseMap<Value, Value>& loweredMemrefByValue;
};

// Flatten a tla.cube region by splicing its body into the parent block.
struct LowerTlaCubePattern : public OpRewritePattern<::tla::CubeOp> {
    using OpRewritePattern<::tla::CubeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::tla::CubeOp op, PatternRewriter& rewriter) const override
    {
        if (op->getNumRegions() == 0 || op->getRegion(0).empty()) {
            rewriter.eraseOp(op);
            return success();
        }
        Block& body = op->getRegion(0).front();
        Block* parentBlock = op->getBlock();
        parentBlock->getOperations().splice(op->getIterator(), body.getOperations(), body.begin(), body.end());
        rewriter.eraseOp(op);
        return success();
    }
};

class TlaCubeRegionPass : public PassWrapper<TlaCubeRegionPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaCubeRegionPass)

    StringRef getArgument() const override
    {
        return "tla-cube-region";
    }
    StringRef getName() const override
    {
        return "TlaCubeRegionPass";
    }
    StringRef getDescription() const override
    {
        return "Lower tla.cube compute ops (tla.copy / tla.mmad) and flatten the cube region.";
    }
    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect, scf::SCFDialect, hivm::HIVMDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();

        // Drive one function at a time so the cube lowering only touches the cube
        // (AIC) function's ops, not a sibling vector (AIV) function in a mixed kernel.
        // Mirrors tla-vector-region, which iterates the module's functions and filters
        // by core type. Snapshot the functions up front: lowering appends runtime-call
        // declarations to the module, which must not be fed back through the loop.
        SmallVector<func::FuncOp, 4> funcOps(module.getOps<func::FuncOp>());
        for (func::FuncOp funcOp : funcOps) {
            if (funcOp.isDeclaration())
                continue;
            // Skip the generated vector_region helpers: they hold lowered AVE ops and
            // no cube work.
            if (funcOp->hasAttr(kHivmVectorFunctionAttrName))
                continue;
            // Only AIC (and not-yet-split MIX) functions hold cube work. Pure vector
            // (AIV) functions have no tla.cube / tla.copy / tla.mmad to lower.
            std::optional<hivm::TFuncCoreType> coreType = getFunctionCoreType(funcOp.getOperation());
            if (coreType != hivm::TFuncCoreType::AIC && coreType != hivm::TFuncCoreType::MIX)
                continue;
            if (failed(runOnCubeFunction(funcOp))) {
                signalPassFailure();
                return;
            }
        }
    }

    // Lower all cube compute ops within a single cube (AIC/MIX) function. The
    // lowering state (descriptors, memref cache, staged erases) is fresh per
    // function, matching tla-vector-region's per-function handoff. Only `root` (the
    // function) is threaded; the ModuleOp needed for runtime-symbol insertion is
    // derived on demand from the op being rewritten (getParentOfType<ModuleOp>()).
    LogicalResult runOnCubeFunction(func::FuncOp funcOp)
    {
        Operation* root = funcOp.getOperation();
        SmallVector<Operation*, 8> toErase;
        ::tla::TlaTensorMemrefLowering lowering;
        auto& tensorDescriptorByValue = lowering.descriptorByValue;
        // Set on a lowering failure that does not abort the remaining work; reported
        // once at the end.
        bool passFailed = false;

        // Read the descriptors materialized by tla-lower-tensor-desc. Cube lowering
        // must not reconstruct metadata from raw tensor producer chains.
        if (failed(::tla::collectMaterializedTensorDescriptors(funcOp, tensorDescriptorByValue)))
            return failure();

        // tla.tensor_ptr / tla.ptr_add were already folded into the inttoptr byte
        // address by tla-lower-ptr (run before tla-lower-tensor-desc), so each
        // tensor_desc.base here is the raw inttoptr boundary and the copy / subview
        // materialization resolves it straight to a memref.

        // Descriptor-driven tla.copy lowering (supported v1 routes -> runtime calls;
        // unsupported combinations stay as tla.copy and fail legalization later).
        LowerTlaCopyPattern lowerCopy(&getContext(), tensorDescriptorByValue, toErase, lowering.loweredMemrefByValue);
        SmallVector<::tla::CopyOp, 16> copyOps;
        root->walk([&](::tla::CopyOp op) { copyOps.push_back(op); });
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
            passFailed = true;

        // tla.copy_mx runs on the same descriptors, before the cube region flattens.
        LowerTlaCopyMxPattern lowerCopyMx(
            &getContext(), tensorDescriptorByValue, toErase, lowering.loweredMemrefByValue);
        SmallVector<::tla::CopyMxOp, 8> copyMxOps;
        root->walk([&](::tla::CopyMxOp op) { copyMxOps.push_back(op); });
        for (::tla::CopyMxOp op : copyMxOps) {
            if (!op || !op->getBlock())
                continue;
            PatternRewriter rewriter(op.getContext());
            rewriter.setInsertionPoint(op);
            if (failed(lowerCopyMx.matchAndRewrite(op, rewriter)))
                passFailed = true;
        }

        LowerTlaCubePattern lowerCube(&getContext());
        SmallVector<::tla::CubeOp, 4> cubeOps;
        root->walk<WalkOrder::PostOrder>([&](::tla::CubeOp op) { cubeOps.push_back(op); });
        for (::tla::CubeOp op : cubeOps) {
            if (!op || !op->getBlock())
                continue;
            PatternRewriter rewriter(op.getContext());
            rewriter.setInsertionPoint(op);
            if (failed(lowerCube.matchAndRewrite(op, rewriter)))
                return failure();
        }

        // Lower tla.mmad.
        // CTRL[51] (mmad M/N compute-direction priority) is global and persists once
        // set, so when every mmad in this function agrees on compute_order it is set
        // once at the function entry. If the function mixes M_FIRST/N_FIRST the
        // per-mmad path in LowerTlaMmadPattern is used instead.
        std::optional<ComputeOrder> funcLevelComputeOrder;
        bool computeOrderConflict = false;
        // Both mmad flavours drive the same CTRL[51] bit, so they are surveyed
        // together -- a function mixing tla.mmad and tla.mmad_mx must still agree.
        root->walk([&](Operation* op) {
            if (!llvm::isa<::tla::MmadOp, ::tla::MmadMxOp>(op))
                return;
            auto attr = op->getAttrOfType<::tla::ComputeOrderAttr>("compute_order");
            ComputeOrder order = attr.getValue();
            if (funcLevelComputeOrder && *funcLevelComputeOrder != order)
                computeOrderConflict = true;
            else if (!funcLevelComputeOrder)
                funcLevelComputeOrder = order;
        });
        bool funcLevelComputeOrderSet = false;
        if (funcLevelComputeOrder && !computeOrderConflict) {
            Block& entry = funcOp.getBody().front();
            PatternRewriter builder(funcOp.getContext());
            builder.setInsertionPointToStart(&entry);
            bool isNFirst = *funcLevelComputeOrder == ComputeOrder::N_FIRST;
            builder.create<hivm::SetCtrlOp>(funcOp.getLoc(), isNFirst, ComputeOrderBit);
            funcLevelComputeOrderSet = true;
        }

        // CTRL[46]/CTRL[47] (mmad HF32 rounding mode) are global and persist once
        // set, so when every mmad in this function agrees on hf32_mode they are set
        // once at the function entry. If the function mixes values the per-mmad path
        // in LowerTlaMmadPattern is used instead.
        std::optional<HF32Mode> funcLevelHF32Mode;
        bool HF32Conflict = false;
        root->walk([&](::tla::MmadOp op) {
            auto attr = op->getAttrOfType<::tla::HF32ModeAttr>("hf32_mode");
            HF32Mode mode = attr.getValue();
            if (funcLevelHF32Mode && *funcLevelHF32Mode != mode)
                HF32Conflict = true;
            else if (!funcLevelHF32Mode)
                funcLevelHF32Mode = mode;
        });
        bool funcLevelHF32Set = false;
        if (funcLevelHF32Mode && !HF32Conflict) {
            Block& entry = funcOp.getBody().front();
            PatternRewriter builder(funcOp.getContext());
            builder.setInsertionPointToStart(&entry);
            HF32Mode mode = *funcLevelHF32Mode;
            bool enableHF32 = mode != HF32Mode::HF32_DISABLE;
            bool nearestZero = mode == HF32Mode::HF32_NEAREST_ZERO;
            builder.create<hivm::SetCtrlOp>(funcOp.getLoc(), enableHF32, HF32ModeBit);
            builder.create<hivm::SetCtrlOp>(funcOp.getLoc(), nearestZero, HF32TransModeBit);
            funcLevelHF32Set = true;
        }
        LowerTlaMmadPattern lowerMmad(
            &getContext(), tensorDescriptorByValue, toErase, lowering.loweredMemrefByValue, funcLevelComputeOrderSet,
            funcLevelHF32Set);
        LowerTlaMmadMxPattern lowerMmadMx(
            &getContext(), tensorDescriptorByValue, toErase, lowering.loweredMemrefByValue, funcLevelComputeOrderSet,
            funcLevelHF32Set);
        SmallVector<Operation*, 16> mmadOps;
        root->walk([&](Operation* op) {
            if (llvm::isa<::tla::MmadOp, ::tla::MmadMxOp>(op))
                mmadOps.push_back(op);
        });
        for (Operation* op : mmadOps) {
            if (!op->getBlock())
                continue;
            PatternRewriter rewriter(op->getContext());
            rewriter.setInsertionPoint(op);
            if (auto mmadOp = llvm::dyn_cast<::tla::MmadOp>(op)) {
                if (failed(lowerMmad.matchAndRewrite(mmadOp, rewriter))) {
                    return failure();
                }
            } else if (auto mmadMxOp = llvm::dyn_cast<::tla::MmadMxOp>(op)) {
                if (failed(lowerMmadMx.matchAndRewrite(mmadMxOp, rewriter))) {
                    return failure();
                }
            }
        }

        // The tla.tensor_desc ops are dead (tla.copy / tla.mmad were lowered off their
        // descriptors, not their values), but they still hold their `base` scaffolding
        // (the inttoptr boundary) which the base-memref materialization staged for
        // erasure. Stage the dead tensor_descs so the flush erases them first; that
        // scaffolding cannot be erased while a live tensor_desc still references it.
        ::tla::stageDeadTensorDescriptors(root, toErase);

        // Flush staged erases: the lowered tla.copy / tla.mmad ops (which
        // tla-finalize-memref marks illegal, so they must be erased here), the dead
        // tla.tensor_desc ops, and the ptr bridges / scaffolding consumed while
        // materializing their tile memrefs.
        DenseSet<Operation*> pendingErase;
        for (Operation* op : toErase)
            if (op && op->getBlock())
                pendingErase.insert(op);
        bool progress = true;
        while (progress && !pendingErase.empty()) {
            progress = false;
            for (Operation* op : toErase) {
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
            for (Operation* op : pendingErase)
                op->emitError() << "staged erase failed for '" << op->getName().getStringRef()
                                << "' in tla-cube-region: operation still has live result users";
            return failure();
        }
        return passFailed ? failure() : success();
    }
};

} // namespace

std::unique_ptr<Pass> createTlaCubeRegionPass()
{
    return std::make_unique<TlaCubeRegionPass>();
}

void registerTlaCubeRegionPass()
{
    PassRegistration<TlaCubeRegionPass>();
}

} // namespace tla
