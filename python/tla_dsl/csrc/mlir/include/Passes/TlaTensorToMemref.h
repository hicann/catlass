#pragma once

// Centralized `!tla.tensor` -> `memref` lowering.
//
// This header is the single owner of the logic that turns a structured
// `!tla.tensor` (and its producers `tla.tile_view` / `tla.make_tensor` /
// `tla.make_tensor_like`, backed by a `tla.ptr`) into a builtin `memref`. Every
// consumer that needs a memref out of a `!tla.tensor` (tla.copy, tla.mmad, the
// vector load/store path, the function ABI) should go through the entry points
// declared here rather than re-deriving the memref itself. See
// tla-memref-cleanup.md for the full plan.
//
// Producer/consumer contract with the ptr-lowering domain: the `!tla.ptr` that
// backs a `tla.make_tensor{,_like}` is produced as a `tla.hivm_memref_as_ptr`
// (wrapping a `hivm.pointer_cast` over a 1-D memref) by the
// `tla-alloc-ptr-to-hivm-pointer-cast` pass. The base-memref materializers in
// this module consume that bridge. `hivmMemref1D` in
// TlaAllocPtrToHivmPointerCastPass stays in the ptr-lowering domain and is
// intentionally NOT part of this header.

#include "Dialect/Tla/IR/TlaOps.h"   // ::tla::TileViewOp
#include "Dialect/Tla/IR/TlaTypes.h" // ::AddressSpace, ::LayoutTag, getTlaIndexTreeLeaves

#include "mlir/Dialect/SCF/IR/SCF.h" // scf::IfOp
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h" // mlir::ModuleOp
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h" // PatternRewriter
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLFunctionalExtras.h" // llvm::function_ref
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <limits>
#include <string>

namespace tla {

/// Staged deletes for bridge casts / control-flow wrappers during lowering.
/// `tla.alloc_ptr` offsets are assigned in `tla-alloc-ptr-to-hivm-pointer-cast`
/// (always run before the region + finalize passes via `buildTlaPipeline`).
/// Defined here (rather than forward-declared) so `TlaTensorMemrefLowering`'s
/// tile-producer methods below have the complete type.
struct AllocatorOffsetState {
  llvm::SmallVectorImpl<mlir::Operation *> *toErase = nullptr;
};

/// Bridge a structured `!tla.tensor<...>` type to a builtin `memref<...,
/// memspace>` type. Canonical tensor-type -> memref-type entry point; forwards
/// to `bridgeTlaFuncTensorType` (PassesCommon.h). Returns failure if the type is
/// not a decodable `!tla.tensor`.
mlir::FailureOr<mlir::MemRefType> bridgeTlaTensorType(mlir::Type tlaTensorType);

/// Tensor layout semantics (descriptor v1). Drives rewrites and validation; it
/// is not serialized in IR.
enum class TensorLayoutTag { Unknown, RowMajor, ColumnMajor, zN, zZ, nZ, L0C };

bool isPackedLayout(TensorLayoutTag layoutTag);
bool isLinearLayout(TensorLayoutTag layoutTag);
llvm::StringRef stringifyTensorLayoutTag(TensorLayoutTag layoutTag);
mlir::FailureOr<TensorLayoutTag> convertTlaLayoutTag(::LayoutTag layoutTag);
mlir::FailureOr<TensorLayoutTag> parseTensorLayoutTagAttr(llvm::StringRef layouttag);
/// Parse the optional `"layouttag"` string attribute off an op (e.g. on
/// tla.tile_view / tla.make_tensor_like) into a TensorLayoutTag; failure if
/// absent or unparseable.
mlir::FailureOr<TensorLayoutTag> getExplicitTensorLayoutTagAttr(mlir::Operation *op);

/// Static metadata decoded from a structured `!tla.tensor` type.
struct TileTypeInfo {
  llvm::SmallVector<int64_t, 4> shapeDims;
  llvm::SmallVector<int64_t, 4> strideDims;
  llvm::SmallVector<int64_t, 4> coordDims;
  llvm::SmallVector<int64_t, 4> originShapeDims;
  std::string addressSpace;
  std::string elementType;
  mlir::Type mlirElementType;
  ::AddressSpace tlaAddressSpace = ::AddressSpace::gm;
  TensorLayoutTag layoutTag = TensorLayoutTag::Unknown;
  int64_t rank = 0;
};

/// Decode `!tla.tensor` metadata from the structured `tla::TlaTensorType`.
mlir::FailureOr<TileTypeInfo> decodeTileTypeInfo(mlir::Type tileType);

/// Raw (non-normalized) decode of a `!tla.tensor` type: the shape / origin /
/// coord / stride index-tree leaves exactly as written (a rank-1 tile stays
/// rank-1, unlike `decodeTileTypeInfo`, which normalizes linear tiles to rank-2),
/// plus the element type, address space, and layout tag. Used by the vector
/// lowering, which branches on the raw tile rank.
struct ParsedTensorInfo {
  llvm::SmallVector<int64_t, 2> shape;
  llvm::SmallVector<int64_t, 2> originShape;
  llvm::SmallVector<int64_t, 2> coord;
  llvm::SmallVector<int64_t, 2> strides;
  ::AddressSpace addressSpace;
  mlir::Type elementType;
  std::string layoutTag;
};

mlir::FailureOr<ParsedTensorInfo> parseTensorInfo(mlir::Type tensorType);

/// SSA-valued tensor descriptor: the materialized coordinates/strides/shapes of
/// a tile, plus the base memref (or ptr) it views.
struct TensorDescriptor {
  mlir::Value base;
  mlir::Type bridgedBaseMemrefType;
  mlir::Value rowOffset;
  mlir::Value colOffset;
  mlir::Value stride0;
  mlir::Value stride1;
  mlir::Value shape0;
  mlir::Value shape1;
  mlir::Value originShape0;
  mlir::Value originShape1;
  mlir::Value absCoord0;
  mlir::Value absCoord1;
  TensorLayoutTag layoutTag = TensorLayoutTag::Unknown;
  std::string addrspace;
  std::string elementType;
  int64_t rank = 0;
  llvm::SmallVector<mlir::Value, 4> packedShape;
  llvm::SmallVector<mlir::Value, 4> packedStride;
};

/// Widen/narrow an integer/index value to i64 (used for runtime-call payloads).
mlir::Value castValueToI64(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value);

/// Cast a memref value to `memrefType` via `memref.cast`; identity if the type
/// already matches, failure if `value` is not a memref.
mlir::FailureOr<mlir::Value> castMemrefToType(mlir::OpBuilder &builder, mlir::Location loc,
                                              mlir::Value value, mlir::MemRefType memrefType);

/// Fully-dynamic shape/stride form of `memrefType` (the runtime-call ABI type).
mlir::MemRefType getDynamicStridedMemrefType(mlir::MemRefType memrefType);

// ---------------------------------------------------------------------------
// Descriptor builders (pure: materialize a TensorDescriptor's SSA index leaves
// from decoded TileTypeInfo; no per-block constant cache).
// ---------------------------------------------------------------------------

/// Structural invariants for a descriptor-v1 (rank-2, index-typed leaves, packed
/// leaves present for packed layouts). Emits `errorMessage` on `op` and returns
/// false on violation.
bool validateTensorDescriptorV1(mlir::Operation *op, const TensorDescriptor &desc,
                                llvm::StringRef errorMessage, bool requireShapeOperands);

mlir::Value makeIndexConstant(mlir::OpBuilder &builder, mlir::Location loc, int64_t value);
mlir::FailureOr<mlir::Value> makeZeroValue(mlir::OpBuilder &builder, mlir::Location loc,
                                           mlir::Type type);

mlir::FailureOr<mlir::Value> makeStaticTensorInfoIndex(mlir::OpBuilder &builder, mlir::Operation *op,
                                                       int64_t value, llvm::StringRef fieldName);
mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
materializeStaticTensorInfoIndices(mlir::OpBuilder &builder, mlir::Operation *op,
                                   llvm::ArrayRef<int64_t> values, llvm::StringRef fieldName);
mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
materializeTensorInfoIndicesWithDynamicValues(mlir::OpBuilder &builder, mlir::Operation *op,
                                              llvm::ArrayRef<int64_t> values,
                                              llvm::ArrayRef<mlir::Value> dynamicValues,
                                              llvm::StringRef fieldName);
mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
packRank2DynamicMetadataLeaves(mlir::Operation *op, llvm::ArrayRef<int64_t> leafDims,
                               mlir::Value axis0, mlir::Value axis1, llvm::StringRef fieldName);

/// Build a full TensorDescriptor from decoded `info` and a base memref value.
/// Dynamic coord / origin-shape leaves are filled from the optional SSA lists
/// (in leaf order).
mlir::FailureOr<TensorDescriptor>
buildTensorDescriptorFromTensorInfo(mlir::OpBuilder &builder, mlir::Operation *op, mlir::Value base,
                                    mlir::Type bridgedBaseMemrefType, const TileTypeInfo &info,
                                    llvm::ArrayRef<mlir::Value> coordDynamicValues = {},
                                    llvm::ArrayRef<mlir::Value> originShapeDynamicValues = {});

/// Factory for cached index/integer constants, keyed per-scope. `bits == 0`
/// means an `index`-typed constant. Injected into descriptor algebra so the
/// caller owns constant deduplication.
using ConstantFactory = llvm::function_ref<mlir::Value(mlir::Operation *anchor, int64_t value,
                                                       unsigned bits)>;

/// Per-scope cache of arith constants (index-typed when `bits == 0`, else an
/// `i<bits>` constant). Keyed by the enclosing tla.func / func.func / module
/// entry block, so a constant is materialized once per function body and reused.
class IndexConstantCache {
public:
  mlir::Value get(mlir::Operation *anchor, int64_t value, unsigned bits = 0);

private:
  struct Key {
    int64_t value;
    unsigned bits;
    bool operator==(const Key &other) const {
      return value == other.value && bits == other.bits;
    }
  };
  struct KeyInfo {
    static inline Key getEmptyKey() { return {std::numeric_limits<int64_t>::min(), 0}; }
    static inline Key getTombstoneKey() { return {std::numeric_limits<int64_t>::min() + 1, 0}; }
    static unsigned getHashValue(const Key &key) { return llvm::hash_combine(key.value, key.bits); }
    static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
  };
  llvm::DenseMap<mlir::Block *, llvm::DenseMap<Key, mlir::Value, KeyInfo>> byScope;
};

/// Owns the per-pass state for `!tla.tensor` -> `memref` lowering: the constant
/// cache, the descriptor (metadata) map, and the lower-once base-memref cache. A
/// single instance is threaded through the tile-producing walk (which populates
/// `descriptorByValue`) and the tla.copy / tla.mmad / tile-view consumers (which
/// read it).
class TlaTensorMemrefLowering {
public:
  /// Index/integer constants, deduplicated per function body.
  IndexConstantCache constants;
  /// Descriptor for each `!tla.tensor` SSA value, keyed by the value.
  llvm::DenseMap<mlir::Value, TensorDescriptor> descriptorByValue;
  /// The single lower-once cache shared by BOTH lowering models: SSA value -> the
  /// memref it was already materialized to, so a tensor is never lowered twice.
  ///   - descriptor model (cube pass): keyed by `TensorDescriptor::base` (root
  ///     storage), populated by `getOrMaterializeDescriptorBaseMemref`, so every
  ///     tile of the same root reuses one materialized base memref;
  ///   - raw-parse model (vector pass): keyed by the tile tensor value, populated
  ///     by `materializeBaseMemref` / `materializeCopySubview*` (the former
  ///     `loweredMemrefByValue`).
  /// The two models never run in the same pass, so their keys never collide.
  llvm::DenseMap<mlir::Value, mlir::Value> loweredMemrefByValue;

  /// Seed root (function-argument) tensor descriptors, then walk tile-producing
  /// ops (tla.tile_view / tla.make_tensor / tla.make_tensor_like) in pre-order,
  /// deriving a TensorDescriptor for each result into `descriptorByValue`.
  /// Returns failure if any producer is malformed (diagnostics emitted on the
  /// offending op). Idempotent enough to run in more than one pass (materialized
  /// index ops are CSE'd downstream).
  mlir::LogicalResult deriveDescriptors(mlir::ModuleOp module);

  /// Lower a single tile producer (tla.tile_view / tla.make_tensor{,_like}) to a
  /// `memref.subview` of its materialized base + an UnrealizedConversionCast back
  /// to the tile's `!tla.tensor` type (so tile-typed consumers stay valid until
  /// the finalize/cleanup pass). Requires the op's descriptor in
  /// `descriptorByValue`; moves the descriptor onto the new cast result.
  mlir::LogicalResult lowerTileProducerToSubview(mlir::Operation *op,
                                                 mlir::PatternRewriter &rewriter,
                                                 AllocatorOffsetState *allocatorState);

  /// Lower ALL tile producers to subviews (stages transitively-dead producers
  /// into `toErase` first, then lowers the rest, then erases dead
  /// make_tensor{,_like} handles). The shared entry point the region passes use
  /// to turn tiles into the memref parameters their compute ops consume.
  mlir::LogicalResult lowerTileProducers(mlir::ModuleOp module,
                                         AllocatorOffsetState *allocatorState,
                                         llvm::SmallVectorImpl<mlir::Operation *> &toErase);
};

/// Build a `tla.tile_view` result descriptor for a `!tla.tensor` source from a
/// parent tile descriptor and the tile's (row, col, sh0, sh1) SSA operands.
/// Follows TLA `TileViewImpl`: `abs = parent.abs + tileCoord`,
/// `origin_i = min(tileShape_i, parent.origin_i - tileCoord_i)`; linear layouts
/// may carry dynamic shape (from sh0/sh1) and dynamic stride (from the parent),
/// packed layouts derive dynamic leaves via ceil-div or parent inheritance.
mlir::FailureOr<TensorDescriptor> buildTileViewResultDescriptorFromParent(
    mlir::Operation *op, mlir::Value base, mlir::MemRefType bridgedBaseType,
    const TileTypeInfo &info, const TensorDescriptor &parent, mlir::Value row, mlir::Value col,
    mlir::Value sh0, mlir::Value sh1, ConstantFactory getConstant);

// ---------------------------------------------------------------------------
// Base-memref materialization: turn a descriptor's `base` (a memref, or a
// `!tla.ptr` bridged as `tla.hivm_memref_as_ptr`, possibly behind an
// scf.if / UnrealizedConversionCast) into a concrete memref of the wanted type.
// Bridge/ptr wrappers consumed along the way are queued into `toErase`.
// ---------------------------------------------------------------------------

void pushStagedErase(llvm::SmallVectorImpl<mlir::Operation *> *toErase, mlir::Operation *op);
void pushStagedErase(llvm::SmallVectorImpl<mlir::Operation *> &toErase, mlir::Operation *op);

/// True iff every result user of `op` is already staged for erasure.
bool hasOnlyStagedResultUsers(mlir::Operation *op, llvm::ArrayRef<mlir::Operation *> stagedErase);
/// Transitively stage tile producers (tla.tile_view / tla.make_tensor{,_like})
/// whose only users are staged-for-erase.
void stageDeadTileProducers(mlir::ModuleOp module,
                            llvm::SmallVectorImpl<mlir::Operation *> &toErase);

/// If `ptrValue` came from `tla-alloc-ptr-to-hivm-pointer-cast`, return the
/// static 1-D element count of its backing memref.
mlir::FailureOr<int64_t> getStatic1DElementCountFromHivmPtrBridge(mlir::Value ptrValue);

/// Wrap `desc.base` in an UnrealizedConversionCast to the bridged memref type
/// (last-resort materialization when no ptr/memref unwrap applies).
mlir::Value createFallbackBaseMemrefCast(mlir::OpBuilder &builder, mlir::Location loc,
                                         const TensorDescriptor &desc);

/// Flatten a (possibly multi-dimensional) contiguous memref to a rank-1 dynamic
/// view over the same element storage (identity if already rank-1). Used to bridge
/// a kernel-arg memref into the rank-1 form `tla.hivm_memref_as_ptr` expects when
/// resolving a `tla.tensor_ptr` source.
mlir::Value flattenMemrefTo1D(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value memref);

/// Materialize a `!tla.ptr` value (expected: `tla.hivm_memref_as_ptr`, or a
/// `tla.ptr_add` / `tla.tensor_ptr` chain over one) as a memref of `memrefType`,
/// applying any accumulated `ptr_add` element offset. Emits a diagnostic on
/// `diagnosticOp` on failure.
mlir::FailureOr<mlir::Value> materializePtrValueAsMemref(mlir::OpBuilder &builder, mlir::Location loc,
                                                         mlir::Value ptrValue,
                                                         mlir::MemRefType memrefType,
                                                         mlir::Operation *diagnosticOp);

/// Rewrite an `scf.if` yielding a `!tla.ptr` into one yielding `memrefType`.
mlir::FailureOr<mlir::Value> materializePtrIfAsMemref(mlir::OpBuilder &builder, mlir::Location loc,
                                                      mlir::scf::IfOp ifOp,
                                                      mlir::MemRefType memrefType,
                                                      AllocatorOffsetState &allocatorState,
                                                      mlir::Operation *diagnosticOp);

/// Materialize `desc.base` as a concrete memref of `desc.bridgedBaseMemrefType`,
/// unwrapping ptr/if/cast wrappers (queued into `allocatorState->toErase`).
mlir::FailureOr<mlir::Value> materializeDescriptorBaseMemref(mlir::OpBuilder &builder,
                                                             mlir::Location loc,
                                                             const TensorDescriptor &desc,
                                                             AllocatorOffsetState *allocatorState,
                                                             mlir::Operation *diagnosticOp);

/// Materialize a tla-tensor-typed value as a memref suitable for a runtime-call
/// operand. If `tensor` is defined by an UnrealizedConversionCast from a memref,
/// return that source memref and stage the cast for erasure; otherwise bridge the
/// tensor type and wrap `tensor` in a fresh cast. Used by tla.mmad; the caller
/// then casts the result to the runtime dynamic-strided ABI type.
mlir::FailureOr<mlir::Value>
materializeTensorOperandAsMemref(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value tensor,
                                 mlir::Type tensorType,
                                 llvm::SmallVectorImpl<mlir::Operation *> &toErase);

/// Lower-once wrapper over `materializeDescriptorBaseMemref`. On the first call
/// for a given `desc.base`, the base memref is materialized at a point that
/// dominates every use of `desc.base` (right after its defining op, or the entry
/// of its block for a block argument) and cached in `baseMemrefCache`. Later
/// calls with the same base return the cached memref (cast to the wanted type at
/// the caller). Because the cached value dominates all uses of `desc.base`, it
/// dominates every consumer, so the reuse is SSA-safe.
mlir::FailureOr<mlir::Value>
getOrMaterializeDescriptorBaseMemref(mlir::OpBuilder &builder, mlir::Location loc,
                                     const TensorDescriptor &desc,
                                     AllocatorOffsetState *allocatorState,
                                     mlir::Operation *diagnosticOp,
                                     llvm::DenseMap<mlir::Value, mlir::Value> &baseMemrefCache);

// ---------------------------------------------------------------------------
// Vector tile memref materialization (TlaVectorRegionPass). Vector-specific
// policy — 256-byte lane memrefs, rank-1<->2 reshape, the per-copy handoff cache
// — layered on the shared decode/bridge above.
// ---------------------------------------------------------------------------

mlir::FailureOr<int64_t> getStaticNumElements(llvm::ArrayRef<int64_t> shape);
mlir::FailureOr<int64_t> getElementByteWidth(mlir::Type elementType);
mlir::FailureOr<int64_t> getVectorLaneCount(mlir::Type elementType);
mlir::FailureOr<int64_t> getVectorLanesForMemref(mlir::MemRefType type);
bool isSupportedVectorTile(mlir::MemRefType type);

/// Bridge `tensor`'s `!tla.tensor` type to a memref (Value form of the bridge).
mlir::FailureOr<mlir::MemRefType> getBridgedTensorMemrefType(mlir::Value tensor);

/// 256-byte vector-tile memref type for a helper argument (origin-clamped).
mlir::FailureOr<mlir::MemRefType>
getVectorHelperMemrefType(mlir::Value tensor,
                          llvm::DenseMap<mlir::Value, mlir::Value> *loweredMemrefByValue = nullptr);

/// Preferred memref type for a vector-helper operand (flat make_tensor_like or bridged).
mlir::FailureOr<mlir::MemRefType> getVectorHelperArgMemrefType(mlir::Value operand);

/// Adapt a memref to `expectedType`: rank-1<->2 reshape when the element storage
/// matches, else a plain memref.cast.
mlir::FailureOr<mlir::Value> castMemrefToExpected(mlir::PatternRewriter &rewriter,
                                                  mlir::Location loc, mlir::Value value,
                                                  mlir::MemRefType expectedType);

mlir::FailureOr<mlir::Value> materializeSingleCoordIndex(mlir::PatternRewriter &rewriter,
                                                         mlir::Location loc, mlir::Value coord);
mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
materializeRank2CoordIndices(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value coord);

/// Materialize a `tla.tile_view` result as a (vector-tile or bridged) memref.
mlir::FailureOr<mlir::Value>
materializeTileViewMemref(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value tensor,
                          ::tla::TileViewOp tileView, bool useVectorHelperType = false,
                          llvm::DenseMap<mlir::Value, mlir::Value> *loweredMemrefByValue = nullptr);

/// Flat 1-D memref backing a `tla.make_tensor{,_like}` (unwrap the ptr bridge).
mlir::FailureOr<mlir::Value> getMakeTensorLikeFlatMemref(mlir::Value tensor);

/// Materialize the base memref a tensor/tile views into (handoff cache, ptr
/// unwrap, tile-view recursion).
mlir::FailureOr<mlir::Value>
materializeBaseMemref(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value tensor,
                      llvm::DenseMap<mlir::Value, mlir::Value> *loweredMemrefByValue = nullptr);

mlir::FailureOr<mlir::Value>
materializeCopySubview1D(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value tensor,
                         llvm::DenseMap<mlir::Value, mlir::Value> *loweredMemrefByValue = nullptr);
mlir::FailureOr<mlir::Value>
materializeCopySubviewRank2(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value tensor,
                            llvm::DenseMap<mlir::Value, mlir::Value> *loweredMemrefByValue = nullptr,
                            llvm::ArrayRef<int64_t> concreteShape = {});
/// Try the 1-D copy subview, then the rank-2 form.
mlir::FailureOr<mlir::Value>
materializeCopySubview(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value tensor,
                       llvm::DenseMap<mlir::Value, mlir::Value> *loweredMemrefByValue = nullptr,
                       llvm::ArrayRef<int64_t> concreteShape = {});

} // namespace tla
