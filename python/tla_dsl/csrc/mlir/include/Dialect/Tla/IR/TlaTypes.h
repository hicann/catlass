#pragma once

#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#define GET_TYPEDEF_CLASSES
#include "tla/Types.h.inc"

namespace tla {
::mlir::LogicalResult getTlaIndexTreeLeaves(::llvm::ArrayRef<int64_t> tree, ::llvm::SmallVectorImpl<int64_t>& leaves);

/// The single source of truth for layouts whose shape and stride each have
/// four physical leaves in tensor descriptors.
inline bool isNZFamilyLayout(::LayoutTag layoutTag)
{
    return layoutTag == ::LayoutTag::zN || layoutTag == ::LayoutTag::nZ || layoutTag == ::LayoutTag::zZ ||
           layoutTag == ::LayoutTag::L0Clayout || layoutTag == ::LayoutTag::zNUnAlign ||
           layoutTag == ::LayoutTag::zZMxScale || layoutTag == ::LayoutTag::nNMxScale;
}

/// GM-side MX scale blocks. Linear as far as the descriptor is concerned -- a
/// contiguous matrix with a pitch -- so they travel with two leaves, and the
/// fractal structure Catlass's copies expect is rebuilt device-side. The tag
/// records which side (A / B) and orientation, which is what selects the copy.

inline bool isMxScaleGmLayout(::LayoutTag layoutTag)
{
    return layoutTag == ::LayoutTag::rowMajorMxScaleA || layoutTag == ::LayoutTag::colMajorMxScaleA ||
           layoutTag == ::LayoutTag::rowMajorMxScaleB || layoutTag == ::LayoutTag::colMajorMxScaleB;
}

inline int64_t getByteSizeOfFixedWidthScalarType(::mlir::Type type)
{
    if (type.isBF16() || type.isF16())
        return 2;
    if (type.isF32())
        return 4;
    if (type.isF64())
        return 8;
    // fp8 cube operand formats (f8E4M3FN / f8E5M2) are byte-sized.
    if (::llvm::isa<::mlir::Float8E4M3FNType, ::mlir::Float8E5M2Type>(type))
        return 1;
    // An e8m0 scale block is one opaque byte per shared exponent.
    if (::llvm::isa<::tla::Float8E8M0Type>(type))
        return 1;
    if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type)) {
        if (intTy.getWidth() % 8 == 0)
            return intTy.getWidth() / 8;
    }
    return 0;
}

/// Width of a scalar element in *bits*.
///
/// The primitive width. getByteSizeOfFixedWidthScalarType above cannot express a
/// sub-byte element and answers 0 for one, which is the right answer for every
/// byte-denominated caller (pointer arithmetic, vector lanes) -- those genuinely
/// cannot address half a byte and should refuse. Callers that convert between an
/// element count and a byte size need this one instead.
inline int64_t getBitSizeOfFixedWidthScalarType(::mlir::Type type)
{
    if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type))
        return intTy.getWidth();
    // The packed fp4 formats are the sub-byte case getByteSize... cannot express.
    if (::llvm::isa<::tla::Float4E2M1Type, ::tla::Float4E1M2Type>(type))
        return 4;
    return getByteSizeOfFixedWidthScalarType(type) * 8;
}

/// True for a packed 4-bit cube float, either encoding.
inline bool isPackedFp4Type(::mlir::Type type)
{
    return ::llvm::isa<::tla::Float4E2M1Type, ::tla::Float4E1M2Type>(type);
}

/// True for an element format this dialect defines itself because MLIR has no
/// builtin for it. These are buffered as i8 at the memref boundary.
inline bool isTlaCustomElementType(::mlir::Type type)
{
    return isPackedFp4Type(type) || ::llvm::isa<::tla::Float8E8M0Type>(type);
}

using coord = ::mlir::Type;
using cross_flag = ::mlir::Type;
using flag = ::mlir::Type;
using index = ::mlir::IndexType;
using memref = ::mlir::Type;
using mutex = ::mlir::Type;
using range = ::mlir::Type;
using shape = ::mlir::Type;
using stride = ::mlir::Type;
using layout = ::mlir::Type;
using tensor = ::mlir::Type;
using tile = ::mlir::Type;
// PtrType is defined in Types.h.inc (TableGen TypeDef).
} // namespace tla
