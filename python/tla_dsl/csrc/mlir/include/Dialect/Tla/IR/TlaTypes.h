#pragma once

#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#define GET_TYPEDEF_CLASSES
#include "tla/Types.h.inc"

namespace tla {
::mlir::LogicalResult getTlaIndexTreeLeaves(::llvm::ArrayRef<int64_t> tree,
                                            ::llvm::SmallVectorImpl<int64_t> &leaves);

using coord = ::mlir::Type;
using cross_flag = ::mlir::Type;
using flag = ::mlir::Type;
using index = ::mlir::IndexType;
using memref = ::mlir::Type;
using range = ::mlir::Type;
using shape = ::mlir::Type;
using stride = ::mlir::Type;
using layout = ::mlir::Type;
using tensor = ::mlir::Type;
using tile = ::mlir::Type;
// PtrType is defined in Types.h.inc (TableGen TypeDef).
using value = ::mlir::Type;
} // namespace tla
