#pragma once

#include "Dialect/Tla/IR/TlaAttrs.h"

#include "mlir/Support/LLVM.h"

namespace tla {

inline mlir::FailureOr<int64_t>
mapTlaAddressSpaceToMlirMemRefSpaceValue(AddressSpace addressSpace) {
  switch (addressSpace) {
  case AddressSpace::generic:
    return 0;
  case AddressSpace::gm:
    return 1;
  case AddressSpace::l1:
    return 2;
  case AddressSpace::l0a:
    return 3;
  case AddressSpace::l0b:
    return 4;
  case AddressSpace::l0c:
    return 5;
  case AddressSpace::ub:
    return 6;
  }
  return mlir::failure();
}

} // namespace tla
