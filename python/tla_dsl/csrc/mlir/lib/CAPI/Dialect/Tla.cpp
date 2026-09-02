//===- Tla.cpp - C API for the CATLASS TLA dialect -----------------------===//
//
// Part of the CATLASS DSL project.
//
//===----------------------------------------------------------------------===//

#include "catlass-c/Dialect/Tla.h"

#include "Dialect/Tla/IR/TlaDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Tla, tla, tla::TlaDialect)
