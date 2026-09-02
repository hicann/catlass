//===- Tla.h - C API for the CATLASS TLA dialect -----------------*- C -*-===//
//
// Part of the CATLASS DSL project.
//
//===----------------------------------------------------------------------===//

#ifndef CATLASS_C_DIALECT_TLA_H
#define CATLASS_C_DIALECT_TLA_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Tla, tla);

#ifdef __cplusplus
}
#endif

#endif // CATLASS_C_DIALECT_TLA_H
