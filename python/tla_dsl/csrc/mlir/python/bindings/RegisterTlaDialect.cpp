//===- RegisterTlaDialect.cpp - Register the CATLASS TLA dialect --------===//
//
// Private `catlass._mlir` package extension. BiShengIR dialects used by the
// lowering pipeline are loaded by the C++ compiler bridge and do not need
// Python-side registration.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir-c/IR.h"

#include "catlass-c/Dialect/Tla.h"

PYBIND11_MODULE(_tlaRegisterDialect, m)
{
    m.doc() = "CATLASS TLA dialect registration for catlass._mlir";

    // MLIR core dialects are registered by the upstream RegisterEverything
    // extension embedded in the same aggregate CAPI.
    m.def("register_dialects", [](MlirDialectRegistry registry) {
        mlirDialectHandleInsertDialect(mlirGetDialectHandle__tla__(), registry);
    });
}
