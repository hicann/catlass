#pragma once

#include "mlir/Pass/Pass.h"

namespace tla {

void registerTlaFuncToHaccPass();
void registerTlaLowerToHivmPass();
void registerSupportTritonPass();
void registerConvertTlaToVectorPass();
void registerTlaSyncToHivmPass();
void registerTlaAllocPtrToHivmPointerCastPass();
void registerTlaLowerToStdPass();

} // namespace tla
