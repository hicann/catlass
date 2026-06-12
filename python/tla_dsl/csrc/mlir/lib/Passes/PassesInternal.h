#pragma once

#include "mlir/Pass/Pass.h"

namespace tla {

void registerTlaFuncToHaccPass();
void registerTlaLowerToHivmPass();
void registerConvertTlaToVectorPass();
void registerTlaSyncToHivmPass();
void registerTlaAllocPtrToHivmPointerCastPass();
void registerTlaLowerToStdPass();

} // namespace tla
