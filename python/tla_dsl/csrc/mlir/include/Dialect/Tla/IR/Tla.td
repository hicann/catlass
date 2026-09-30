//===----------------------------------------------------------------------===//
// The Tla dialect.
//
// This file is the single entry point mlir-tblgen is run on; it only pulls in
// the pieces below, in dependency order. Add new definitions to the matching
// file, and list any new file here *and* in TLA_TD_SOURCES in
// csrc/mlir/CMakeLists.txt.
//===----------------------------------------------------------------------===//

include "Dialect/Tla/IR/TlaBase.td"
include "Dialect/Tla/IR/TlaEnums.td"
include "Dialect/Tla/IR/TlaTypes.td"
include "Dialect/Tla/IR/TlaRegionOps.td"
include "Dialect/Tla/IR/TlaArchOps.td"
include "Dialect/Tla/IR/TlaScalarOps.td"
include "Dialect/Tla/IR/TlaStructureOps.td"
include "Dialect/Tla/IR/TlaTensorOps.td"
include "Dialect/Tla/IR/TlaMemoryOps.td"
include "Dialect/Tla/IR/TlaSyncOps.td"
include "Dialect/Tla/IR/TlaCubeOps.td"
include "Dialect/Tla/IR/TlaSimdOps.td"
include "Dialect/Tla/IR/TlaSimtOps.td"
