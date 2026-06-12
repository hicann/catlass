add_library(TlaPasses
  PassRegistry.cpp
  TlaFuncToHaccPass.cpp
  TlaLowerToHivmPass.cpp
  TlaAllocPtrToHivmPointerCastPass.cpp
  TlaLowerToStdPass.cpp
  TlaSyncToHivmPass.cpp
  SupportTritonPass.cpp
  ConvertTlaToVectorPass.cpp
  ../Dialect/Tla/IR/TlaDialect.cpp
  ../Dialect/Tla/IR/TlaAttrs.cpp
  ../Dialect/Tla/IR/TlaOps.cpp
)
add_dependencies(TlaPasses TlaTblgen)
set_target_properties(TlaPasses PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(TlaPasses PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/tblgen
  ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(TlaPasses PUBLIC -Wl,--start-group ${TLA_DSL_HIVM_REQUIRED_LIBS} -Wl,--end-group)
target_link_libraries(TlaPasses
  PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRDialect
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRDestinationStyleOpInterface
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRMeshDialect
    MLIRSideEffectInterfaces
    MLIRTensorDialect
    MLIRViewLikeInterface
    MLIRMemRefUtils
    MLIRDLTIDialect
    MLIRLLVMCommonConversion
    MLIRConvertToLLVMInterface
    MLIRLLVMDialect
    MLIRSCFToControlFlow
)
