add_library(TlaCompilePipeline CompilePipeline.cpp)
set_target_properties(TlaCompilePipeline PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(TlaCompilePipeline PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/tblgen
  ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(TlaCompilePipeline
  PUBLIC
    TlaPasses
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRLLVMDialect
    MLIRTargetLLVMIRExport
    MLIRToLLVMIRTranslationRegistration
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRMemRefToLLVM
    MLIRConvertToLLVMPass
    MLIRReconcileUnrealizedCasts
)

# In-process Tla TypeDef construction/accessors for Python frontend descriptors.
Python3_add_library(_tla_type_bridge_native MODULE TypeBridge.cpp WITH_SOABI)
target_include_directories(_tla_type_bridge_native PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/tblgen
  ${Python3_INCLUDE_DIRS}
  ${CATLASS_DSL_PYBIND11_INCLUDE_DIRS}
)
target_link_libraries(_tla_type_bridge_native PRIVATE
  Python3::Module
  ${CATLASS_DSL_MLIR_PYTHON_CAPI_LIBRARY}
)
target_compile_options(_tla_type_bridge_native PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>
)
if(pybind11_FOUND)
  target_link_libraries(_tla_type_bridge_native PRIVATE pybind11::headers)
endif()
add_dependencies(_tla_type_bridge_native TlaCompilePipeline TlaPasses)
set_target_properties(_tla_type_bridge_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CATLASS_DSL_PYTHON_PACKAGE_OUTPUT_DIR}
  BUILD_RPATH "\$ORIGIN/_mlir/_mlir_libs"
  INSTALL_RPATH "\$ORIGIN/_mlir/_mlir_libs"
  BUILD_WITH_INSTALL_RPATH ON
)
