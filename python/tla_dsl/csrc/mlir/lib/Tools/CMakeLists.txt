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
    MLIRSCFDialect
    MLIRMemRefDialect
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
  ${TLA_DSL_PYBIND11_INCLUDE_DIRS}
)
target_link_libraries(_tla_type_bridge_native PRIVATE
  Python3::Module
  $<TARGET_FILE:TlaCompilePipeline>
  $<TARGET_FILE:TlaPasses>
  ${TLA_DSL_HIVM_REQUIRED_LIBS}
  ${TLA_DSL_MLIR_PYTHON_CAPI_LIBRARY}
)
if(pybind11_FOUND)
  target_link_libraries(_tla_type_bridge_native PRIVATE pybind11::headers)
endif()
add_dependencies(_tla_type_bridge_native TlaCompilePipeline TlaPasses)
set_target_properties(_tla_type_bridge_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${TLA_DSL_PYTHON_PACKAGE_OUTPUT_DIR}
  BUILD_RPATH "${TLA_DSL_MLIR_PYTHON_CAPI_DIR};${TLA_DSL_HIVM_LIB_DIR}"
)

if(TLA_DSL_BUILD_RUNTIME_WRAPPER)
  add_library(tla_dsl_runtime_wrapper SHARED RuntimeWrapper.cpp)
  target_include_directories(tla_dsl_runtime_wrapper PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/experiment
    ${ASCEND_HOME_PATH}/include/experiment/msprof/toolchain
    ${ASCEND_HOME_PATH}/../ascend-toolkit/latest/x86_64-linux/pkg_inc
    ${ASCEND_HOME_PATH}/../ascend-toolkit/latest/aarch64-linux/pkg_inc
    ${ASCEND_HOME_PATH}/../ascend-toolkit/latest/x86_64-linux/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/../ascend-toolkit/latest/aarch64-linux/pkg_inc/runtime
  )
  target_link_directories(tla_dsl_runtime_wrapper PRIVATE ${ASCEND_HOME_PATH}/lib64)
  target_link_libraries(tla_dsl_runtime_wrapper PRIVATE runtime ascendcl dl pthread m c_sec)
  set_target_properties(tla_dsl_runtime_wrapper PROPERTIES
    OUTPUT_NAME "tla_dsl_runtime_wrapper"
    BUILD_RPATH "${ASCEND_HOME_PATH}/lib64"
  )
endif()
