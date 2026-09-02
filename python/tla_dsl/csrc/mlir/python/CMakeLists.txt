set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL
    "Suppress developer warnings from the pinned MLIR Python helper" FORCE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

set(_npuir_mlir_src
    "${_catlass_dsl_ascendnpu_ir_source_dir}/third-party/llvm-project/mlir")
if(NOT EXISTS "${_npuir_mlir_src}/cmake/modules/AddMLIRPython.cmake")
  message(FATAL_ERROR
          "AddMLIRPython.cmake not found under the AscendNPU-IR source "
          "tree: ${_npuir_mlir_src}/cmake/modules/AddMLIRPython.cmake")
endif()
include("${_npuir_mlir_src}/cmake/modules/AddMLIRPython.cmake")

declare_mlir_python_sources(CatlassMLIRPythonSources
  ROOT_DIR "${_npuir_mlir_src}/python/mlir"
  SOURCES
    _mlir_libs/__init__.py
    ir.py
    passmanager.py
    rewrite.py
    dialects/_ods_common.py
    extras/types.py
    extras/meta.py
    _mlir_libs/_mlir/__init__.pyi
    _mlir_libs/_mlir/ir.pyi
    _mlir_libs/_mlir/passmanager.pyi
)

set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" APPEND PROPERTY
  INCLUDE_DIRECTORIES "${_npuir_mlir_src}/include")

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CatlassMLIRPythonSources
  ROOT_DIR "${_npuir_mlir_src}/python/mlir"
  TD_FILE dialects/ArithOps.td
  SOURCES
    dialects/arith.py
  DIALECT_NAME arith
  GEN_ENUM_BINDINGS)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CatlassMLIRPythonSources
  ROOT_DIR "${_npuir_mlir_src}/python/mlir"
  TD_FILE dialects/MemRefOps.td
  SOURCES
    dialects/memref.py
  DIALECT_NAME memref)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CatlassMLIRPythonSources
  ROOT_DIR "${_npuir_mlir_src}/python/mlir"
  TD_FILE dialects/SCFOps.td
  SOURCES
    dialects/scf.py
  DIALECT_NAME scf)

declare_mlir_python_extension(CatlassMLIRPythonExtension.Core
  MODULE_NAME _mlir
  ROOT_DIR "${_npuir_mlir_src}/lib/Bindings/Python"
  SOURCES
    MainModule.cpp
    IRAffine.cpp
    IRAttributes.cpp
    IRCore.cpp
    IRInterfaces.cpp
    IRModule.cpp
    IRTypes.cpp
    Pass.cpp
    Rewrite.cpp

    Globals.h
    IRModule.h
    Pass.h
    PybindUtils.h
    Rewrite.h
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIDebug
    MLIRCAPIIR
    MLIRCAPIInterfaces
    MLIRCAPIFunc
)

declare_mlir_python_extension(CatlassMLIRPythonExtension.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ROOT_DIR "${_npuir_mlir_src}/lib/Bindings/Python"
  SOURCES
    RegisterEverything.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIConversion
    MLIRCAPITransforms
    MLIRCAPIRegisterEverything
)

set(CATLASS_MLIR_PYTHON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/catlass_mlir")

declare_mlir_python_sources(CatlassPythonSources
  ROOT_DIR "${CATLASS_MLIR_PYTHON_ROOT}"
  SOURCES
    __init__.py
    dialects/__init__.py
    _mlir_libs/_site_initialize_0.py
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CatlassPythonSources
  ROOT_DIR "${CATLASS_MLIR_PYTHON_ROOT}"
  TD_FILE dialects/TlaOps.td
  SOURCES
    dialects/tla.py
  DIALECT_NAME tla)

set(CATLASS_PYTHON_BINDINGS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bindings")

set(_catlass_embed_capi_libs
  TlaCAPI)

declare_mlir_python_extension(
  TlaPythonExtension.RegisterDialect
  MODULE_NAME _tlaRegisterDialect
  ROOT_DIR "${CATLASS_PYTHON_BINDINGS_SOURCE_DIR}"
  SOURCES
    RegisterTlaDialect.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    ${_catlass_embed_capi_libs}
)

set(CatlassPythonComponents
  CatlassMLIRPythonSources
  CatlassMLIRPythonExtension.Core
  CatlassMLIRPythonExtension.RegisterEverything
  CatlassPythonSources
  TlaPythonExtension.RegisterDialect
)

if(NOT DEFINED CATLASS_DSL_PYTHON_PACKAGE_OUTPUT_DIR)
  message(FATAL_ERROR "CATLASS_DSL_PYTHON_PACKAGE_OUTPUT_DIR is not set")
endif()

add_mlir_python_common_capi_library(CatlassAggregateCAPI
  INSTALL_COMPONENT CatlassPythonModules
  INSTALL_DESTINATION catlass/_mlir/_mlir_libs
  OUTPUT_DIRECTORY
    "${CATLASS_DSL_PYTHON_PACKAGE_OUTPUT_DIR}/_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../.."
  DECLARED_SOURCES ${CatlassPythonComponents}
)
set(_catlass_aggregate_capi_version "${LLVM_PACKAGE_VERSION}")
if(NOT _catlass_aggregate_capi_version)
  set(_catlass_aggregate_capi_version "19.1")
endif()
string(REGEX REPLACE "^([0-9]+\\.[0-9]+).*$" "\\1"
       _catlass_aggregate_capi_version "${_catlass_aggregate_capi_version}")
set_target_properties(CatlassAggregateCAPI PROPERTIES
  VERSION "${_catlass_aggregate_capi_version}"
  SOVERSION "${_catlass_aggregate_capi_version}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
  BUILD_WITH_INSTALL_RPATH ON
)
target_link_libraries(CatlassAggregateCAPI PRIVATE
  -Wl,--whole-archive
  $<TARGET_FILE:TlaCompilePipeline>
  $<TARGET_FILE:TlaPasses>
  -Wl,--no-whole-archive
  -Wl,--start-group
  ${CATLASS_DSL_HIVM_REQUIRED_LIB_FILES}
  -Wl,--end-group
)
add_dependencies(CatlassAggregateCAPI TlaCompilePipeline TlaPasses)

add_mlir_python_modules(CatlassPythonModules
  ROOT_PREFIX "${CATLASS_DSL_PYTHON_PACKAGE_OUTPUT_DIR}/_mlir"
  INSTALL_PREFIX "catlass/_mlir"
  DECLARED_SOURCES ${CatlassPythonComponents}
  COMMON_CAPI_LINK_LIBS CatlassAggregateCAPI
)

foreach(_extension_target
    CatlassPythonModules.extension._mlir.dso
    CatlassPythonModules.extension._mlirRegisterEverything.dso
    CatlassPythonModules.extension._tlaRegisterDialect.dso)
  if(TARGET ${_extension_target})
    set_target_properties(${_extension_target} PROPERTIES
      BUILD_RPATH "\$ORIGIN"
      INSTALL_RPATH "\$ORIGIN"
      BUILD_WITH_INSTALL_RPATH ON
    )
  endif()
endforeach()
