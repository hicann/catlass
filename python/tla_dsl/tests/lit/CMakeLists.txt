find_program(TLA_FILECHECK_TOOL NAMES FileCheck HINTS ${LLVM_TOOLS_BINARY_DIR})
if(NOT TLA_FILECHECK_TOOL)
  set(TLA_FILECHECK_TOOL FileCheck)
endif()

if(DEFINED LLVM_EXTERNAL_LIT AND EXISTS "${LLVM_EXTERNAL_LIT}")
  set(TLA_LIT_RUNNER "${Python3_EXECUTABLE}" "${LLVM_EXTERNAL_LIT}")
else()
  find_program(
    TLA_LLVM_LIT_TOOL
    NAMES llvm-lit lit
    HINTS ${LLVM_TOOLS_BINARY_DIR}
    PATHS "$ENV{HOME}/.local/bin"
  )
  if(TLA_LLVM_LIT_TOOL)
    # Prefer an executable lit runner; this gives deterministic pass/fail behavior.
    set(TLA_LIT_RUNNER "${TLA_LLVM_LIT_TOOL}")
  else()
    # Fallback when no lit executable is discoverable.
    # Use a helper script rather than inline `-c` shell text to avoid quoting issues.
    set(TLA_LIT_RUNNER
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_BINARY_DIR}/run_lit.py"
    )
  endif()
endif()

set(TLA_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TLA_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
if(DEFINED TLA_DSL_TLA_COMPILE_TOOL)
  set(TLA_COMPILE_TOOL "${TLA_DSL_TLA_COMPILE_TOOL}")
else()
  set(TLA_COMPILE_TOOL ${CMAKE_BINARY_DIR}/tools/tla-compile/TlaCompile)
endif()
set(TLA_LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})
set(TLA_FILECHECK_TOOL ${TLA_FILECHECK_TOOL})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  @ONLY
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run_lit.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/run_lit.py
  COPYONLY
)

add_custom_target(check-tla-lit
  COMMAND ${TLA_LIT_RUNNER} -sv ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS TlaCompile
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running ascend-catlass-dsl lit tests"
)
