add_executable(TlaCompile main.cpp)
target_include_directories(TlaCompile PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/tblgen
  ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(TlaCompile PRIVATE TlaCompilePipeline TlaPasses MLIRParser)
