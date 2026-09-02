add_mlir_public_c_api_library(TlaCAPI
  Dialect/Tla.cpp
  DISABLE_INSTALL
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/catlass-c
  PARTIAL_SOURCES_INTENDED
  DEPENDS
  TlaTblgen
  LINK_LIBS PUBLIC
  MLIRCAPIIR
)

target_include_directories(obj.TlaCAPI
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${MLIR_INCLUDE_DIRS}
)
set_target_properties(obj.TlaCAPI PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
