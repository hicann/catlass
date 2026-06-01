add_kernel(NAME big_matmul_tla
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/big_matmul_tla.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/big_matmul_tla_impl.cpp)
