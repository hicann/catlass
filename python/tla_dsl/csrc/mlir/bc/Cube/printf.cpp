#include "../common.h"

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) ||                    \
     (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_printf_x_i8(int8_t value, uint64_t printWorkspace) {
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t *>(printWorkspace);
    AscendC::printf("x=%d", value);
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_printf_x_i16(int16_t value, uint64_t printWorkspace) {
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t *>(printWorkspace);
    AscendC::printf("x=%d", value);
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_printf_x_u8(uint8_t value, uint64_t printWorkspace) {
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t *>(printWorkspace);
    AscendC::printf("x=%u", static_cast<uint32_t>(value));
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_printf_x_u16(uint16_t value, uint64_t printWorkspace) {
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t *>(printWorkspace);
    AscendC::printf("x=%u", static_cast<uint32_t>(value));
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_printf_x_u32(uint32_t value, uint64_t printWorkspace) {
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t *>(printWorkspace);
    AscendC::printf("x=%u", value);
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_printf_x_i32(int32_t value, uint64_t printWorkspace) {
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t *>(printWorkspace);
    AscendC::printf("x=%d", value);
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_printf_v_f16(half value, uint64_t printWorkspace) {
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t *>(printWorkspace);
    AscendC::printf("v=%f", static_cast<float>(value));
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_printf_v_f32(float value, uint64_t printWorkspace) {
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t *>(printWorkspace);
    AscendC::printf("v=%f", value);
}
#endif
}
