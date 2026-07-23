#include "../common.h"

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) ||                    \
     (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_printf_x_i32(int32_t value, uint64_t printWorkspace) {
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t *>(printWorkspace);
    AscendC::printf("x=%d", value);
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_printf_v_f32(float value, uint64_t printWorkspace) {
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t *>(printWorkspace);
    AscendC::printf("v=%f", value);
}
#endif
}
