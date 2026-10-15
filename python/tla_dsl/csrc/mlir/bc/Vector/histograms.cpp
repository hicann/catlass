#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

// Reg Histograms via CCE dhistv2 / chistv2.
// FREQUENCY → dhistv2 (per-value counts in [0,127] / [128,255]).
// ACCUMULATE → chistv2 (CDF: dst[n]+=#{src<=n} / dst[n]+=#{src<=128+n}).
// BIN0/BIN1 select which 128-lane half is written.

namespace {

template <bool Accumulate, bool Bin1>
__aiv__ __attribute__((always_inline)) VectorReg<uint16_t> histogramsU8(
    VectorReg<uint8_t> src, VectorReg<uint16_t> dst, ave_preg preg)
{
    vector_bool mask = convertAVEPregToVecBool(preg);
    if constexpr (Accumulate) {
        if constexpr (Bin1)
            chistv2(dst, src, mask, Bin_N1);
        else
            chistv2(dst, src, mask, Bin_N0);
    } else {
        if constexpr (Bin1)
            dhistv2(dst, src, mask, Bin_N1);
        else
            dhistv2(dst, src, mask, Bin_N0);
    }
    return dst;
}

#define DECLARE_HISTOGRAMS(name)                                     \
    __aiv__ __attribute__((always_inline)) VectorReg<uint16_t> name( \
        VectorReg<uint8_t> src, VectorReg<uint16_t> dst, ave_preg preg)

#define REGISTER_HISTOGRAMS(name, Acc, Bin1)            \
    DECLARE_HISTOGRAMS(name)                            \
    {                                                   \
        return histogramsU8<Acc, Bin1>(src, dst, preg); \
    }

} // namespace

extern "C" {
REGISTER_HISTOGRAMS(histograms_frequency_bin0_u8, false, false);
REGISTER_HISTOGRAMS(histograms_frequency_bin1_u8, false, true);
REGISTER_HISTOGRAMS(histograms_accumulate_bin0_u8, true, false);
REGISTER_HISTOGRAMS(histograms_accumulate_bin1_u8, true, true);
}

#endif
