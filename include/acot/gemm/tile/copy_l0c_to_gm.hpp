#ifndef ACOT_GEMM_TILE_COPY_L0C_TO_GM_HPP
#define ACOT_GEMM_TILE_COPY_L0C_TO_GM_HPP

#include "acot/gemm/gemm_type.hpp"

namespace acot::gemm::tile{

template<
    typename DstType,
    typename SrcType
>
__aicore__ inline __attribute__((always_inline)) auto getQuantMode() {
    if (std::is_same<DstType, SrcType>::value) {
        return QuantMode_t::NoQuant;
    }
    static_assert("not implement quantMode!");
    return QuantMode_t::NoQuant;
}

template<>
__aicore__ inline __attribute__((always_inline)) auto getQuantMode<half, float>() {
    return QuantMode_t::F322F16;
}

template<>
__aicore__ inline __attribute__((always_inline)) auto getQuantMode<bfloat16_t, float>() {
    return QuantMode_t::F322BF16;
}

template<
    class ArchTag,
    class ElementAccumulator,
    class GmType
>
struct CopyL0CToGm{};

template<>
struct CopyL0CToGm<acot::arch::AscendC910B3,float,
    acot::gemm::GemmType<half, layout::RowMajor>
>
{
    ACOT_DEVICE
    CopyL0CToGm(){}

    ACOT_DEVICE
    void operator()(
        AscendC::GlobalTensor<half> gmTensor,
        AscendC::LocalTensor<float> l0CTensor,
        uint32_t MRound,
        uint32_t NRound,
        uint32_t MActual,
        uint32_t NActual,
        uint32_t strideC
    ){
        AscendC::DataCopyCO12DstParams params;
        // 一定是Nz2Nd的，而且L0C一定是行优先的
        params.nSize = NActual; // 参数写反了  解决了问题
        params.mSize = MActual;
        params.dstStride = strideC;
        params.srcStride = MRound;
        params.quantPre = getQuantMode<half, float>();
        params.reluPre = 0;
        params.channelSplit = false;
        params.nz2ndEn = true;
        AscendC::DataCopy(gmTensor,l0CTensor,params);
    }
};
}

#endif // ACOT_GEMM_TILE_COPY_L0C_TO_GM_HPP