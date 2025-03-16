#ifndef ACOT_GEMM_TILE_COPY_L0C_TO_GM_HPP
#define ACOT_GEMM_TILE_COPY_L0C_TO_GM_HPP

#include "acot/matmul/matmul_type.hpp"

namespace acot::gemm::tile{

enum class ScaleGranularity {
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};

template <
    class ArchTag,
    class ElementSrc,
    class ElementDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT
>
struct CopyL0CToGmQuantMode {};

// CopyL0CToGm cast fp32 to fp16
template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AtlasA2,
    float, half,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

// CopyL0CToGm cast fp32 to bf16
template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AtlasA2,
    float, bfloat16_t,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

// CopyL0CToGm output fp32
template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AtlasA2,
    float, float,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

// CopyL0CToGm output int32
template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AtlasA2,
    int32_t, int32_t,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AtlasA2,
    int32_t, half,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::DEQF16;
};

// CopyL0CToGm cast int32_t to fp16
template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AtlasA2,
    int32_t, half,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::DEQF16;
};

template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AtlasA2,
    int32_t, half,
    ScaleGranularity::PER_CHANNEL
> {
    static constexpr auto VALUE = QuantMode_t::VDEQF16;
};

template<
    class ArchTag,
    class ElementAccumulator,
    class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
struct CopyL0CToGm{};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<acot::arch::AtlasA2, ElementAccumulator_,
    acot::matmul::MatmulType<ElementDst_, layout::RowMajor>,
    ScaleGranularity::NO_QUANT,
    ReluEnable_>
{
    using ArchTag = acot::arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = acot::layout::zN;
    using LayoutDst = acot::layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(ElementDst);

    ACOT_DEVICE
    CopyL0CToGm(){}

    ACOT_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> dstTensor,
        AscendC::LocalTensor<ElementSrc> srcTensor,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0
    ){
        uint32_t MActual = dstLayout.shape(0);
        uint32_t NActual = dstLayout.shape(1);
        uint32_t MRound = srcLayout.shape(0) * srcLayout.shape(1);
        uint32_t strideC = dstLayout.stride(0);
        AscendC::DataCopyCO12DstParams params;
        params.nSize = NActual;
        params.mSize = MActual;
        params.dstStride = strideC;
        params.srcStride = MRound; 
        params.quantPre = quantPre;
        params.reluPre = 0;
        params.channelSplit = false;
        params.unitFlag = unitFlag;
        params.nz2ndEn = true;
        AscendC::DataCopy(dstTensor, srcTensor, params);
    }
};


template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<acot::arch::AtlasA2, ElementAccumulator_,
    acot::matmul::MatmulType<ElementDst_, layout::ColumnMajor>,
    ScaleGranularity::NO_QUANT,
    ReluEnable_>
{
    using ArchTag = acot::arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = acot::layout::zN;
    using LayoutDst = acot::layout::ColumnMajor;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(ElementDst);

    ACOT_DEVICE
    CopyL0CToGm(){}

    ACOT_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> dstTensor,
        AscendC::LocalTensor<ElementSrc> srcTensor,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0
    ){
        uint32_t MActual = dstLayout.shape(0);
        uint32_t NActual = dstLayout.shape(1);
        uint32_t NRound = srcLayout.shape(2) * srcLayout.shape(3);
        uint32_t strideC = dstLayout.stride(1);
        AscendC::DataCopyCO12DstParams params;
        params.nSize = MActual;
        params.mSize = NActual;
        params.dstStride = strideC;
        params.srcStride = NRound;
        params.quantPre = quantPre;
        params.reluPre = 0;
        params.channelSplit = false;
        params.nz2ndEn = true;
        AscendC::DataCopy(dstTensor, srcTensor, params);
    }
};
}

#endif // ACOT_GEMM_TILE_COPY_L0C_TO_GM_HPP