#ifndef ACOT_GEMM_TILE_COPY_L0C_TO_GM_HPP
#define ACOT_GEMM_TILE_COPY_L0C_TO_GM_HPP

#include "acot/gemm/gemm_type.hpp"

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
    acot::arch::AscendC910B3,
    float, half,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

// CopyL0CToGm cast fp32 to bf16
template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AscendC910B3,
    float, bfloat16_t,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

// CopyL0CToGm output fp32
template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AscendC910B3,
    float, float,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

// CopyL0CToGm output int32
template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AscendC910B3,
    int32_t, int32_t,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AscendC910B3,
    int32_t, half,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::DEQF16;
};

// CopyL0CToGm cast int32_t to fp16
template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AscendC910B3,
    int32_t, half,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::DEQF16;
};

template <>
struct CopyL0CToGmQuantMode<
    acot::arch::AscendC910B3,
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
struct CopyL0CToGm<acot::arch::AscendC910B3, ElementAccumulator_,
    acot::gemm::GemmType<ElementDst_, layout::RowMajor>,
    ScaleGranularity::NO_QUANT,
    ReluEnable_>
{
    using ArchTag = acot::arch::AscendC910B3;
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
        uint32_t MAlignment = srcLayout.shape(0);
        uint32_t NAlignment = ELE_NUM_PER_C0;
        if constexpr (std::is_same<ElementSrc, float>::value && std::is_same<ElementDst, float>::value){
            // 说明原来的一定是float数据类型
            NAlignment = srcLayout.shape(0);
        }
        uint32_t MActual = dstLayout.shape(0);
        uint32_t NActual = dstLayout.shape(1);
        uint32_t MRound = RoundUp(MActual, MAlignment);
        uint32_t NRound = RoundUp(NActual, NAlignment);
        uint32_t strideC = dstLayout.stride(0);
        AscendC::DataCopyCO12DstParams params;
        // 一定是Nz2Nd的，而且L0C一定是行优先的
        params.nSize = NActual; // 参数写反了  解决了问题
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
struct CopyL0CToGm<acot::arch::AscendC910B3, ElementAccumulator_,
    acot::gemm::GemmType<ElementDst_, layout::ColumnMajor>,
    ScaleGranularity::NO_QUANT,
    ReluEnable_>
{
    using ArchTag = acot::arch::AscendC910B3;
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
        uint32_t MAlignment = srcLayout.shape(0); // 对齐32byte
        if constexpr (std::is_same<ElementDst, int32_t>::value){
            MAlignment = ELE_NUM_PER_C0;
        }
        uint32_t NAlignment = srcLayout.shape(0); // 对齐16行
        uint32_t MActual = dstLayout.shape(0);
        uint32_t NActual = dstLayout.shape(1);
        uint32_t MRound = RoundUp(MActual, MAlignment);
        uint32_t NRound = RoundUp(NActual, NAlignment);
        uint32_t strideC = dstLayout.stride(1);
        AscendC::DataCopyCO12DstParams params;
        // 一定是Nz2Nd的，而且L0C一定是行优先的
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