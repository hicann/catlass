#ifndef ACOT_GEMM_TILE_TILE_MMAD_HPP
#define ACOT_GEMM_TILE_TILE_MMAD_HPP

#include "acot/acot.hpp"
#include "acot/gemm/helper.hpp"

namespace acot::gemm::tile{
template<
    class ArchTag,
    class AType,
    class BType,
    class CType
>
struct TileMmad{
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    ACOT_DEVICE
    TileMmad() {}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementAccumulator> dstTensor,
        AscendC::LocalTensor<ElementA> src0Tensor,
        AscendC::LocalTensor<ElementB> src1Tensor,
        uint32_t leftH,
        uint32_t rightW,
        uint32_t common,
        bool isFirst
    ){
        AscendC::MmadParams params;
        params.m = leftH;
        params.n = rightW;
        params.k = common;
        params.cmatrixInitVal = isFirst;
        params.cmatrixSource = false; // 指定从CO1来数据
        AscendC::Mmad(
            dstTensor,
            src0Tensor,
            src1Tensor,
            params
        );
    }
};
}

#endif // ACOT_GEMM_TILE_TILE_MMAD_HPP