#ifndef ACOT_GEMM_TILE_COPY_L1_TO_L0_HPP
#define ACOT_GEMM_TILE_COPY_L1_TO_L0_HPP

#include "acot/acot.hpp"
#include "acot/layout/layout.hpp"
#include "acot/gemm/gemm_type.hpp"

namespace acot::gemm::tile{
template<
    class ArchTag,
    class GmType
>
struct CopyL1ToL0A{};

template<>
struct CopyL1ToL0A<acot::arch::AscendC910B3, acot::gemm::GemmType<half, layout::RowMajor>>{
   
    ACOT_DEVICE
    CopyL1ToL0A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<half> l1Tensor,
        AscendC::LocalTensor<half> l0Tensor,
        uint32_t MActual,
        uint32_t MRound,
        uint32_t KActual,
        uint32_t KRound
    ){
        uint32_t ML0Alignment = C0_NUM_PER_FRACTAL;
        uint32_t KL0Alignment = BYTE_PER_C0 / sizeof(half);
        uint32_t MLoops = CeilDiv(MActual, ML0Alignment);
        AscendC::LoadData2DParams params;
        for(uint32_t i = 0; i < MLoops; i++){
            params.startIndex = 0;
            params.repeatTimes = static_cast<uint8_t>(KRound / KL0Alignment);
            params.srcStride = MRound / ML0Alignment;
            params.sid = 0;
            params.dstGap = 0;
            params.ifTranspose = false;
            params.addrMode = 0;
            AscendC::LoadData(l0Tensor[i * ML0Alignment * KRound],l1Tensor[i * KL0Alignment * ML0Alignment],params);
        }
    }
};


template<
    class ArchTag,
    class GmType
>
struct CopyL1ToL0B{};

template<>
struct CopyL1ToL0B<acot::arch::AscendC910B3, acot::gemm::GemmType<half, layout::RowMajor>>{
   
    ACOT_DEVICE
    CopyL1ToL0B(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<half> l1Tensor,
        AscendC::LocalTensor<half> l0Tensor,
        uint32_t NActual,
        uint32_t NRound,
        uint32_t KL0Actual,
        uint32_t KL0Round
    ){
       // 16 字节的
       uint32_t NL0Alignment = BYTE_PER_C0 / sizeof(half);
       uint32_t KL0Alignment = C0_NUM_PER_FRACTAL; // 保证了方阵
       uint32_t KLoops = CeilDiv(KL0Round, KL0Alignment);
       AscendC::LoadData2DParams params;
       for(uint32_t i = 0; i < KLoops; i++){ // 还是k方向进行切割
           params.startIndex = 0;
           params.repeatTimes = static_cast<uint8_t>(NRound / NL0Alignment);
           params.srcStride = 1;
           params.sid = 0;
           params.dstGap = 0;
           params.ifTranspose = true;
           params.addrMode = 0;
           AscendC::LoadData(l0Tensor[i * NRound * KL0Alignment], l1Tensor[i * NRound * KL0Alignment], params);
       }
    }
};
}

#endif // ACOT_GEMM_TILE_COPY_L1_TO_L0_HPP