#ifndef ACOT_GEMM_TILE_COPY_GM_TO_L1_HPP
#define ACOT_GEMM_TILE_COPY_GM_TO_L1_HPP

#include "acot/acot.hpp"
#include "acot/layout/layout.hpp"
#include "acot/gemm/gemm_type.hpp"

namespace acot::gemm::tile{
template<
    class ArchTag,
    class GmType
>
struct CopyGmToL1A{};

template<>
struct CopyGmToL1A<acot::arch::AscendC910B3, acot::gemm::GemmType<half, acot::layout::RowMajor>>{
    
    ACOT_DEVICE
    CopyGmToL1A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::GlobalTensor<half> gmTensor,
        AscendC::LocalTensor<half> l1Tensor,
        uint32_t MActual,
        uint32_t MRound,
        uint32_t KActual,
        uint32_t KRound,
        uint32_t stride
    ){
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = MActual;
        params.dValue = KRound; // 这里直接搬整数倍
        params.srcNdMatrixStride = 0;
        params.srcDValue = stride;
        params.dstNzC0Stride = MRound; // 这边进行填充操作
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        AscendC::DataCopy(l1Tensor,gmTensor,params);
    }
};

template<
    class ArchTag,
    class GmType
>
struct CopyGmToL1B{};

template<>
struct CopyGmToL1B<arch::AscendC910B3, gemm::GemmType<half, acot::layout::RowMajor>>{
    
    ACOT_DEVICE
    CopyGmToL1B(){}
    
    ACOT_DEVICE
    void operator()(
        AscendC::GlobalTensor<half> gmTensor,
        AscendC::LocalTensor<half> l1Tensor,
        uint32_t KActual,
        uint32_t KRound,
        uint32_t NActual,
        uint32_t NRound,
        uint32_t stride
    ){
        uint32_t KL1AlignmentB = C0_NUM_PER_FRACTAL;
        uint32_t KL1Loops = CeilDiv(KActual,KL1AlignmentB);
        // K 方向的切分
        // for循环放进来
        AscendC::Nd2NzParams params;
        for(uint32_t kL1BIdx = 0; kL1BIdx < KL1Loops; kL1BIdx++){ // 没问题了
            uint32_t KGmBActual = (kL1BIdx == KL1Loops - 1) ? (KActual - kL1BIdx * KL1AlignmentB) : KL1AlignmentB;
            uint32_t KGmBRound = RoundUp(KGmBActual, KL1AlignmentB);
            params.ndNum = 1;
            params.nValue = KGmBActual;
            params.dValue = NRound; // 这里直接搬16倍数据内容
            params.srcNdMatrixStride = 0;
            params.srcDValue = stride;
            params.dstNzC0Stride = KGmBRound;  // 这里对齐16 这里会进行填充操作
            params.dstNzNStride = 1;
            params.dstNzMatrixStride = 1;
            AscendC::DataCopy(l1Tensor[kL1BIdx * KL1AlignmentB * NRound], gmTensor[kL1BIdx * KL1AlignmentB * stride], params);
        }
    }
};
}

#endif // ACOT_GEMM_TILE_COPY_GM_TO_L1_HPP