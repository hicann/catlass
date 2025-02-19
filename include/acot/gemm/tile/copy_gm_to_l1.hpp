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

template<class Element>
struct CopyGmToL1A<acot::arch::AscendC910B3, acot::gemm::GemmType<Element, acot::layout::RowMajor>>{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    ACOT_DEVICE
    CopyGmToL1A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t MActual = layoutSrc.shape(0);
        uint32_t MRound = layoutDst.shape(1) * layoutDst.shape(0);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t stride = layoutSrc.stride(0); // 注意layoutSrc的构造函数 
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = MActual;
        params.dValue = KRound; // 这里直接搬整数倍
        params.srcNdMatrixStride = 0;
        params.srcDValue = stride;
        params.dstNzC0Stride = MRound; // 这边进行填充操作
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        AscendC::DataCopy(dstTensor, srcTensor, params);
    }
};

// int8_t 需要特例化
template<class Element>
struct CopyGmToL1A<acot::arch::AscendC910B3, acot::gemm::GemmType<Element, acot::layout::ColumnMajor>>{
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::ColumnMajor;

    ACOT_DEVICE
    CopyGmToL1A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t MAlignment = layoutDst.shape(0); // 对齐32byte
        uint32_t MActual = layoutSrc.shape(1);
        uint32_t MRound = RoundUp(MActual, MAlignment);
        uint32_t KRound = layoutDst.shape(0) * layoutDst.shape(1);
        uint32_t KActual = layoutSrc.shape(0);
        uint32_t stride = layoutSrc.stride(1); // ColumnMajor
        uint32_t KL1AlignmentA = layoutDst.shape(0);
        uint32_t KL1Loops = CeilDiv(KActual, KL1AlignmentA);
        AscendC::Nd2NzParams params;
        for(uint32_t kL1AIdx = 0; kL1AIdx < KL1Loops; kL1AIdx++){ // 16行进行切分，变成zZ
            uint32_t KGmAActual = (kL1AIdx == KL1Loops - 1) ? (KActual - kL1AIdx * KL1AlignmentA) : KL1AlignmentA;
            uint32_t KGmARound = RoundUp(KGmAActual, KL1AlignmentA);
            params.ndNum = 1;
            params.nValue = KGmAActual;
            params.dValue = MRound;
            params.srcNdMatrixStride = 0;
            params.srcDValue = stride;
            params.dstNzC0Stride = KGmARound; // 填充操作
            params.dstNzNStride = 1;
            params.dstNzMatrixStride = 1;
            AscendC::DataCopy(dstTensor[kL1AIdx * KL1AlignmentA * MRound], srcTensor[kL1AIdx * KL1AlignmentA * stride], params);
        }
    }
};

// 特例化int8_t
template<>
struct CopyGmToL1A<acot::arch::AscendC910B3, acot::gemm::GemmType<int8_t, acot::layout::ColumnMajor>>{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::ColumnMajor;

    ACOT_DEVICE
    CopyGmToL1A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> dstTensor,
        AscendC::GlobalTensor<int8_t> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t KAlignment = layoutDst.shape(0) * 2;
        uint32_t MRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t KActual = layoutSrc.shape(0);
        uint32_t KRound = RoundUp(KActual, KAlignment);
        uint32_t stride = layoutSrc.stride(1); // ColumnMajor
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = KActual;
        params.dValue = MRound;
        params.srcNdMatrixStride = 0;
        params.srcDValue = stride;
        params.dstNzC0Stride = KRound; // 填充操作
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        AscendC::DataCopy(dstTensor, srcTensor, params);
    }
};

// int8_t 需要特例化
template<
    class ArchTag,
    class GmType
>
struct CopyGmToL1B{};

template<class Element>
struct CopyGmToL1B<arch::AscendC910B3, gemm::GemmType<Element, acot::layout::RowMajor>>{
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::RowMajor;

    ACOT_DEVICE
    CopyGmToL1B(){}
    
    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t NAlignment = layoutDst.shape(2);
        if constexpr (std::is_same<Element, float>::value){
            NAlignment = layoutDst.shape(0);
        }
        uint32_t NActual = layoutSrc.shape(1);
        uint32_t NRound = RoundUp(NActual, NAlignment);
        uint32_t KActual = layoutSrc.shape(0);
        uint32_t KRound = layoutDst.shape(1) * layoutDst.shape(0);
        uint32_t stride = layoutSrc.stride(0);
        uint32_t KL1AlignmentB = layoutDst.shape(0);
        uint32_t KL1Loops = CeilDiv(KActual, KL1AlignmentB);
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
            AscendC::DataCopy(dstTensor[kL1BIdx * KL1AlignmentB * NRound], srcTensor[kL1BIdx * KL1AlignmentB * stride], params);
        }
    }
};

// 特例化int8_t
template<>
struct CopyGmToL1B<arch::AscendC910B3, gemm::GemmType<int8_t, acot::layout::RowMajor>>{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    ACOT_DEVICE
    CopyGmToL1B(){}
    
    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> dstTensor,
        AscendC::GlobalTensor<int8_t> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t NActual = layoutSrc.shape(1);
        uint32_t NRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t KActual = layoutSrc.shape(0);
        uint32_t KRound = RoundUp(KActual, layoutDst.shape(0) * 2);
        uint32_t stride = layoutSrc.stride(0);
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = KActual;
        params.dValue = NRound; // 这里直接搬16倍数据内容
        params.srcNdMatrixStride = 0;
        params.srcDValue = stride;
        params.dstNzC0Stride = KRound;  // 这里对齐16 这里会进行填充操作
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        AscendC::DataCopy(dstTensor, srcTensor, params);
    }
};

template<class Element>
struct CopyGmToL1B<arch::AscendC910B3, gemm::GemmType<Element, acot::layout::ColumnMajor>>{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::ColumnMajor;

    ACOT_DEVICE
    CopyGmToL1B(){}
    
    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t NActual = layoutSrc.shape(0);
        uint32_t NRound = layoutDst.shape(0) * layoutDst.shape(1);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t stride = layoutSrc.stride(1); // ColumnMajor
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = NActual;
        params.dValue = KRound;
        params.srcNdMatrixStride = 0;
        params.srcDValue = stride;
        params.dstNzC0Stride = NRound;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        AscendC::DataCopy(dstTensor, srcTensor, params);
    }
};
}

#endif // ACOT_GEMM_TILE_COPY_GM_TO_L1_HPP