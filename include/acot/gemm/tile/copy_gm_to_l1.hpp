#ifndef ACOT_GEMM_TILE_COPY_GM_TO_L1_HPP
#define ACOT_GEMM_TILE_COPY_GM_TO_L1_HPP

#include "acot/acot.hpp"
#include "acot/layout/layout.hpp"
#include "acot/gemm/gemm_type.hpp"

constexpr uint32_t STRIDE_LIMIT = 65536;

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
        uint32_t MRound = layoutDst.shape(0) * layoutDst.shape(1);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t stride = layoutSrc.stride(0); // 注意layoutSrc的构造函数 
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = MActual;
        params.dValue = KRound;
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
        uint32_t MRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t KActual = layoutSrc.shape(0);
        uint32_t stride = layoutSrc.stride(1); // ColumnMajor
        uint32_t ndNum = KActual / C0_NUM_PER_FRACTAL;
        uint32_t remains = KActual % C0_NUM_PER_FRACTAL;
        uint32_t srcNdStride = stride * C0_NUM_PER_FRACTAL;
        if(srcNdStride < STRIDE_LIMIT){
            AscendC::Nd2NzParams params;
            if(ndNum){ 
                params.ndNum = ndNum;
                params.nValue = C0_NUM_PER_FRACTAL;
                params.dValue = MRound;
                params.srcNdMatrixStride = stride * C0_NUM_PER_FRACTAL;
                params.srcDValue = stride;
                params.dstNzC0Stride = C0_NUM_PER_FRACTAL;
                params.dstNzNStride = 1;
                params.dstNzMatrixStride = MRound * C0_NUM_PER_FRACTAL;
                AscendC::DataCopy(dstTensor, srcTensor, params);
            }
            if(remains){
                params.ndNum = 1;
                params.nValue = remains;
                params.dValue = MRound;
                params.srcNdMatrixStride = 0;
                params.srcDValue = stride;
                params.dstNzC0Stride = C0_NUM_PER_FRACTAL;
                params.dstNzNStride = 1;
                params.dstNzMatrixStride = 1;
                AscendC::DataCopy(dstTensor[ndNum * MRound * C0_NUM_PER_FRACTAL], srcTensor[ndNum * C0_NUM_PER_FRACTAL * stride], params);
            }
        }else if(stride < STRIDE_LIMIT){
            AscendC::Nd2NzParams params;
            params.ndNum = 1;
            params.nValue = C0_NUM_PER_FRACTAL;
            params.dValue = MRound;
            params.srcNdMatrixStride = 0;
            params.srcDValue = stride;
            params.dstNzC0Stride = C0_NUM_PER_FRACTAL;
            params.dstNzNStride = 1;
            params.dstNzMatrixStride = 1;
            for(uint32_t i = 0; i < ndNum; i++){
                AscendC::DataCopy(dstTensor[i * MRound * C0_NUM_PER_FRACTAL], srcTensor[i * C0_NUM_PER_FRACTAL * stride], params);
            }
            if(remains){
                params.ndNum = 1;
                params.nValue = remains;
                params.dValue = MRound;
                params.srcNdMatrixStride = 0;
                params.srcDValue = stride;
                params.dstNzC0Stride = C0_NUM_PER_FRACTAL;
                params.dstNzNStride = 1;
                params.dstNzMatrixStride = 1;
                AscendC::DataCopy(dstTensor[ndNum * MRound * C0_NUM_PER_FRACTAL], srcTensor[ndNum * C0_NUM_PER_FRACTAL * stride], params);
            }
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
        uint32_t MRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t KActual = layoutSrc.shape(0);
        uint32_t KRound = layoutDst.shape(0) * layoutDst.shape(1);
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
        uint32_t NRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t KActual = layoutSrc.shape(0);
        uint32_t stride = layoutSrc.stride(0); // 这个变了
        uint32_t ndNum = KActual / C0_NUM_PER_FRACTAL;
        uint32_t remains = KActual % C0_NUM_PER_FRACTAL;
        uint32_t srcNdStride = stride * C0_NUM_PER_FRACTAL;
        if(srcNdStride < STRIDE_LIMIT){
            AscendC::Nd2NzParams params;
            if(ndNum){ 
                params.ndNum = ndNum;
                params.nValue = C0_NUM_PER_FRACTAL;
                params.dValue = NRound;
                params.srcNdMatrixStride = stride * C0_NUM_PER_FRACTAL;
                params.srcDValue = stride;
                params.dstNzC0Stride = C0_NUM_PER_FRACTAL;
                params.dstNzNStride = 1;
                params.dstNzMatrixStride = NRound * C0_NUM_PER_FRACTAL;
                AscendC::DataCopy(dstTensor, srcTensor, params);
            }
            if(remains){
                params.ndNum = 1;
                params.nValue = remains;
                params.dValue = NRound;
                params.srcNdMatrixStride = 0;
                params.srcDValue = stride;
                params.dstNzC0Stride = C0_NUM_PER_FRACTAL;
                params.dstNzNStride = 1;
                params.dstNzMatrixStride = 1;
                AscendC::DataCopy(dstTensor[ndNum * NRound * C0_NUM_PER_FRACTAL], srcTensor[ndNum * C0_NUM_PER_FRACTAL * stride], params);
            }
        }else if(stride < STRIDE_LIMIT){
            AscendC::Nd2NzParams params;
            params.ndNum = 1;
            params.nValue = C0_NUM_PER_FRACTAL;
            params.dValue = NRound;
            params.srcNdMatrixStride = 0;
            params.srcDValue = stride;
            params.dstNzC0Stride = C0_NUM_PER_FRACTAL;
            params.dstNzNStride = 1;
            params.dstNzMatrixStride = 1;
            for(uint32_t i = 0; i < ndNum; i++){
                AscendC::DataCopy(dstTensor[i * NRound * C0_NUM_PER_FRACTAL], srcTensor[i * C0_NUM_PER_FRACTAL * stride], params);
            }
            if(remains){
                params.ndNum = 1;
                params.nValue = remains;
                params.dValue = NRound;
                params.srcNdMatrixStride = 0;
                params.srcDValue = stride;
                params.dstNzC0Stride = C0_NUM_PER_FRACTAL;
                params.dstNzNStride = 1;
                params.dstNzMatrixStride = 1;
                AscendC::DataCopy(dstTensor[ndNum * NRound * C0_NUM_PER_FRACTAL], srcTensor[ndNum * C0_NUM_PER_FRACTAL * stride], params);
            }
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
        uint32_t NRound = layoutDst.shape(2) * layoutDst.shape(3); 
        uint32_t KActual = layoutSrc.shape(0);
        uint32_t KRound = layoutDst.shape(0) * layoutDst.shape(1); // 组成方阵
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