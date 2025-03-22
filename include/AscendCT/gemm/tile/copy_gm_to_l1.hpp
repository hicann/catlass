/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDCT_MATMUL_TILE_COPY_GM_TO_L1_HPP
#define ASCENDCT_MATMUL_TILE_COPY_GM_TO_L1_HPP

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/layout/layout.hpp"
#include "AscendCT/gemm/matmul_type.hpp"
#include "tla/tensor.hpp"

using namespace tla;

namespace AscendCT::gemm::tile {

template <
    class ArchTag,
    /// MatmulType for matrix operand
    class GmType,
    class L1Type = void
>
struct CopyGmToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

/// Partial specialization for AtlasA2, RowMajor in and zN out.
/// Matrix A confirm
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, gemm::MatmulType<Element, layout::RowMajor>, gemm::MatmulType<Element, layout::zN>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    ASCENDCT_DEVICE
    CopyGmToL1() {};

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t MActual = layoutSrc.shape(0);
        uint32_t MRound = layoutDst.shape(0) * layoutDst.shape(1);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t stride = layoutSrc.stride(0); 
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = MActual;
        params.dValue = KRound;
        params.srcNdMatrixStride = 0;
        params.srcDValue = stride;
        params.dstNzC0Stride = MRound;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        AscendC::DataCopy(dstTensor, srcTensor, params);
    }
};

/// Matrix B int8_t
template<class ArchTag>
struct CopyGmToL1<ArchTag, gemm::MatmulType<int8_t, layout::RowMajor>, gemm::MatmulType<int8_t, layout::zN>>{
    using Element = int8_t;
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    ASCENDCT_DEVICE
    CopyGmToL1(){}
    
    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t NRound = layoutDst.shape(2) * layoutDst.shape(3); 
        uint32_t KActual = layoutSrc.shape(0);
        uint32_t KRound = layoutDst.shape(0) * layoutDst.shape(1); 
        uint32_t stride = layoutSrc.stride(0);
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = KActual;
        params.dValue = NRound; 
        params.srcNdMatrixStride = 0;
        params.srcDValue = stride;
        params.dstNzC0Stride = KRound; 
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        AscendC::DataCopy(dstTensor, srcTensor, params);
    }
};

/// Matrix B new add
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, gemm::MatmulType<Element, layout::RowMajor>, gemm::MatmulType<Element, layout::zZ>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    ASCENDCT_DEVICE
    CopyGmToL1() {};

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t NRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t KActual = layoutSrc.shape(0);
        uint32_t stride = layoutSrc.stride(0); 
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

/// Matrix A  new add
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, gemm::MatmulType<Element, layout::ColumnMajor>, gemm::MatmulType<Element, layout::nN>> {
    using LayoutDst = layout::nN;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    ASCENDCT_DEVICE
    CopyGmToL1() {};

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t MRound = layoutDst.shape(0) * layoutDst.shape(1);
        uint32_t KActual = layoutSrc.shape(1);
        uint32_t stride = layoutSrc.stride(1); 
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

/// matrix A int8_t
template <class ArchTag>
struct CopyGmToL1<ArchTag, gemm::MatmulType<int8_t, layout::ColumnMajor>, gemm::MatmulType<int8_t, layout::nZ>> {
    using Element = int8_t;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    ASCENDCT_DEVICE
    CopyGmToL1() {};

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t MRound = layoutDst.shape(0) * layoutDst.shape(1);
        uint32_t KActual = layoutSrc.shape(1);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t stride = layoutSrc.stride(1);
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = KActual;
        params.dValue = MRound;
        params.srcNdMatrixStride = 0;
        params.srcDValue = stride;
        params.dstNzC0Stride = KRound;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        AscendC::DataCopy(dstTensor, srcTensor, params);
    }
};

/// Matrix B
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, gemm::MatmulType<Element, layout::ColumnMajor>, gemm::MatmulType<Element, layout::nZ>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    ASCENDCT_DEVICE
    CopyGmToL1() {};

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t NActual = layoutSrc.shape(1);
        uint32_t NRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t KRound = layoutDst.shape(0) * layoutDst.shape(1);
        uint32_t stride = layoutSrc.stride(1); 
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

/// Partial specialization for AtlasA2, RowMajor in and zN out.
template <class Element>
struct CopyGmToL1<arch::AtlasA2, gemm::MatmulType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    ASCENDCT_DEVICE
    CopyGmToL1() {};

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }

    // layoutSrc must be the layout of one of the src matrices
    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t ndNum, uint32_t srcNdMatrixStride,
        uint32_t dstNzNStride, uint32_t dstNzMatrixStride,
        uint32_t dstNzC0Stride)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.nValue = layoutSrc.shape(0);
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = dstNzNStride;
        intriParams.dstNzC0Stride = dstNzC0Stride;
        if (srcNdMatrixStride < STRIDE_LIMIT) {
            intriParams.ndNum = ndNum;
            intriParams.srcNdMatrixStride = srcNdMatrixStride;
            intriParams.dstNzMatrixStride = dstNzMatrixStride;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.ndNum = 1;
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * srcNdMatrixStride], intriParams);
            }
        }
    }
};

/// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
template <
    class Element
>
struct CopyGmToL1<arch::AtlasA2, gemm::MatmulType<Element, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    ASCENDCT_DEVICE
    CopyGmToL1() {};

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};

/// Partial specialization for zN in and zN out.
template <
    class ArchTag,
    class Element
>
struct CopyGmToL1<ArchTag, gemm::MatmulType<Element, layout::zN>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    ASCENDCT_DEVICE
    CopyGmToL1() {};

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1));
        uint32_t blockLen = RoundUp<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(0));

        AscendC::DataCopyParams repeatParams;

        if (layoutSrc.stride(3) / ELE_NUM_PER_C0 < STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = 0;
            AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                uint64_t dstOffset = i * layoutDst.stride(3);
                uint64_t srcOffset = i * layoutSrc.stride(3);
                AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], repeatParams);
            }
        }
    }
};

/// Partial specialization for nZ in and nZ out.
template <
    class ArchTag,
    class Element
>
struct CopyGmToL1<ArchTag, gemm::MatmulType<Element, layout::nZ>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    ASCENDCT_DEVICE
    CopyGmToL1() {};

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0));
        uint32_t blockLen = RoundUp<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(1));

        AscendC::DataCopyParams repeatParams;

        if (layoutSrc.stride(1) / ELE_NUM_PER_C0 < STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = 0;
            AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                uint64_t dstOffset = i * layoutDst.stride(1);
                uint64_t srcOffset = i * layoutSrc.stride(1);
                AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], repeatParams);
            }
        }
    }
};

/// Partial specialization for AtlasA2, PaddingRowMajor in and zN out.
template <class Element>
struct CopyGmToL1<arch::AtlasA2, gemm::MatmulType<Element, layout::PaddingRowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::PaddingRowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    ASCENDCT_DEVICE
    CopyGmToL1() {};

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.orgShape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = layoutSrc.orgShape(0);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
template <
    class Element
>
struct CopyGmToL1<arch::AtlasA2, gemm::MatmulType<Element, layout::PaddingColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::PaddingColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    ASCENDCT_DEVICE
    CopyGmToL1() {};

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.orgShape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = layoutSrc.orgShape(1);
        intriParams.srcDValue = layoutSrc.stride(2);
        intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class Element>
struct CopyGmToL1<arch::AtlasA2, gemm::MatmulType<Element, layout::RowMajor>,
    gemm::MatmulType<Element, layout::RowMajor>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
    static constexpr uint32_t MAX_REPEAT = 4095;

    // Mehtods

    ASCENDCT_DEVICE
    CopyGmToL1() {};

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t rows = layoutSrc.shape(0);
        uint32_t cols = layoutSrc.shape(1);
        uint32_t srcStride = (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK;
        uint32_t dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;

        if ((layoutSrc.shape(1) == layoutSrc.stride(0)) && (layoutDst.shape(1) == layoutDst.stride(0))) {
            DataCopy(dstTensor, srcTensor, rows * cols);
        } else if (srcStride < STRIDE_LIMIT && dstStride < STRIDE_LIMIT && (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
            uint32_t rLoops = CeilDiv(rows, MAX_REPEAT);
            for (uint32_t i = 0; i < rLoops; ++i) {
                uint32_t rActual = (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
                AscendC::DataCopyParams dataCopyParams(
                    rActual, cols / ELE_NUM_PER_BLK, srcStride, dstStride
                );
                DataCopy(dstTensor[i * MAX_REPEAT * layoutDst.stride(0)],
                         srcTensor[i * MAX_REPEAT * layoutSrc.stride(0)], dataCopyParams);
            }
        } else {
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * layoutSrc.stride(0)], cols);
            }
        }
    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////
/// Partial specialization for CopyGmToL1, AtlasA2, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<arch::AtlasA2, Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc_>::value &&
                     tla::detail::iszN<ElementDst, LayoutDst_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::A1>;
    using TensorSrc = Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::GM>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    ASCENDCT_DEVICE
    TileCopyTla() {};

    ASCENDCT_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        const uint32_t nValue = get<0>(srcTensor.shape());
        const uint32_t dValue = get<1>(srcTensor.shape());
        const uint32_t srcDValue = get<0>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (srcDValue < STRIDE_LIMIT) {
            intriParams.nValue = nValue;
            intriParams.srcDValue = srcDValue;
            intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor.data(), srcTensor.data(), intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < nValue; i++) {
                AscendC::DataCopy(dstTensor.data()[i * ELE_NUM_PER_C0], srcTensor.data()[i * srcDValue], intriParams);
            }
        }
    }
};

/// Partial specialization for CopyGmToL1, AtlasA2, ColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<arch::AtlasA2, Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc_>::value &&
                     tla::detail::isnZ<ElementDst, LayoutDst_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::A1>;
    using TensorSrc = Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::GM>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    ASCENDCT_DEVICE
    TileCopyTla() {};

    ASCENDCT_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        const uint32_t nValue = get<1>(srcTensor.shape());
        const uint32_t dValue = get<0>(srcTensor.shape());
        const uint32_t srcDValue = get<1>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = get<1, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (srcDValue < STRIDE_LIMIT) {
            intriParams.nValue = nValue;
            intriParams.srcDValue = srcDValue;
            intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor.data(), srcTensor.data(), intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < nValue; i++) {
                AscendC::DataCopy(dstTensor.data()[i * ELE_NUM_PER_C0], srcTensor.data()[i * srcDValue], intriParams);
            }
        }
    }
};

/// Partial specialization for CopyGmToL1, AtlasA2, PaddingRowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTlaExt<arch::AtlasA2, Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::A1>,
    layout::PaddingRowMajor, layout::zN> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::A1>;
    using TensorSrc = Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::GM>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    ASCENDCT_DEVICE
    TileCopyTlaExt() {};

    ASCENDCT_DEVICE
    void operator()(
        TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = get<1>(srcTensor.orgShape());
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = get<0>(srcTensor.orgShape());
        intriParams.srcDValue = get<0, 0>(srcTensor.stride());
        intriParams.dstNzNStride = get<0, 0>(dstTensor.stride()) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor.data(), srcTensor.data(), intriParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA2, PaddingColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTlaExt<arch::AtlasA2, Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::A1>,
    layout::PaddingColumnMajor, layout::nZ> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::A1>;
    using TensorSrc = Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::GM>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    ASCENDCT_DEVICE
    TileCopyTlaExt() {};

    ASCENDCT_DEVICE
    void operator()(
        TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = get<0>(srcTensor.orgShape());
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = get<1>(srcTensor.orgShape());
        intriParams.srcDValue = get<1, 0>(srcTensor.stride());
        intriParams.dstNzNStride = get<1, 0>(dstTensor.stride()) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor.data(), srcTensor.data(), intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace AscendCT::gemm::tile

#endif // ASCENDCT_MATMUL_TILE_COPY_GM_TO_L1_HPP
