/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMM_TILE_COPY_L1_TO_L0_HPP
#define ACOT_GEMM_TILE_COPY_L1_TO_L0_HPP

#include "acot/acot.hpp"
#include "acot/layout/layout.hpp"
#include "acot/matmul/matmul_type.hpp"

namespace acot::gemm::tile{
template<
    class ArchTag,
    class GmType
>
struct CopyL1ToL0A{};

template<class Element>
struct CopyL1ToL0A<acot::arch::AtlasA2, acot::matmul::MatmulType<Element, layout::RowMajor>>{
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);

    ACOT_DEVICE
    CopyL1ToL0A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t MRound = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t MLoops = CeilDiv(MRound, C0_NUM_PER_FRACTAL);
        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(KRound / ELE_NUM_PER_C0);
        params.srcStride = MRound / C0_NUM_PER_FRACTAL;
        params.sid = 0;
        params.dstGap = 0;
        params.ifTranspose = false;
        params.addrMode = 0;
        for(uint32_t i = 0; i < MLoops; i++){
            AscendC::LoadData(dstTensor[i * C0_NUM_PER_FRACTAL * KRound], srcTensor[i * ELE_NUM_PER_C0 * C0_NUM_PER_FRACTAL], params);
        }
    }
};

template<class Element>
struct CopyL1ToL0A<acot::arch::AtlasA2, acot::matmul::MatmulType<Element, layout::ColumnMajor>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);

    ACOT_DEVICE
    CopyL1ToL0A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t MRound = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t KLoops = CeilDiv(KRound, C0_NUM_PER_FRACTAL);
        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(MRound / ELE_NUM_PER_C0); 
        params.srcStride = 1;
        params.sid = 0;
        params.dstGap = 0;
        params.ifTranspose = true;
        params.addrMode = 0;
        for(uint32_t i = 0; i < KLoops; i++){
            AscendC::LoadData(dstTensor[i * C0_NUM_PER_FRACTAL * MRound], srcTensor[i * C0_NUM_PER_FRACTAL * MRound], params);
        }
    }
};

template<>
struct CopyL1ToL0A<acot::arch::AtlasA2, acot::matmul::MatmulType<float, layout::ColumnMajor>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(float);

    ACOT_DEVICE
    CopyL1ToL0A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<float> dstTensor,
        AscendC::LocalTensor<float> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t MRound = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t ML0Alignment = ELE_NUM_PER_C0 * 2;
        uint32_t KLoops = CeilDiv(KRound, C0_NUM_PER_FRACTAL);
        AscendC::LoadData2dTransposeParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(MRound / ML0Alignment); 
        params.srcStride = 1;
        params.dstGap = 0;
        params.dstFracGap = static_cast<uint16_t>(MRound / ML0Alignment) - 1;
        for(uint32_t i = 0; i < KLoops; i++){ 
            AscendC::LoadDataWithTranspose(dstTensor[i * MRound * C0_NUM_PER_FRACTAL], srcTensor[i * C0_NUM_PER_FRACTAL * MRound], params);
        }
    }
};

template<>
struct CopyL1ToL0A<acot::arch::AtlasA2, acot::matmul::MatmulType<int8_t, layout::ColumnMajor>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(int8_t);

    ACOT_DEVICE
    CopyL1ToL0A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> dstTensor,
        AscendC::LocalTensor<int8_t> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t MRound = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint32_t KRound = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint32_t KL0Alignment = C0_NUM_PER_FRACTAL * 2; 
        uint32_t KLoops = CeilDiv(KRound, KL0Alignment);
        AscendC::LoadData2dTransposeParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(MRound / ELE_NUM_PER_C0); 
        params.srcStride = static_cast<uint16_t>(KRound / KL0Alignment); 
        params.dstGap = 1; 
        params.dstFracGap = 0;
        for(uint32_t i = 0; i < KLoops; i++){
            AscendC::LoadDataWithTranspose(dstTensor[i * MRound * KL0Alignment], srcTensor[i * KL0Alignment * ELE_NUM_PER_C0], params);
        }
    }
};

template<
    class ArchTag,
    class GmType
>
struct CopyL1ToL0B{};

template<class Element>
struct CopyL1ToL0B<acot::arch::AtlasA2, acot::matmul::MatmulType<Element, layout::RowMajor>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);

    ACOT_DEVICE
    CopyL1ToL0B(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t NRound = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t KLoops = CeilDiv(KRound, C0_NUM_PER_FRACTAL);
        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(NRound / ELE_NUM_PER_C0);
        params.srcStride = 1;
        params.sid = 0;
        params.dstGap = 0;
        params.ifTranspose = true;
        params.addrMode = 0;
        for(uint32_t i = 0; i < KLoops; i++){ 
            AscendC::LoadData(dstTensor[i * NRound * C0_NUM_PER_FRACTAL], srcTensor[i * NRound * C0_NUM_PER_FRACTAL], params);
        }
    }
};

template<>
struct CopyL1ToL0B<acot::arch::AtlasA2, acot::matmul::MatmulType<float, layout::RowMajor>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(float);

    ACOT_DEVICE
    CopyL1ToL0B(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<float> dstTensor,
        AscendC::LocalTensor<float> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t NRound = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t NL0Alignment = ELE_NUM_PER_C0 * 2; 
        uint32_t KLoops = CeilDiv(KRound, C0_NUM_PER_FRACTAL); 
        AscendC::LoadData2dTransposeParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(NRound / NL0Alignment); 
        params.srcStride = 1;
        params.dstGap = 0;
        params.dstFracGap = static_cast<uint16_t>(NRound / NL0Alignment) - 1;
        for(uint32_t i = 0; i < KLoops; i++){
            AscendC::LoadDataWithTranspose(dstTensor[i * NRound * C0_NUM_PER_FRACTAL], srcTensor[i * C0_NUM_PER_FRACTAL * NRound], params);
        }
    }
};

template<>
struct CopyL1ToL0B<acot::arch::AtlasA2, acot::matmul::MatmulType<int8_t, layout::RowMajor>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(int8_t);

    ACOT_DEVICE
    CopyL1ToL0B(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> dstTensor,
        AscendC::LocalTensor<int8_t> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t NRound = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint32_t KRound = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint32_t KL0Alignment = C0_NUM_PER_FRACTAL * 2; 
        uint32_t KLoops = CeilDiv(KRound, KL0Alignment);
        AscendC::LoadData2dTransposeParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(NRound / ELE_NUM_PER_C0); 
        params.srcStride = static_cast<uint16_t>(KRound / KL0Alignment);
        params.dstGap = 1; // 单位为512B
        params.dstFracGap = 0;
        for(uint32_t i = 0; i < KLoops; i++){
            AscendC::LoadDataWithTranspose(dstTensor[i * NRound * KL0Alignment], srcTensor[i * KL0Alignment * ELE_NUM_PER_C0], params);
        }
    }
};

template<class Element>
struct CopyL1ToL0B<acot::arch::AtlasA2, acot::matmul::MatmulType<Element, layout::ColumnMajor>>{
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);

    ACOT_DEVICE
    CopyL1ToL0B(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t NRound = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t NLoops = CeilDiv(NRound, C0_NUM_PER_FRACTAL);
        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(KRound / ELE_NUM_PER_C0);
        params.srcStride = NRound / C0_NUM_PER_FRACTAL;
        params.sid = 0;
        params.dstGap = 0;
        params.ifTranspose = false;
        params.addrMode = 0;
        for(uint32_t i = 0; i < NLoops; i++){
            AscendC::LoadData(dstTensor[i * C0_NUM_PER_FRACTAL * KRound], srcTensor[i * C0_NUM_PER_FRACTAL * ELE_NUM_PER_C0], params);
        }
    }
};
}

#endif // ACOT_GEMM_TILE_COPY_L1_TO_L0_HPP