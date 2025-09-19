/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV2D_TILE_COPY_L1_TO_L0_HPP
#define CATLASS_CONV2D_TILE_COPY_L1_TO_L0_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/conv2d/conv2d_type.hpp"

namespace Catlass::Conv2d::Tile {

////////////////////////////////
template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

template<class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Catlass::Conv2d::Conv2dType<Element, layout::Fmap, AscendC::TPosition::A1>>{
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::Fmap;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    Conv2dConfigs configs;

    CATLASS_DEVICE
    CopyL1ToL0A(const Conv2dConfigs& configs_) : configs(configs_) {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor, // (ho, wo, cin1, Kh, Kw, C0)
        AscendC::LocalTensor<Element> srcTensor, // (cin1, hi, wi, C0)
        uint8_t* blockPadList,
        uint32_t hiActual,
        uint32_t wiActual,
        uint32_t cin1L0Actual,
        uint32_t howoRound)
    {
        AscendC::LoadData(
            dstTensor,
            srcTensor,
            { /// load3dv2
                blockPadList, // {padLeft, padRight, padTop, padBottom}
                static_cast<uint16_t>(hiActual), // 源操作数 height
                static_cast<uint16_t>(wiActual), // 源操作数 width
                static_cast<uint16_t>(cin1L0Actual * ELE_NUM_PER_C0), // 源操作数的通道数(channelSize为 4, 8, N*16, N*16+4, N*16+8)
                static_cast<uint16_t>(cin1L0Actual * configs.kh() * configs.kw() * ELE_NUM_PER_C0), // 目的操作数Width维度的传输长度(16的倍数)
                static_cast<uint16_t>(howoRound), // 目的操作数height维度的传输长度(16的倍数)
                0, // 目的操作数Width维度的起点
                0, // 目的操作数height维度的起点
                configs.strideW(), configs.strideH(),
                configs.kw(), configs.kh(),
                configs.dilationW(), configs.dilationH(),
                false, // 是否启用转置
                false, // 是否使能small k特性
                (half)(0) // Pad填充值的数值
            }
        );
    }
};

////////////////////////////////////////

template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

template<class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Catlass::Conv2d::Conv2dType<Element, layout::Filter, AscendC::TPosition::A1>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::Filter;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    Conv2dConfigs configs;

    CATLASS_DEVICE
    CopyL1ToL0B(const Conv2dConfigs& configs_) : configs(configs_) {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t cin1L0Actual, uint32_t coutRound)
    {
        // layoutSrc = layoutFilterInL1 shape(cin1L1, kh, kw, coutRound, C0)
        // layoutDst = layoutFilterInL0 shape(cinL0Actual*kh*kw*C0, nPartActual)
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = 1;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < LayoutDst.shape(1); i++) {
            AscendC::LoadData(
                dstTensor[i * layoutDst.stride(1)],
                srcTensor[i * LayoutSrc.shape(3) * ELE_NUM_PER_C0],
                loadDataParams
            );
        }
    }
};

// template<class ArchTag, class Element>
// struct CopyL1ToL0B<ArchTag, Catlass::Conv2d::Conv2dType<Element, layout::Filter, AscendC::TPosition::A1>>{
//     using LayoutDst = layout::nZ;
//     using LayoutSrc = layout::Filter;

//     static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);
//     static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

//     Conv2dConfigs configs;

//     CATLASS_DEVICE
//     CopyL1ToL0B(const Conv2dConfigs& configs_) : configs(configs_) {};

//     CATLASS_DEVICE
//     void operator()(
//         AscendC::LocalTensor<Element> dstTensor,
//         AscendC::LocalTensor<Element> srcTensor,
//         uint32_t cin1L0Actual, uint32_t coutRound)
//     {
//       uint8_t nIters = 
//         (cin1L0Actual * configs.kh() * configs.kw() * coutRound * ELE_NUM_PER_C0) / ELE_NUM_PER_FRACTAL;
        
//       AscendC::LoadData(
//           dstTensor,
//           srcTensor,
//           {
//             0, // startIndex
//             nIters, // 迭代次数
//             1, // srcStride (相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B)
//             0, // 预留参数，配置为0即可
//             0, // dstGap (相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，单位：512B)
//             false, // 不启用转置
//             0 // 预留参数，配置为0即可
//           }
//       );
//     }
// };


/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_L1_TO_L0_HPP
