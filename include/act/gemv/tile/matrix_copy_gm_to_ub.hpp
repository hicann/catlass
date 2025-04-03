/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ACT_GEMV_TILE_MATRIX_COPY_GM_TO_UB_HPP
 #define ACT_GEMV_TILE_MATRIX_COPY_GM_TO_UB_HPP
 
 #include "act/act.hpp"
 #include "act/layout/layout.hpp"
 #include "act/gemm/gemm_type.hpp"
 
 namespace Act::Gemv::Tile
 {
 
     template <
         class ArchTag,
         class GmType
         >
 
     struct MatrixCopyGmToUB
     {
         static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to UB, can not find the specialization.");
     };
 
     template <class Element>
     struct MatrixCopyGmToUB<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>>
     {
         using LayoutDst = layout::RowMajor;
         using LayoutSrc = layout::RowMajor;
 
         static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element); 
 
         // Mehtods
 
         ACT_DEVICE
         MatrixCopyGmToUB() {};
 
         ACT_DEVICE
         void operator()(
             AscendC::LocalTensor<Element> dstTensor,
             AscendC::GlobalTensor<Element> srcTensor,
             LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
         {
             uint32_t m_actual = layoutSrc.shape(0);
             uint32_t n_actual = layoutSrc.shape(1);
             uint32_t m_round = layoutDst.shape(0);
             uint32_t n_round = layoutDst.shape(1);
             uint32_t stride = layoutSrc.stride(0);
 
             AscendC::DataCopyParams params;
             if ((n_actual % ELE_NUM_PER_C0 == 0) && (stride % ELE_NUM_PER_C0 == 0) && (stride < STRIDE_LIMIT))
             {
                 params.blockCount = m_actual;
                 params.blockLen = CeilDiv(n_actual, ELE_NUM_PER_C0);
                 params.srcStride = (stride - n_actual) / ELE_NUM_PER_C0;
                 params.dstStride = (n_round - n_actual) / ELE_NUM_PER_C0;
                 AscendC::DataCopy(
                     dstTensor,
                     srcTensor,
                     params);
             }
             else if ((n_actual % ELE_NUM_PER_C0 == 0) && (stride * ELE_NUM_PER_C0 < STRIDE_LIMIT))
             {
                 uint32_t counts = m_actual / ELE_NUM_PER_C0;
                 uint32_t remain = m_actual % ELE_NUM_PER_C0;
                 if (counts > 0)
                 {
                     params.blockCount = counts;
                     params.blockLen = CeilDiv(n_actual, ELE_NUM_PER_C0);
                     params.srcStride = (ELE_NUM_PER_C0 * stride - n_actual) / ELE_NUM_PER_C0;
                     params.dstStride = (ELE_NUM_PER_C0 * n_round - n_actual) / ELE_NUM_PER_C0;
                     for (uint32_t i = 0; i < ELE_NUM_PER_C0; i++)
                     {
                         AscendC::DataCopy(
                             dstTensor[i * n_round],
                             srcTensor[i * stride],
                             params);
                     }
                 }
                 if (remain > 0)
                 {
                     params.blockCount = 1;
                     params.blockLen = CeilDiv(n_actual, ELE_NUM_PER_C0);
                     params.srcStride = 0;
                     params.dstStride = 0;
                     for (uint32_t i = 0; i < remain; i++)
                     {
                         AscendC::DataCopy(
                             dstTensor[counts * n_round * ELE_NUM_PER_C0 + i * n_round],
                             srcTensor[counts * stride * ELE_NUM_PER_C0 + i * stride],
                             params);
                     }
                 }
             }
             else
             {
                 params.blockCount = 1;
                 params.blockLen = CeilDiv(n_actual, ELE_NUM_PER_C0);
                 params.srcStride = 0;
                 params.dstStride = 0;
                 for (uint32_t i = 0; i < m_actual; i++)
                 {
                     AscendC::DataCopy(
                         dstTensor[i * n_round],
                         srcTensor[i * stride],
                         params);
                 }
             }
         }
     };
 
     template <class Element>
     struct MatrixCopyGmToUB<Arch::AtlasA2, Gemm::GemmType<Element, layout::ColumnMajor>>
     {
         using LayoutDst = layout::ColumnMajor;
         using LayoutSrc = layout::ColumnMajor;
 
         static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
 
         // Mehtods
 
         ACT_DEVICE
         MatrixCopyGmToUB() {};
 
         ACT_DEVICE
         void operator()(
             AscendC::LocalTensor<Element> const &dstTensor,
             AscendC::GlobalTensor<Element> const &srcTensor,
             LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
         {
             uint32_t m_actual = layoutSrc.shape(0);
             uint32_t n_actual = layoutSrc.shape(1);
             uint32_t m_round = layoutDst.shape(0);
             uint32_t n_round = layoutDst.shape(1);
             uint32_t stride = layoutSrc.stride(1); 
 
             AscendC::DataCopyParams params;
             if ((m_actual % ELE_NUM_PER_C0 == 0) && (stride % ELE_NUM_PER_C0 == 0) && (stride < STRIDE_LIMIT))
             {
                 params.blockCount = n_actual;
                 params.blockLen = CeilDiv(m_actual, ELE_NUM_PER_C0);
                 params.srcStride = (stride - m_actual) / ELE_NUM_PER_C0;
                 params.dstStride = (m_round - m_actual) / ELE_NUM_PER_C0;
                 AscendC::DataCopy(
                     dstTensor,
                     srcTensor,
                     params);
             }
             else if ((m_actual % ELE_NUM_PER_C0 == 0) && (stride * ELE_NUM_PER_C0 < STRIDE_LIMIT))
             {
                 uint32_t counts = n_actual / ELE_NUM_PER_C0;
                 uint32_t remain = n_actual % ELE_NUM_PER_C0;
                 if (counts > 0)
                 {
                     params.blockCount = counts;
                     params.blockLen = CeilDiv(m_actual, ELE_NUM_PER_C0);
                     params.srcStride = (ELE_NUM_PER_C0 * stride - m_actual) / ELE_NUM_PER_C0;
                     params.dstStride = (ELE_NUM_PER_C0 * m_round - m_actual) / ELE_NUM_PER_C0;
                     for (uint32_t i = 0; i < ELE_NUM_PER_C0; i++)
                     {
                         AscendC::DataCopy(
                             dstTensor[i * m_round],
                             srcTensor[i * stride],
                             params);
                     }
                 }
                 if (remain > 0)
                 {
                     params.blockCount = 1;
                     params.blockLen = CeilDiv(m_actual, ELE_NUM_PER_C0);
                     params.srcStride = 0;
                     params.dstStride = 0;
                     for (uint32_t i = 0; i < remain; i++)
                     {
                         AscendC::DataCopy(
                             dstTensor[counts * m_round * ELE_NUM_PER_C0 + i * m_round],
                             srcTensor[counts * stride * ELE_NUM_PER_C0 + i * stride],
                             params);
                     }
                 }
             }
             else
             {
                 params.blockCount = 1;
                 params.blockLen = CeilDiv(m_actual, ELE_NUM_PER_C0);
                 params.srcStride = 0;
                 params.dstStride = 0;
                 for (uint32_t i = 0; i < n_actual; i++)
                 {
                     AscendC::DataCopy(
                         dstTensor[i * m_round],
                         srcTensor[i * stride],
                         params);
                 }
             }
         }
     };
 
 } // namespace Act::Gemv::Tile
 
 #endif // ACT_GEMV_TILE_MATRIX_COPY_GM_TO_UB_HPP
 