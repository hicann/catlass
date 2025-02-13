/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ACOT_GEMV_TILE_VEC_COPY_GM_TO_UB_HPP
 #define ACOT_GEMV_TILE_VEC_COPY_GM_TO_UB_HPP
 
 #include "acot/acot.hpp"
 #include "acot/layout/layout.hpp"
 #include "acot/gemv_aiv/gemv_type.hpp"
 
 constexpr uint32_t STRIDE_LIMIT = 65536;
 
 namespace acot::gemv::tile {
 
 template <
     class ArchTag,
     typename Element
 >
 struct VecCopyGmToUB {
     static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);      //32B,一个block的大小
 
     // Mehtods
 
     ACOT_DEVICE
     VecCopyGmToUB() {};
 
     ACOT_DEVICE
     void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        uint32_t const &len
    ) {
        AscendC::DataCopyParams params;
        params.blockCount = 1;  //连续传输数据块个数
        params.blockLen = CeilDiv(len, ELE_NUM_PER_C0);  //每个连续传输数据块长度
        params.srcStride = 0;  //源操作数，相邻连续数据块的间隔
        params.dstStride = 0;  //目的操作数，相邻连续数据块间的间隔
        AscendC::DataCopy(
            dstTensor, 
            srcTensor, 
            params);
    }
 };
 } // namespace acot::matmul::tile
 
 #endif // ACOT_MATMUL_TILE_COPY_GM_TO_L1_HPP
 