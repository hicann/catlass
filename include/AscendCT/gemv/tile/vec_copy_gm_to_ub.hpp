/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ASCENDCT_GEMV_TILE_VEC_COPY_GM_TO_UB_HPP
 #define ASCENDCT_GEMV_TILE_VEC_COPY_GM_TO_UB_HPP
 
 #include "AscendCT/AscendCT.hpp"
 #include "AscendCT/layout/layout.hpp"
 #include "AscendCT/gemm/matmul_type.hpp"
 constexpr uint32_t STRIDE_LIMIT = 65536;
 
 namespace AscendCT::gemv::tile {
 
 template <
     class ArchTag_,
     class VType_
 >
 struct VecCopyGmToUB {
     using Element = typename VType_::Element;
     static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
 
     // Mehtods
 
     ASCENDCT_DEVICE
     VecCopyGmToUB() {};
 
     ASCENDCT_DEVICE
     void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        uint32_t len
    ) {
        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = CeilDiv(len, ELE_NUM_PER_C0); 
        params.srcStride = 0; 
        params.dstStride = 0;  
        AscendC::DataCopy(
            dstTensor, 
            srcTensor, 
            params);
    }
 };
 } // namespace AscendCT::gemv::tile
 
 #endif // ASCENDCT_GEMV_TILE_VEC_COPY_GM_TO_UB_HPP
 