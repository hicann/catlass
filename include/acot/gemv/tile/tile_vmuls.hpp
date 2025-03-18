/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ACOT_GEMV_TILE_VMULS_HPP
 #define ACOT_GEMV_TILE_VMULS_HPP
 
 #include "acot/acot.hpp"
 #include "acot/layout/layout.hpp"
 
 namespace acot::gemv::tile
 {
 
     template <
         class ArchTag,
         typename Element>
     struct TileVmuls
     {
         static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element); // 32B,一个block的大小
 
         // Mehtods
 
         ACOT_DEVICE
         TileVmuls() {};
 
         ACOT_DEVICE
         void operator()(
             AscendC::LocalTensor<Element> dstTensor,
             AscendC::LocalTensor<Element> srcTensor,
             Element scalar,
             uint32_t len)
         {
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<Element, AscendC::MaskMode::COUNTER>(len);  // 设置counter模式
            // AscendC::UnaryRepeatParams params{1,1,8,8};
             // 连续模式
             AscendC::Muls<Element,false>(
                 dstTensor,
                 srcTensor,
                 scalar,
                 AscendC::MASK_PLACEHOLDER,
                 1,
                 AscendC::UnaryRepeatParams{}
                );
            AscendC::SetMaskNorm();
            AscendC::ResetMask();  // 还原mask值
         }
     };
 } // namespace acot::matmul::tile
 
 #endif // ACOT_MATMUL_TILE_COPY_GM_TO_L1_HPP
 