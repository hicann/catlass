/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMV_TILE_TILE_MMAD_HPP
#define ACOT_GEMV_TILE_TILE_MMAD_HPP

#include "acot/acot.hpp"
#include "acot/gemv/helper.hpp"

namespace acot::gemv::tile
{

    ///////////////////////////////////////////////////////////
    template <
        /// Tag indicating architecture
        class ArchTag_,
        /// MatmulType for A matrix operand
        class AType_,
        /// MatmulType type for B matrix operand
        class BType_,
        /// MatmulType type for Bias operand
        class BiasType_>
    struct TileMmad
    {
        using ElementA = typename AType_::Element;
        using ElementB = typename BType_::Element;
        using ElementAccumulator =
            typename gemv::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

        // Methods

        ACOT_DEVICE
        TileMmad() {}

        ACOT_DEVICE
        void operator()(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
                        AscendC::LocalTensor<ElementA> const &l0ATensor,
                        AscendC::LocalTensor<ElementB> const &l0BTensor,
                        uint32_t m, uint32_t n, uint32_t k,
                        bool initC = true, uint8_t unitFlag = 0)
        {
            AscendC::MmadParams mmadParams;
            mmadParams.m = m;
            mmadParams.n = n;
            mmadParams.k = k;
            mmadParams.unitFlag = unitFlag; //文档说不管这个参数
            mmadParams.cmatrixInitVal = initC;

            AscendC::Mmad(l0CTensor,
                          l0ATensor,
                          l0BTensor,
                          mmadParams);

            const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
            if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD)
            {
                AscendC::PipeBarrier<PIPE_M>();
            }
        }
    };
} // namespace acot::gemv::tile

#endif // ACOT_GEMV_TILE_TILE_MMAD_HPP