/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMM_TILE_TILE_MMAD_HPP
#define ACOT_GEMM_TILE_TILE_MMAD_HPP

#include "acot/acot.hpp"
#include "acot/gemm/helper.hpp"

namespace acot::gemm::tile{
template<
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class BiasType = void
>
struct TileMmad{
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    ACOT_DEVICE
    TileMmad() {}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementAccumulator> dstTensor,
        AscendC::LocalTensor<ElementA> src0Tensor,
        AscendC::LocalTensor<ElementB> src1Tensor,
        uint32_t m,
        uint32_t n,
        uint32_t k,
        bool isFirst = true,
        uint8_t unitFlag = 0
    ){
        AscendC::MmadParams params;
        params.m = m;
        params.n = n;
        params.k = k;
        params.cmatrixInitVal = isFirst;
        params.unitFlag = unitFlag;
        params.cmatrixSource = false;
        AscendC::Mmad(
            dstTensor,
            src0Tensor,
            src1Tensor,
            params
        );
        
        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD)
        {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};
}

#endif // ACOT_GEMM_TILE_TILE_MMAD_HPP