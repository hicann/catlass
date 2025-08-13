/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_SWISH_HPP
#define CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_SWISH_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Tile {
template <
    // / Tag indicating architecture
    class ArchTag_,
    // / Compute data type
    class ComputeType_,
    // / COMPUTE_LENGTH of the compute buffer
    uint32_t COMPUTE_COMPUTE_LENGTH_>
struct TileElemWiseSwish {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_COMPUTE_LENGTH_;

    CATLASS_DEVICE
    TileElemWiseSwish() {}

    CATLASS_DEVICE
    void operator () (AscendC::LocalTensor<ElementCompute> const & dstLocal,
        AscendC::LocalTensor<ElementCompute> const & srcLocal)
    {
        using namespace AscendC;
        if constexpr (!std::is_same_v<ElementCompute, float32_t>) {
            TPipe pipe;
            TBuf<QuePosition::VECCALC> tmp1;
            TBuf<QuePosition::VECCALC> tmp2;
            pipe.InitBuffer(tmp1, COMPUTE_LENGTH * sizeof(float));
            pipe.InitBuffer(tmp2, COMPUTE_LENGTH * sizeof(float));
            LocalTensor<float> p1 = tmp1.Get<float>();
            LocalTensor<float> p2 = tmp2.Get<float>();
            Cast(p1, srcLocal, RoundMode::CAST_NONE, COMPUTE_LENGTH);
            Cast(p2, srcLocal, RoundMode::CAST_NONE, COMPUTE_LENGTH);
            Muls(p2, p1, (float)-1, COMPUTE_LENGTH);
            Exp(p2, p2, COMPUTE_LENGTH);
            Adds(p2, p2, (float)1, COMPUTE_LENGTH);
            Div(p2, p1, p2, COMPUTE_LENGTH);
            Cast(dstLocal, p2, RoundMode::CAST_RINT, COMPUTE_LENGTH);
            tmp1.FreeTensor(p1);
            tmp2.FreeTensor(p2);
        } else {
            Mul(dstLocal, srcLocal, (ComputeType_)-1, COMPUTE_LENGTH);
            Exp(dstLocal, dstLocal, COMPUTE_LENGTH);
            Adds(dstLocal, dstLocal, (ComputeType_)1, COMPUTE_LENGTH);
            Div(dstLocal, srcLocal, dstLocal, COMPUTE_LENGTH);
        }
    }
};
} // namespace Catlass::Epilogue::Tile

#endif
