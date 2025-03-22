/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDCT_EPILOGUE_TILE_TILE_ELEMWISE_ADD_HPP
#define ASCENDCT_EPILOGUE_TILE_TILE_ELEMWISE_ADD_HPP

#include "AscendCT/AscendCT.hpp"

namespace AscendCT::epilogue::tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class ComputeType_,
    /// Length of the compute buffer
    uint32_t COMPUTE_LENGTH_
>
struct TileElemWiseAdd {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    ASCENDCT_DEVICE
    TileElemWiseAdd() {}

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementCompute> const &ubOut,
        AscendC::LocalTensor<ElementCompute> const &ubIn0,
        AscendC::LocalTensor<ElementCompute> const &ubIn1
    )
    {
        // Do the calculation
        AscendC::Add(ubOut, ubIn0, ubIn1, COMPUTE_LENGTH);
    }
};

} // namespace AscendCT::epilogue::tile

#endif
