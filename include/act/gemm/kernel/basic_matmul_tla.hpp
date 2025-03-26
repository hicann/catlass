/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACT_GEMM_KERNEL_MATMUL_TLA_HPP
#define ACT_GEMM_KERNEL_MATMUL_TLA_HPP

#include "act/act.hpp"
#include "act/arch/resource.hpp"
#include "act/coord.hpp"
#include "act/gemm_coord.hpp"
#include "act/matrix_coord.hpp"
#include "tla/tensor.hpp"

namespace Act::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class BasicMatmulTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = get<2>(L1TileShape{});

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;

        // Methods
        ACT_DEVICE
        Params() {}

        ACT_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_,
               LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_) {}
    };

    // Methods
    ACT_DEVICE
    BasicMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    ACT_DEVICE
    void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    ACT_DEVICE
    void operator()<AscendC::AIC>(Params const &params) {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        // Represent the full tensors
        auto tensorA = MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto tensorC = MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Make tiled views
            auto tensorBlockA = GetTile(tensorA,
                                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                                        MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tensorBlockB = GetTile(tensorB,
                                        tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                                        MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockC = GetTile(tensorC,
                                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                                        MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            // Compute block-scoped matrix multiply-add
            blockMmad(tensorBlockA, tensorBlockB, tensorBlockC);
        }
    }

    template <>
    ACT_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {}
};

} // namespace Act::Gemm::Kernel

#endif // ACT_GEMM_KERNEL_MATMUL_TLA_HPP