/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_BASIC_SYRK_TLA_HPP
#define CATLASS_GEMM_KERNEL_BASIC_SYRK_TLA_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/matrix.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include <cstdint>

namespace Catlass::Gemm::Kernel {

/**
 * @brief Kernel-level SYRK: Y = X * X^T (Ascend950 only).
 *
 * Scheduling rules (lower-triangle only):
 * 1. blockCoord.m() <  blockCoord.n() -> skip
 * 2. blockCoord.m() == blockCoord.n() -> compute & write once
 * 3. blockCoord.m() >  blockCoord.n() -> compute & dual-write (nz2nd + nz2dn)
 *
 */
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class BasicSyrkTla {
public:
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;

    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementX = typename BlockMmad::ElementX;
    using LayoutTagX = typename BlockMmad::LayoutTagX;
    using LayoutX = typename BlockMmad::LayoutX;
    using ElementXt = typename BlockMmad::ElementXt;
    using LayoutTagXt = typename BlockMmad::LayoutTagXt;
    using LayoutXt = typename BlockMmad::LayoutXt;
    using ElementY = typename BlockMmad::ElementY;
    using LayoutTagY = typename BlockMmad::LayoutTagY;
    using LayoutY = typename BlockMmad::LayoutY;
    using LayoutTagYT = layout::ColumnMajor; // ColumnMajor view of the same Y buffer for nz2dn (transpose) store.
    using LayoutYT = detail::TagToLayout_t<ElementY, LayoutTagYT>;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        GemmCoord problemShape; // (M, M, K)
        GM_ADDR ptrX;
        GM_ADDR ptrY;
        LayoutX layoutX;
        LayoutXt layoutXt;
        LayoutY layoutY;
        LayoutYT layoutYT;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, GM_ADDR ptrX_, GM_ADDR ptrY_, LayoutX layoutX_, LayoutXt layoutXt_,
            LayoutY layoutY_, LayoutYT layoutYT_)
            : problemShape(problemShape_),
              ptrX(ptrX_),
              ptrY(ptrY_),
              layoutX(layoutX_),
              layoutXt(layoutXt_),
              layoutY(layoutY_),
              layoutYT(layoutYT_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape; // (M, M, K)
        uint8_t* ptrX;
        uint8_t* ptrY;
    };

    static bool CanImplement(const Arguments& args)
    {
        return args.problemShape.m() == args.problemShape.n();
    }

    static size_t GetWorkspaceSize(const Arguments& /*args*/)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* /*workspace*/)
    {
        uint32_t m = args.problemShape.m();
        uint32_t k = args.problemShape.k();
        return Params{
            args.problemShape,
            args.ptrX,
            args.ptrY,
            tla::MakeLayout<ElementX, LayoutTagX>(m, k),
            tla::MakeLayout<ElementXt, LayoutTagXt>(k, m),
            tla::MakeLayout<ElementY, LayoutTagY>(m, m),
            tla::MakeLayout<ElementY, LayoutTagYT>(m, m),
        };
    }

    CATLASS_DEVICE
    BasicSyrkTla()
    {}

    template <int32_t CoreType_ = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler blockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = blockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX*)params.ptrX);
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY*)params.ptrY);

        auto tensorX = tla::MakeTensor(gmX, params.layoutX, Arch::PositionGM{});
        auto tensorXt = tla::MakeTensor(gmX, params.layoutXt, Arch::PositionGM{});
        auto tensorY = tla::MakeTensor(gmY, params.layoutY, Arch::PositionGM{});
        auto tensorYT = tla::MakeTensor(gmY, params.layoutYT, Arch::PositionGM{});

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            if (blockCoord.m() < blockCoord.n()) {
                continue;
            }

            GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
            auto tileX = tla::GetTile(
                tensorX, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tileXt = tla::GetTile(
                tensorXt, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tileY = tla::GetTile(
                tensorY, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
            auto tileYT = tla::GetTile(
                tensorYT, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            blockMmad(tileX, tileXt, tileY, tileYT, actualBlockShape, blockCoord);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& /*params*/)
    {}
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_BASIC_SYRK_TLA_HPP
