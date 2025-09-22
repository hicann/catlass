/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_DYNAMIC_PADDING_MUTI_CORE_SPLITK_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_DYNAMIC_PADDING_MUTI_CORE_SPLITK_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"
#include "catlass/gemm/kernel/splitk_matmul.hpp"

namespace Catlass::Gemm::Kernel {

template <class PrologueA_, class PrologueB_, class BlockMmad_, 
    class BlockEpilogue_, class BlockScheduler_, class ReduceAdd_>
class DynamicPaddingMutiCoreSplitkMatmul {
public:
    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;
    using ReduceAdd = ReduceAdd_;

    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;

    using LayoutC = typename BlockMmad::LayoutC;

    template<class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template<>
    struct LayoutHelper<void> {
        using type = void;
    };

    using LayoutA = std::conditional_t<
        std::is_void_v<PrologueA>, typename BlockMmad::LayoutA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<
        std::is_void_v<PrologueB>, typename BlockMmad::LayoutB, typename LayoutHelper<PrologueB>::type>;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GemmCoord l1TileShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWA;
        GM_ADDR ptrWB;
        GM_ADDR ptrReduceW;
        uint32_t splitkFactor = 1;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GemmCoord const &l1TileShape_, GM_ADDR ptrA_, LayoutA& layoutA_,
            GM_ADDR ptrB_, LayoutB& layoutB_, GM_ADDR ptrC_, LayoutC& layoutC_, GM_ADDR ptrWA_, GM_ADDR ptrWB_,
            GM_ADDR ptrReduceW_, uint32_t splitkFactor_)
            : problemShape(problemShape_), l1TileShape(l1TileShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_),
              layoutB(layoutB_), ptrC(ptrC_), layoutC(layoutC_), ptrWA(ptrWA_), ptrWB(ptrWB_), ptrReduceW(ptrReduceW_),
              splitkFactor(splitkFactor_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    DynamicPaddingMutiCoreSplitkMatmul()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> &resource);

    template<>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
         if constexpr (!std::is_void_v<PrologueA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            typename BlockMmad::LayoutA layoutWA;
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }
            PrologueA prologueA(resource);
            prologueA(gmWA, gmA, layoutWA, params.layoutA);
        }

        if constexpr (!std::is_void_v<PrologueB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            typename BlockMmad::LayoutB layoutWB;
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }
            PrologueB prologueB(resource);
            prologueB(gmWB, gmB, layoutWB, params.layoutB);
            // 0x0 synchronization control between AI Core
        }
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        using ElementOut = typename ReduceAdd::ElementOut;
        using ElementAccumulator = typename ReduceAdd::ElementAccumulator;

        Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

        AscendC::GlobalTensor<ElementOut> gmC;
        AscendC::GlobalTensor<ElementAccumulator> gmReduceW;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementOut *>(params.ptrC));
        gmReduceW.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator *>(params.ptrReduceW));
        ReduceAdd reduceAdd(resource);
        reduceAdd( gmC, gmReduceW,
            static_cast<uint64_t>(params.problemShape.m()) * params.problemShape.n(), params.splitkFactor);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// Executes matmul
    template<>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        BlockScheduler matmulBlockScheduler(params.problemShape, params.l1TileShape, params.splitkFactor);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        typename BlockMmad::LayoutA layoutA;
        typename BlockMmad::LayoutB layoutB;
        typename BlockMmad::LayoutC layoutC;

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        if constexpr (std::is_void_v<PrologueA>) {
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
            layoutA = params.layoutA;
        } else {
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }
        }
        AscendC::GlobalTensor<ElementB> gmB;
        if constexpr (std::is_void_v<PrologueB>) {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
            layoutB = params.layoutB;
        } else {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }
        }
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrReduceW);

        BlockMmad blockMmad(params.l1TileShape, resource);

        GemmCoord blockCoord;
        GemmCoord actualBlockShape;
        GemmCoord nextBlockCoord;
        GemmCoord nextActualBlockShape;

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            if (isFirstBlock) {
                blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                actualBlockShape = matmulBlockScheduler.GetActualBlockShape(
                    blockCoord, matmulBlockScheduler.GetSplitkSliceIdx(loopIdx));
            } else {
                blockCoord = nextBlockCoord;
                actualBlockShape = nextActualBlockShape;
            }

            bool hasNextBlock = false;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                nextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord,
                    matmulBlockScheduler.GetSplitkSliceIdx(loopIdx + AscendC::GetBlockNum()));
            }

            // Compute initial location in logical coordinates
            MatrixCoord coordA{blockCoord.m() * params.l1TileShape.m(), blockCoord.k() * params.l1TileShape.k()};
            MatrixCoord coordB{blockCoord.k() * params.l1TileShape.k(), blockCoord.n() * params.l1TileShape.n()};
            MatrixCoord coordC{blockCoord.m() * params.l1TileShape.m(), blockCoord.n() * params.l1TileShape.n()};
            int64_t gmOffsetA = layoutA.GetOffset(coordA);
            int64_t gmOffsetB = layoutB.GetOffset(coordB);
            int64_t gmOffsetC = params.layoutC.GetOffset(coordC)
                + static_cast<uint64_t>(params.problemShape.m()) * params.problemShape.n() 
                * matmulBlockScheduler.GetSplitkSliceIdx(loopIdx);

            MatrixCoord coordNextA{
                nextBlockCoord.m() * params.l1TileShape.m(), nextBlockCoord.k() * params.l1TileShape.k()};
            MatrixCoord coordNextB{
                nextBlockCoord.k() * params.l1TileShape.k(), nextBlockCoord.n() * params.l1TileShape.n()};
            int64_t gmOffsetNextA = layoutA.GetOffset(coordNextA);
            int64_t gmOffsetNextB = layoutB.GetOffset(coordNextB);

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA],
                layoutA,
                gmB[gmOffsetB],
                layoutB,
                gmC[gmOffsetC],
                params.layoutC,
                gmA[gmOffsetNextA],
                gmB[gmOffsetNextB],
                actualBlockShape,
                nextActualBlockShape,
                isFirstBlock,
                hasNextBlock);
        }

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 1;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_DYNAMIC_PADDING_MUTI_CORE_SPLITK_MATMUL_HPP