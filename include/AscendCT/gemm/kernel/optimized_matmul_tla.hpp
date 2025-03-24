/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDCT_MATMUL_KERNEL_OPTIMIZED_MATMUL_TLA_HPP
#define ASCENDCT_MATMUL_KERNEL_OPTIMIZED_MATMUL_TLA_HPP

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/coord.hpp"
#include "AscendCT/gemm_coord.hpp"
#include "AscendCT/matrix_coord.hpp"
#include "AscendCT/arch/resource.hpp"
#include "AscendCT/arch/cross_core_sync.hpp"

namespace AscendCT::gemm::kernel {

template<
    class ArchTag_,
    class TensorIn_,
    class TensorOut_,
    uint32_t COMPUTE_LENGTH
>
struct PaddingMatrixBlockND {
public:
    using ArchTag = ArchTag_;
    using TensorIn = TensorIn_;
    using TensorOut = TensorOut_;
    using Element = typename TensorIn::Element;
    using LayoutIn = typename TensorIn::Layout;
    using LayoutOut = typename TensorOut::Layout;

    using LayoutInner = Layout<Shape<uint32_t, uint32_t>, Stride<int64_t, Int<1>>, Shape<uint32_t, uint32_t>>;
    using TensorInnerUb = Tensor<AscendC::LocalTensor<Element>, LayoutInner, AscendC::TPosition::VECCALC>;
    using TensorInnerSrcGm = Tensor<AscendC::GlobalTensor<Element>, LayoutInner, AscendC::TPosition::GM>;

    using LayoutInnerDstGm = Layout<
        Shape<Shape<uint32_t, uint32_t>, Shape<uint32_t, uint32_t>>,
        Stride<Stride<int64_t, int64_t>, Stride<Int<1>, int64_t>>,
        Shape<uint32_t, uint32_t>>;
    using TensorInnerDstGm = Tensor<AscendC::GlobalTensor<Element>, LayoutInnerDstGm, AscendC::TPosition::GM>;

    using CopyGm2Ub = AscendCT::gemm::tile::TileCopyTla<ArchTag, TensorInnerSrcGm, TensorInnerUb>;
    using CopyUb2Gm = AscendCT::gemm::tile::TileCopyTlaExt<ArchTag, TensorInnerUb,
        TensorInnerDstGm, layout::RowMajor, layout::PaddingRowMajor>;

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    ASCENDCT_DEVICE
    PaddingMatrixBlockND(arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    template<class Tensor>
    ASCENDCT_DEVICE
    auto GetPaddingTensorSrc(Tensor const &tensor)
    {
        if constexpr (std::is_same_v<typename Tensor::Layout, LayoutInner>) {
            return tensor;
        } else {
            auto shape = MakeShape(get<1>(tensor.shape()), get<0>(tensor.shape()));
            auto stride = MakeStride(get<1>(tensor.stride()), get<0>(tensor.stride()));
            return MakeTensor(tensor.data(), MakeLayout(shape, stride), arch::PositionGM{});
        }
    }

    template<class Tensor>
    ASCENDCT_DEVICE
    auto GetPaddingTensorDst(Tensor const &tensor)
    {
        if constexpr (std::is_same_v<typename Tensor::Layout, LayoutInnerDstGm>) {
            return tensor;
        } else {
            auto shape = MakeShape(get<1>(tensor.shape()), get<0>(tensor.shape()));
            auto stride = MakeStride(get<1>(tensor.stride()), get<0>(tensor.stride()));
            return MakeTensor(tensor.data(), MakeLayout(shape, stride), arch::PositionGM{});
        }
    }

    ASCENDCT_DEVICE
    void operator()(TensorOut &tensorDst, TensorIn const& tensorSrc)
    {
        auto paddingTensorSrc = GetPaddingTensorSrc(tensorSrc);
        auto paddingTensorDst = GetPaddingTensorDst(tensorDst);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        // Each line is a tile.
        uint32_t tilesNum = get<0>(paddingTensorSrc.shape());
        uint32_t tileLen = get<1>(paddingTensorSrc.shape());
        uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(Element)>(get<1>(paddingTensorSrc.shape()));

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
        uint32_t coreLoops{ 0 };
        if (roundTileLen > COMPUTE_LENGTH) {
            // Handle the same tile on multiple loops.
            uint32_t loopsPerTile = (tileLen + COMPUTE_LENGTH - 1) / COMPUTE_LENGTH;
            coreLoops = tilesPerAiv * loopsPerTile;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx / loopsPerTile;
                uint32_t inTileLoopIdx = loopIdx % loopsPerTile;
                auto offset = tla::MakeCoord(mIdx + tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) {
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                auto tensorTileSrc = GetTile(
                    paddingTensorSrc,
                    offset,
                    MakeShape(static_cast<uint32_t>(1), actualDataNum)
                );
                auto tensorTileDst = GetTile(
                    paddingTensorDst,
                    offset,
                    MakeShape(static_cast<uint32_t>(1), actualDataNum)
                );

                auto layoutDstUb = MakeLayout(
                    MakeShape(static_cast<uint32_t>(1), actualDataNum),
                    MakeStride(static_cast<int64_t>(COMPUTE_LENGTH), Int<1>{})
                );
                auto tensorDstUb = MakeTensor(inputBuffer[bufferIndex], layoutDstUb, arch::PositionUB{});

                copyGm2Ub(tensorDstUb, tensorTileSrc);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                auto layoutSrcUb = MakeLayout(
                    MakeShape(
                        CeilDiv(actualDataNum, get<1, 0>(paddingTensorDst.shape())),
                        get<1, 0>(paddingTensorDst.shape())
                    ),
                    MakeStride(static_cast<int64_t>(get<1, 0>(paddingTensorDst.shape())), Int<1>{})
                );
                auto tensorSrcUb = MakeTensor(inputBuffer[bufferIndex], layoutSrcUb, arch::PositionUB{});
                copyUb2Gm(tensorTileDst, tensorSrcUb);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / roundTileLen;
            coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }
                auto offset = tla::MakeCoord(mIdx + tileIdx, static_cast<uint32_t>(0));

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                auto tensorTileSrc = GetTile(
                    paddingTensorSrc,
                    offset,
                    MakeShape(actualTilesNum, tileLen)
                );

                auto layoutDstUb = MakeLayout(
                    MakeShape(actualTilesNum, tileLen),
                    MakeStride(static_cast<int64_t>(roundTileLen), Int<1>{})
                );
                auto tensorDstUb = MakeTensor(inputBuffer[bufferIndex], layoutDstUb, arch::PositionUB{});

                copyGm2Ub(tensorDstUb, tensorTileSrc);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                auto layoutSrcUb = MakeLayout(
                    MakeShape(
                        CeilDiv(tileLen, get<1, 0>(paddingTensorDst.shape())),
                        get<1, 0>(paddingTensorDst.shape())
                    ),
                    MakeStride(static_cast<int64_t>(get<1, 0>(paddingTensorDst.shape())), Int<1>{})
                );
                for (uint32_t i = 0; i < actualTilesNum; ++i) {
                    auto tensorTileDst = GetTile(
                        paddingTensorDst,
                        tla::MakeCoord(mIdx + tileIdx + i, static_cast<uint32_t>(0)),
                        MakeShape(static_cast<uint32_t>(1), tileLen)
                    );
                    auto tensorSrcUb = MakeTensor(
                        inputBuffer[bufferIndex][i * roundTileLen],
                        layoutSrcUb,
                        arch::PositionUB{}
                    );
                    copyUb2Gm(tensorTileDst, tensorSrcUb);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    ASCENDCT_DEVICE
    ~PaddingMatrixBlockND() {}
private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<Element> inputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(Element) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class PaddingA,
    class PaddingB
>
class OptimizedMatmulTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    template<class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template<>
    struct LayoutHelper<void> {
        using type = void;
    };
    using LayoutA = std::conditional_t<std::is_void_v<PaddingA>, LayoutWA, typename LayoutHelper<PaddingA>::type>;
    using LayoutB = std::conditional_t<std::is_void_v<PaddingB>, LayoutWB, typename LayoutHelper<PaddingB>::type>;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

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
        GM_ADDR ptrWA;
        LayoutWA layoutWA;
        GM_ADDR ptrWB;
        LayoutWB layoutWB;

        // Methods
        ASCENDCT_DEVICE
        Params() {}

        ASCENDCT_DEVICE
        Params(GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_,
               GM_ADDR ptrWA_, LayoutWA layoutWA_, GM_ADDR ptrWB_, LayoutWB layoutWB_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_), ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_) {}
    };

    // Methods
    ASCENDCT_DEVICE
    OptimizedMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    ASCENDCT_DEVICE
    void operator()(Params const &params);

    template <>
    ASCENDCT_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        if constexpr (!std::is_void_v<PaddingA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            auto tensorA = MakeTensor<AscendC::GlobalTensor<ElementA>, LayoutA, AscendC::TPosition::GM>(
                gmA, params.layoutA);
            auto tensorWA = MakeTensor<AscendC::GlobalTensor<ElementA>, LayoutWA, AscendC::TPosition::GM>(
                gmWA, params.layoutWA);
            PaddingA paddingA(resource);
            paddingA(tensorWA, tensorA);
        }

        if constexpr (!std::is_void_v<PaddingB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            auto tensorB = MakeTensor<AscendC::GlobalTensor<ElementB>, LayoutB, AscendC::TPosition::GM>(
                gmB, params.layoutB);
            auto tensorWB = MakeTensor<AscendC::GlobalTensor<ElementB>, LayoutWB, AscendC::TPosition::GM>(
                gmWB, params.layoutWB);
            PaddingB paddingB(resource);
            paddingB(tensorWB, tensorB);
            // 0x0 synchronization control between AI Core
        }
        if constexpr (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            AscendCT::arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            AscendCT::arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }
    }

    /// Executes matmul
    template <>
    ASCENDCT_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        if (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            AscendCT::arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        auto tensorA = MakeTensor<AscendC::GlobalTensor<ElementA>, LayoutWA, AscendC::TPosition::GM>(
            gmA, params.layoutWA);
        auto tensorB = MakeTensor<AscendC::GlobalTensor<ElementB>, LayoutWB, AscendC::TPosition::GM>(
            gmB, params.layoutWB);
        auto tensorC = MakeTensor<AscendC::GlobalTensor<ElementC>, LayoutC, AscendC::TPosition::GM>(
            gmC, params.layoutC);

        BlockMmad blockMmad(resource);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Compute initial location in logical coordinates
            auto tensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                MakeShape(actualBlockShape.m(), actualBlockShape.k())
            );
            auto tensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                MakeShape(actualBlockShape.k(), actualBlockShape.n())
            );
            auto tensorBlockC = GetTile(
                tensorC,
                tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                MakeShape(actualBlockShape.m(), actualBlockShape.n())
            );

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            GemmCoord nextBlockCoord;
            GemmCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                nextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord);
            }

            auto nextTensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(nextBlockCoord.m() * L1_TILE_M, nextBlockCoord.k() * L1_TILE_K),
                MakeShape(nextActualBlockShape.m(), nextActualBlockShape.k())
            );
            auto nextTensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(nextBlockCoord.k() * L1_TILE_K, nextBlockCoord.n() * L1_TILE_N),
                MakeShape(nextActualBlockShape.k(), nextActualBlockShape.n())
            );

            // Compute block-scoped matrix multiply-add
            blockMmad(
                tensorBlockA, tensorBlockB, tensorBlockC, nextTensorBlockA, nextTensorBlockB,
                isFirstBlock, hasNextBlock
            );
        }
    }

private:
    static constexpr arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    arch::Resource<ArchTag> resource;
};

} // namespace AscendCT::gemm::kernel

#endif // ASCENDCT_MATMUL_KERNEL_OPTIMIZED_MATMUL_TLA_HPP
