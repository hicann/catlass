/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#ifndef CATLASS_GEMM_KERNEL_HSTU_INFER_HPP
#define CATLASS_GEMM_KERNEL_HSTU_INFER_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "tla/layout.hpp"

#include "catlass/gemm/gemm_type.hpp"

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"

#include "tla/tensor.hpp"
#include "tla/layout.hpp"

using namespace Catlass;

namespace Catlass::layout {

struct NTD {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 3;
};

struct TND {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 3;
};

struct NHD { // paged kv cache layout: num_blocks, kv_block_size, kv_heads, head_dim
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;
};
} // namespace Catlass::layout

namespace tla {

template <class Element, class LayoutTag, class T, class U, class V>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(T const& rows, U const& cols, V const& depth)
{
    static_assert(
        std::is_same_v<LayoutTag, Catlass::layout::NTD> || std::is_same_v<LayoutTag, Catlass::layout::TND>,
        "Unsupported LayoutTag for MakeLayoutFromTag, only support Catlass::layout::NTD and Catlass::layout::TND");

    constexpr uint32_t ELE_NUM_PER_C0 =
        Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<Element>::value;
    constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;

    return MakeLayout(
        MakeShape(rows, cols, depth), MakeStride((int64_t)cols * depth, (int64_t)depth, Int<1>{}),
        MakeShape(rows, cols, depth));
}

template <class Element, class LayoutTag, class T, class U, class V, class W>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(
    T const& numBlocks, U const& blockSize, V const& headNum, W const& headDim)
{
    static_assert(
        std::is_same_v<LayoutTag, Catlass::layout::NHD>,
        "Unsupported LayoutTag for MakeLayoutFromTag, only support Catlass::layout::NHD");

    constexpr uint32_t ELE_NUM_PER_C0 =
        Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<Element>::value;
    constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;

    return MakeLayout(
        MakeShape(numBlocks, blockSize, headNum, headDim),
        MakeStride((int64_t)blockSize * headNum * headDim, (int64_t)headNum * headDim, (int64_t)headDim, Int<1>{}),
        MakeShape(numBlocks, blockSize, headNum, headDim));
}

template <class OriginTensor, class ElementTable>
struct PagedTensor {
public:
    /// Logical rank of tensor
    OriginTensor tensor;
    AscendC::GlobalTensor<ElementTable> blockTable;
    uint32_t headIdx;
    MatrixCoord coordBTopLeftDot;
};
} // namespace tla

namespace Catlass::Gemm::Kernel {

// Template for HSTU kernel.
template <
    class BlockMmadQK, class BlockMmadPV, class EpilogueSiluScale, class EpilogueCast,
    class LayoutTagIntfQO_ = Catlass::layout::NTD, class LayoutTagIntfKV_ = Catlass::layout::NTD>
class HstuInfer {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;

    // block interface
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;
    using LayoutTagQ = typename BlockMmadQK::LayoutTagA;

    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;
    using LayoutTagK = typename BlockMmadQK::LayoutTagB;
    using TensorK = typename tla::Tensor<
        AscendC::GlobalTensor<ElementK>, LayoutK, tla::Coord<tla::Int<0>, tla::Int<0>>, AscendC::TPosition::GM>;

    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutS = typename BlockMmadQK::LayoutC;
    using LayoutTagS = typename BlockMmadQK::LayoutTagC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutP = typename BlockMmadPV::LayoutA;
    using LayoutTagP = typename BlockMmadPV::LayoutTagA;

    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutV = typename BlockMmadPV::LayoutB;
    using LayoutTagV = typename BlockMmadPV::LayoutTagB;
    using TensorV = typename tla::Tensor<
        AscendC::GlobalTensor<ElementV>, LayoutV, tla::Coord<tla::Int<0>, tla::Int<0>>, AscendC::TPosition::GM>;

    using ElementO = typename BlockMmadPV::ElementC;
    using LayoutO = typename BlockMmadPV::LayoutC;
    using LayoutTagO = typename BlockMmadPV::LayoutTagC;

    using ElementL1Q = typename BlockMmadQK::ElementL1A;
    using LayoutTagL1Q = typename BlockMmadQK::LayoutTagL1A;
    using LayoutL1Q = typename BlockMmadQK::LayoutL1A;

    using ElementL1P = typename BlockMmadPV::ElementL1A;
    using LayoutTagL1P = typename BlockMmadPV::LayoutTagL1A;
    using LayoutL1P = typename BlockMmadPV::LayoutL1A;

    // task interface
    using StrideTaskQ = tla::Stride<int64_t, tla::Int<1>>;
    using StrideTaskK = tla::Stride<tla::Int<1>, int64_t>;
    using StrideTaskV = tla::Stride<int64_t, tla::Int<1>>;
    using StrideTaskO = tla::Stride<int64_t, tla::Int<1>>;

    // kernel interface
    using LayoutTagIntfQO = LayoutTagIntfQO_;
    using LayoutTagIntfKV = LayoutTagIntfKV_;

    using LayoutIntfQO =
        tla::Layout<tla::Shape<uint32_t, uint32_t, uint32_t>, tla::Stride<int64_t, int64_t, tla::Int<1>>>;
    using CoordIntfQO = tla::Coord<tla::Int<0>, tla::Int<0>, tla::Int<0>>; // TND or NTD

    using LayoutIntfKV = std::conditional_t<
        std::is_same_v<LayoutTagIntfKV, layout::TND> || std::is_same_v<LayoutTagIntfKV, layout::NTD>,
        tla::Layout<tla::Shape<uint32_t, uint32_t, uint32_t>, tla::Stride<int64_t, int64_t, tla::Int<1>>>,
        tla::Layout<
            tla::Shape<uint32_t, uint32_t, uint32_t, uint32_t>,
            tla::Stride<int64_t, int64_t, int64_t, tla::Int<1>>> // num_blocks, block_size, kv_heads, head_dim
        >;
    using CoordIntfKV = std::conditional_t<
        std::is_same_v<LayoutTagIntfKV, layout::TND> || std::is_same_v<LayoutTagIntfKV, layout::NTD>,
        tla::Coord<tla::Int<0>, tla::Int<0>, tla::Int<0>>,             // TND or NTD
        tla::Coord<tla::Int<0>, tla::Int<0>, tla::Int<0>, tla::Int<0>> // num_blocks, block_size, kv_heads, head_dim
        >;

    using ElementIntfQ = ElementQ;
    using LayoutTagIntfQ = LayoutTagIntfQO;
    using LayoutIntfQ = LayoutIntfQO;
    using TensorIntfQ = tla::Tensor<AscendC::GlobalTensor<ElementQ>, LayoutIntfQ, CoordIntfQO, AscendC::TPosition::GM>;

    using ElementIntfK = ElementK;
    using LayoutTagIntfK = LayoutTagIntfKV;
    using LayoutIntfK = LayoutIntfKV;
    using TensorIntfK = tla::Tensor<AscendC::GlobalTensor<ElementK>, LayoutIntfK, CoordIntfKV, AscendC::TPosition::GM>;

    using ElementIntfV = ElementV;
    using LayoutTagIntfV = LayoutTagIntfKV;
    using LayoutIntfV = LayoutIntfKV;
    using TensorIntfV = tla::Tensor<AscendC::GlobalTensor<ElementV>, LayoutIntfV, CoordIntfKV, AscendC::TPosition::GM>;

    using ElementIntfO = ElementO;
    using LayoutTagIntfO = LayoutTagIntfQO;
    using LayoutIntfO = LayoutIntfQO;
    using TensorIntfO = tla::Tensor<AscendC::GlobalTensor<ElementO>, LayoutIntfO, CoordIntfQO, AscendC::TPosition::GM>;

    static_assert(
        std::is_same_v<LayoutTagIntfQO, layout::NTD> || std::is_same_v<LayoutTagIntfQO, layout::TND>,
        "layout tag intf qo must be NTD or TND");

    // QK tile
    using QkL1TileShape = typename BlockMmadQK::L1TileShape;
    using QkL0TileShape = typename BlockMmadQK::L0TileShape;
    static constexpr uint32_t QK_L1_TILE_M = tla::get<0>(QkL1TileShape{});
    static constexpr uint32_t QK_L1_TILE_N = tla::get<1>(QkL1TileShape{});
    static constexpr uint32_t QK_L1_TILE_K = tla::get<2>(QkL1TileShape{});
    static constexpr uint32_t QK_L0_TILE_M = tla::get<0>(QkL0TileShape{});
    static constexpr uint32_t QK_L0_TILE_N = tla::get<1>(QkL0TileShape{});
    static constexpr uint32_t QK_L0_TILE_K = tla::get<2>(QkL0TileShape{});

    // PV tile
    using PvL1TileShape = typename BlockMmadPV::L1TileShape;
    using PvL0TileShape = typename BlockMmadPV::L0TileShape;
    static constexpr uint32_t PV_L1_TILE_M = tla::get<0>(PvL1TileShape{});
    static constexpr uint32_t PV_L1_TILE_N = tla::get<1>(PvL1TileShape{});
    static constexpr uint32_t PV_L1_TILE_K = tla::get<2>(PvL1TileShape{});
    static constexpr uint32_t PV_L0_TILE_M = tla::get<0>(PvL0TileShape{});
    static constexpr uint32_t PV_L0_TILE_N = tla::get<1>(PvL0TileShape{});
    static constexpr uint32_t PV_L0_TILE_K = tla::get<2>(PvL0TileShape{});

    static_assert(BlockMmadQK::L1_TILE_N == BlockMmadPV::L1_TILE_K, "qk l1 tile n and pv l1 k must be the same");

    // CV stages
    static constexpr uint32_t CV_STAGES = 2;

    // l1 malloc
    static_assert(BlockMmadQK::L1B_STAGES == BlockMmadPV::L1B_STAGES, "qk and pv l1b stages must be the same");

    static constexpr uint32_t L1_Q_BUFF_PING_SIZE = BlockMmadQK::L1A_BUFF_PING_SIZE;
    static constexpr uint32_t L1_Q_BUFF_SIZE = BlockMmadQK::L1A_BUFF_SIZE;
    static constexpr uint32_t L1_K_BUFF_PING_SIZE = BlockMmadQK::L1B_BUFF_PING_SIZE;
    static constexpr uint32_t L1_K_BUFF_SIZE = BlockMmadQK::L1B_BUFF_SIZE;
    static constexpr uint32_t L1_P_BUFF_PING_SIZE = BlockMmadPV::L1A_BUFF_PING_SIZE;
    static constexpr uint32_t L1_P_BUFF_SIZE = BlockMmadPV::L1A_BUFF_SIZE;
    static constexpr uint32_t L1_V_BUFF_PING_SIZE = BlockMmadPV::L1B_BUFF_PING_SIZE;
    static constexpr uint32_t L1_V_BUFF_SIZE = BlockMmadPV::L1B_BUFF_SIZE;
    static constexpr uint32_t L1B_BUFF_PING_SIZE = Max(L1_K_BUFF_PING_SIZE, L1_V_BUFF_PING_SIZE); // k/v buff reused
    static constexpr uint32_t L1B_BUFF_SIZE = Max(L1_K_BUFF_SIZE, L1_V_BUFF_SIZE);
    static constexpr uint32_t L1_BUFF_SIZE = L1_Q_BUFF_SIZE + L1_P_BUFF_SIZE + L1B_BUFF_SIZE;

    static constexpr uint32_t L1_Q_BUFF_OFFSET = 0;
    static constexpr uint32_t L1_P_BUFF_OFFSET = L1_Q_BUFF_OFFSET + L1_Q_BUFF_SIZE;

    static constexpr uint32_t L1B_BUFF_OFFSET = ArchTag::L1_SIZE - L1B_BUFF_SIZE;
    static constexpr uint32_t L1_K_BUFF_OFFSET = L1B_BUFF_OFFSET;
    static constexpr uint32_t L1_V_BUFF_OFFSET = L1B_BUFF_OFFSET;

    // l0a malloc
    static constexpr uint32_t L0A_Q_BUFF_PING_SIZE = BlockMmadQK::L0A_BUFF_PING_SIZE;
    static constexpr uint32_t L0A_Q_BUFF_SIZE = BlockMmadQK::L0A_BUFF_SIZE;
    static constexpr uint32_t L0A_P_BUFF_PING_SIZE = BlockMmadPV::L0A_BUFF_PING_SIZE;
    static constexpr uint32_t L0A_P_BUFF_SIZE = BlockMmadPV::L0A_BUFF_SIZE;
    static constexpr uint32_t L0A_BUFF_PING_SIZE = Max(L0A_Q_BUFF_PING_SIZE, L0A_P_BUFF_PING_SIZE);
    static constexpr uint32_t L0A_BUFF_SIZE = Max(L0A_Q_BUFF_SIZE, L0A_P_BUFF_SIZE);
    static constexpr uint32_t L0A_BUFF_OFFSET = 0;
    static constexpr uint32_t L0A_Q_BUFF_OFFSET = L0A_BUFF_OFFSET;
    static constexpr uint32_t L0A_P_BUFF_OFFSET = L0A_BUFF_OFFSET;

    // l0b malloc
    static constexpr uint32_t L0B_K_BUFF_PING_SIZE = BlockMmadQK::L0B_BUFF_PING_SIZE;
    static constexpr uint32_t L0B_K_BUFF_SIZE = BlockMmadQK::L0B_BUFF_SIZE;
    static constexpr uint32_t L0B_V_BUFF_PING_SIZE = BlockMmadPV::L0B_BUFF_PING_SIZE;
    static constexpr uint32_t L0B_V_BUFF_SIZE = BlockMmadPV::L0B_BUFF_SIZE;
    static constexpr uint32_t L0B_BUFF_PING_SIZE = Max(L0B_K_BUFF_PING_SIZE, L0B_V_BUFF_PING_SIZE);
    static constexpr uint32_t L0B_BUFF_SIZE = Max(L0B_K_BUFF_SIZE, L0B_V_BUFF_SIZE);
    static constexpr uint32_t L0B_BUFF_OFFSET = 0;
    static constexpr uint32_t L0B_K_BUFF_OFFSET = L0B_BUFF_OFFSET;
    static constexpr uint32_t L0B_V_BUFF_OFFSET = L0B_BUFF_OFFSET;

    // l0c malloc
    static_assert(
        BlockMmadQK::DispatchPolicy::ENABLE_PV_RESIDENT_L0C == BlockMmadPV::DispatchPolicy::ENABLE_PV_RESIDENT_L0C,
        "qk and pv enable pv resident l0c must be the same");
    static constexpr bool ENABLE_PV_RESIDENT_L0C = BlockMmadQK::DispatchPolicy::ENABLE_PV_RESIDENT_L0C;

    static constexpr uint32_t L0C_S_BUFF_PING_SIZE = BlockMmadQK::L0C_BUFF_PING_SIZE;
    static constexpr uint32_t L0C_S_BUFF_SIZE = BlockMmadQK::L0C_BUFF_SIZE;
    static constexpr uint32_t L0C_O_BUFF_PING_SIZE = BlockMmadPV::L0C_BUFF_PING_SIZE;
    static constexpr uint32_t L0C_O_BUFF_SIZE = BlockMmadPV::L0C_BUFF_SIZE;
    static constexpr uint32_t L0C_BUFF_PING_SIZE =
        ENABLE_PV_RESIDENT_L0C ?
            L0C_O_BUFF_PING_SIZE + L0C_S_BUFF_PING_SIZE :
            ((L0C_O_BUFF_PING_SIZE > L0C_S_BUFF_PING_SIZE) ? L0C_O_BUFF_PING_SIZE : L0C_S_BUFF_PING_SIZE);
    static constexpr uint32_t L0C_BUFF_SIZE =
        ENABLE_PV_RESIDENT_L0C ? L0C_O_BUFF_SIZE + L0C_S_BUFF_SIZE :
                                 ((L0C_O_BUFF_SIZE > L0C_S_BUFF_SIZE) ? L0C_O_BUFF_SIZE : L0C_S_BUFF_SIZE);
    static constexpr uint32_t L0C_S_BUFF_OFFSET = 0;
    static constexpr uint32_t L0C_O_BUFF_OFFSET =
        ENABLE_PV_RESIDENT_L0C ? L0C_S_BUFF_OFFSET + L0C_S_BUFF_SIZE : L0C_S_BUFF_OFFSET;

    static_assert(L1_BUFF_SIZE <= ArchTag::L1_SIZE, "l1 q k p/v exceeding the l1 space!");
    static_assert(L0A_BUFF_SIZE <= ArchTag::L0A_SIZE, "l0a exceeding the l0a space!");
    static_assert(L0B_BUFF_SIZE <= ArchTag::L0B_SIZE, "l0b exceeding the l0b space!");
    static_assert(L0C_BUFF_SIZE <= ArchTag::L0C_SIZE, "l0c s o exceeding the L0A space!");

    // ub
    static constexpr uint32_t UB_S_BUFF_PING_SIZE = EpilogueSiluScale::UB_SRC_BUFF_PING_SIZE;
    static constexpr uint32_t UB_S_BUFF_SIZE = EpilogueSiluScale::UB_SRC_BUFF_SIZE;
    static constexpr uint32_t UB_P_BUFF_PING_SIZE = EpilogueSiluScale::UB_DST_BUFF_PING_SIZE;
    static constexpr uint32_t UB_P_BUFF_SIZE = EpilogueSiluScale::UB_DST_BUFF_SIZE;
    static constexpr uint32_t UB_BUFF_SIZE = UB_S_BUFF_SIZE + UB_P_BUFF_SIZE;
    static constexpr uint32_t UB_S_BUFF_OFFSET = 0;
    static constexpr uint32_t UB_P_BUFF_OFFSET = UB_S_BUFF_OFFSET + UB_S_BUFF_SIZE;

    static_assert(UB_BUFF_SIZE <= ArchTag::UB_SIZE, "ub s/p exceeding the UB space!");
    static_assert(QK_L1_TILE_M == EpilogueSiluScale::UB_TILE_M, "QK L1 TILE M must be the same as UB S BUFF TILE M!");
    static_assert(QK_L1_TILE_N == EpilogueSiluScale::UB_TILE_N, "QK L1 TILE N must be the same as UB S BUFF TILE N!");

    // layout check
    static constexpr bool ENABLE_PAGED_KV_CACHE = BlockMmadQK::DispatchPolicy::ENABLE_PAGED_KV_CACHE;
    static_assert(
        BlockMmadQK::DispatchPolicy::ENABLE_PAGED_KV_CACHE == BlockMmadPV::DispatchPolicy::ENABLE_PAGED_KV_CACHE,
        "kv cache must be the same for qk and pv");
    static_assert(
        (!ENABLE_PAGED_KV_CACHE &&
         (std::is_same_v<LayoutTagIntfKV, layout::NTD> || std::is_same_v<LayoutTagIntfKV, layout::TND>)) ||
            (ENABLE_PAGED_KV_CACHE && std::is_same_v<LayoutTagIntfKV, layout::NHD>),
        "paged disabled: layout tag intf kv must be NTD, TND; paged enabled: layout tag intf kv must be NHD");

    // Parameters structure
    struct Params {
        // Data members
        uint32_t batch;
        uint32_t numHeads;
        uint32_t headDim;
        uint32_t kvHeads;
        uint32_t maxKvSeqlen;
        uint32_t numPagedBlocks;
        uint32_t pagedBlockSize;
        float siluScale;
        uint32_t maskType;

        GM_ADDR ptrQ;
        GM_ADDR ptrK;
        GM_ADDR ptrV;
        GM_ADDR ptrO;
        GM_ADDR ptrActualQSeqlen;
        GM_ADDR ptrActualKvSeqlen;
        GM_ADDR ptrBlockTable;
        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            uint32_t batch_, uint32_t numHeads_, uint32_t headDim_, uint32_t kvHeads_, uint32_t maxKvSeqlen_,
            uint32_t numPagedBlocks_, uint32_t pagedBlockSize_, float siluScale_, uint32_t maskType_, GM_ADDR ptrQ_,
            GM_ADDR ptrK_, GM_ADDR ptrV_, GM_ADDR ptrO_, GM_ADDR ptrActualQSeqlen_, GM_ADDR ptrActualKvSeqlen_,
            GM_ADDR ptrBlockTable_, GM_ADDR ptrWorkspace_)
            : batch(batch_),
              numHeads(numHeads_),
              headDim(headDim_),
              kvHeads(kvHeads_),
              maxKvSeqlen(maxKvSeqlen_),
              numPagedBlocks(numPagedBlocks_),
              pagedBlockSize(pagedBlockSize_),
              siluScale(siluScale_),
              maskType(maskType_),
              ptrQ(ptrQ_),
              ptrK(ptrK_),
              ptrV(ptrV_),
              ptrO(ptrO_),
              ptrActualQSeqlen(ptrActualQSeqlen_),
              ptrActualKvSeqlen(ptrActualKvSeqlen_),
              ptrBlockTable(ptrBlockTable_),
              ptrWorkspace(ptrWorkspace_)
        {}
    };

    struct Arguments {
        uint32_t batch;
        uint32_t numHeads;
        uint32_t headDim;
        uint32_t kvHeads;
        uint32_t maxKvSeqlen;
        uint32_t numPagedBlocks;
        uint32_t pagedBlockSize;
        float siluScale;
        uint32_t maskType;

        GM_ADDR ptrQ;
        GM_ADDR ptrK;
        GM_ADDR ptrV;
        GM_ADDR ptrO;
        GM_ADDR ptrActualQSeqlen;
        GM_ADDR ptrActualKvSeqlen;
        GM_ADDR ptrBlockTable;
    };

    static bool CanImplement(const Arguments& args)
    {
        if (args.headDim == 0 || args.headDim > 256) {
            return false;
        }
        if (args.numHeads != args.kvHeads) {
            return false;
        }
        if (args.maskType != 0 && args.maskType != 1) {
            return false;
        }
        if (ENABLE_PAGED_KV_CACHE && args.pagedBlockSize == 0) {
            return false;
        }
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        Params params{
            args.batch,
            args.numHeads,
            args.headDim,
            args.kvHeads,
            args.maxKvSeqlen,
            args.numPagedBlocks,
            args.pagedBlockSize,
            args.siluScale,
            args.maskType,
            args.ptrQ,
            args.ptrK,
            args.ptrV,
            args.ptrO,
            args.ptrActualQSeqlen,
            args.ptrActualKvSeqlen,
            args.ptrBlockTable,
            workspace};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    HstuInfer()
    {
        for (uint32_t cvStageIdx = 0; cvStageIdx < CV_STAGES; cvStageIdx++) {
            ubS[cvStageIdx] =
                resource.ubBuf.template GetBufferByByte<ElementS>(UB_S_BUFF_OFFSET + cvStageIdx * UB_S_BUFF_PING_SIZE);
            l1P[cvStageIdx] =
                resource.l1Buf.template GetBufferByByte<ElementP>(L1_P_BUFF_OFFSET + cvStageIdx * L1_P_BUFF_PING_SIZE);
#ifdef __DAV_CUBE__
            pvWaitPdataReadyFuncList[cvStageIdx] = PvWaitPdataReadyFunc{this, cvStageIdx};
#endif
        }

        for (uint32_t l1aStageIdx = 0; l1aStageIdx < BlockMmadQK::L1A_STAGES; l1aStageIdx++) {
#ifdef __DAV_CUBE__
            l1Q[l1aStageIdx] =
                resource.l1Buf.template GetBufferByByte<ElementQ>(L1_Q_BUFF_OFFSET + l1aStageIdx * L1_Q_BUFF_PING_SIZE);
#endif
        }
    }

    CATLASS_DEVICE
    ~HstuInfer()
    {}

    CATLASS_DEVICE
    void Init(Params const& params)
    {
#ifdef __DAV_CUBE__
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
#endif
#ifdef __DAV_VEC__
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
#endif

        gmQ.SetGlobalBuffer((__gm__ ElementQ*)params.ptrQ);
        gmK.SetGlobalBuffer((__gm__ ElementK*)params.ptrK);
        gmV.SetGlobalBuffer((__gm__ ElementV*)params.ptrV);
        gmActualQSeqlen.SetGlobalBuffer((__gm__ int64_t*)params.ptrActualQSeqlen);
        gmActualKvSeqlen.SetGlobalBuffer((__gm__ int64_t*)params.ptrActualKvSeqlen);
        gmO.SetGlobalBuffer((__gm__ ElementO*)params.ptrO);
        if constexpr (ENABLE_PAGED_KV_CACHE) {
            gmBlockTable.SetGlobalBuffer((__gm__ uint32_t*)params.ptrBlockTable);
        }
    }

    using TensorL1P_ =
        tla::Tensor<AscendC::LocalTensor<ElementQ>, LayoutL1P, tla::Coord<int32_t, int32_t>, AscendC::TPosition::A1>;
    using TensorGmV_ =
        tla::Tensor<AscendC::GlobalTensor<ElementV>, LayoutV, tla::Coord<uint32_t, int32_t>, AscendC::TPosition::GM>;
    using TensorGmO_ =
        tla::Tensor<AscendC::GlobalTensor<ElementO>, LayoutO, tla::Coord<int32_t, int32_t>, AscendC::TPosition::GM>;

    struct PvCastParam {
        TensorL1P_ tensorL1P;
        TensorGmV_ tensorGmV;
        TensorGmO_ tensorGmO;

        PagedTensor<TensorIntfV, uint32_t> pagedTensorIntfV;

        GemmCoord actualPvCvBlockShape;
        GemmCoord pvL1TileShape;
        GemmCoord pvL0TileShape;

        uint32_t cvBlockFlag;
        uint32_t cvStageId;

        uint32_t coreIdx;
        uint32_t subCoreIdx;
        uint32_t batchIdx;
        uint32_t qSeqBlockIdx;
        uint32_t kvHeadIdx;
        uint32_t kvSeqBlockIdx;
    };

    static constexpr uint32_t PRE_LAUNCH_NUM = 1;
    static constexpr uint32_t PV_CAST_PARAM_NUM = PRE_LAUNCH_NUM + 1;
    PvCastParam pvCastParam_[PV_CAST_PARAM_NUM];

#ifdef __DAV_CUBE__
    CATLASS_DEVICE void PvProcess(BlockMmadPV& blockMmadPV, PvCastParam* pvCastParam)
    {
        uint32_t cvStageId = pvCastParam->cvStageId;
        Callback callbackPvWaitSdataReady = MakeCallback(&pvWaitPdataReadyFuncList[cvStageId]);

        if constexpr (!ENABLE_PAGED_KV_CACHE) {
            blockMmadPV.computePV(
                pvCastParam->tensorGmO, pvCastParam->tensorL1P, pvCastParam->tensorGmV,
                pvCastParam->actualPvCvBlockShape, pvCastParam->pvL1TileShape, pvCastParam->pvL0TileShape,
                pvCastParam->cvBlockFlag, 0, callbackPvWaitSdataReady);
        } else {
            blockMmadPV.computePV(
                pvCastParam->tensorGmO, pvCastParam->tensorL1P, pvCastParam->pagedTensorIntfV,
                pvCastParam->actualPvCvBlockShape, pvCastParam->pvL1TileShape, pvCastParam->pvL0TileShape,
                pvCastParam->cvBlockFlag, 0, callbackPvWaitSdataReady);
        }
    }
#endif

    CATLASS_DEVICE void operator()(Params const& params)
    {
        // coreIdx
#ifdef __DAV_CUBE__
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t subCoreIdx = AscendC::GetSubBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
#endif
#ifdef __DAV_VEC__
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subCoreIdx = AscendC::GetSubBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
#endif

        Init(params);
        uint32_t batch = params.batch;
        uint32_t qHeadNum = params.numHeads;
        uint32_t headDim = params.headDim;
        uint32_t kvHeadNum = params.kvHeads;
        uint32_t pagedBlockSize = params.pagedBlockSize;
        uint32_t totalPagedBlockNum = params.numPagedBlocks;
        uint32_t maxKvSeqlen = params.maxKvSeqlen;
        uint32_t maxPagedBlockNum = CeilDiv(maxKvSeqlen, pagedBlockSize);
        uint32_t maskType = params.maskType;
        uint32_t totalQSeqLen = gmActualQSeqlen.GetValue(batch);
        uint32_t totalKvSeqLen = gmActualKvSeqlen.GetValue(batch);

#ifdef __DAV_CUBE__
        BlockMmadQK blockMmadQK(
            resource, L1_Q_BUFF_OFFSET, L1_Q_BUFF_PING_SIZE, L1_K_BUFF_OFFSET, L1_K_BUFF_PING_SIZE, L0A_Q_BUFF_OFFSET,
            L0A_Q_BUFF_PING_SIZE, L0B_K_BUFF_OFFSET, L0B_K_BUFF_PING_SIZE, L0C_S_BUFF_OFFSET, L0C_S_BUFF_PING_SIZE);
        blockMmadQK.InitEvent(resource);

        BlockMmadPV blockMmadPV(
            resource, L1_P_BUFF_OFFSET, L1_P_BUFF_PING_SIZE, L1_V_BUFF_OFFSET, L1_V_BUFF_PING_SIZE, L0A_P_BUFF_OFFSET,
            L0A_P_BUFF_PING_SIZE, L0B_V_BUFF_OFFSET, L0B_V_BUFF_PING_SIZE, L0C_O_BUFF_OFFSET, L0C_O_BUFF_PING_SIZE);
        blockMmadPV.InitEvent(resource);
#endif

#ifdef __DAV_VEC__
        EpilogueSiluScale epilogueSiluScale(
            resource, UB_S_BUFF_OFFSET, UB_S_BUFF_PING_SIZE, UB_P_BUFF_OFFSET, UB_P_BUFF_PING_SIZE);
        epilogueSiluScale.InitEvent(resource);
#endif
        uint32_t globalTaskIdx = coreIdx;
        uint32_t AccTaskNum = 0;
        uint32_t AccQSeqLen = 0;
        uint32_t AccKvSeqLen = 0;
        uint32_t cvStageId = 0;

        uint32_t pvCastNeedProcId = 0;
        uint32_t pvCastRealProcId = 0;

        static constexpr uint32_t Q_BLOCK_SIZE = BlockMmadQK::L1_TILE_M;
        static constexpr uint32_t KV_BLOCK_SIZE = BlockMmadQK::L1_TILE_N;

        GemmCoord qkL1TileShape{QK_L1_TILE_M, QK_L1_TILE_N, QK_L1_TILE_K};
        GemmCoord qkL0TileShape{QK_L0_TILE_M, QK_L0_TILE_N, QK_L0_TILE_K};

        GemmCoord pvL1TileShape{PV_L1_TILE_M, PV_L1_TILE_N, PV_L1_TILE_K};
        GemmCoord pvL0TileShape{PV_L0_TILE_M, PV_L0_TILE_N, PV_L0_TILE_K};

        LayoutIntfQO layoutIntfQ;
        LayoutIntfKV layoutIntfK;
        LayoutIntfKV layoutIntfV;
        LayoutIntfQO layoutIntfO;

        if constexpr (std::is_same_v<LayoutTagIntfQO, layout::NTD>) {
            layoutIntfQ = tla::MakeLayout<ElementIntfQ, LayoutTagIntfQ>(qHeadNum, totalQSeqLen, headDim);
            layoutIntfO = tla::MakeLayout<ElementIntfO, LayoutTagIntfO>(qHeadNum, totalQSeqLen, headDim);
        } else {
            layoutIntfQ = tla::MakeLayout<ElementIntfQ, LayoutTagIntfQ>(totalQSeqLen, qHeadNum, headDim);
            layoutIntfO = tla::MakeLayout<ElementIntfO, LayoutTagIntfO>(totalQSeqLen, qHeadNum, headDim);
        }

        if constexpr (!ENABLE_PAGED_KV_CACHE) {
            if constexpr (std::is_same_v<LayoutTagIntfQO, layout::NTD>) {
                layoutIntfK = tla::MakeLayout<ElementIntfK, LayoutTagIntfK>(kvHeadNum, totalKvSeqLen, headDim);
                layoutIntfV = tla::MakeLayout<ElementIntfV, LayoutTagIntfV>(kvHeadNum, totalKvSeqLen, headDim);
            } else {
                layoutIntfK = tla::MakeLayout<ElementIntfK, LayoutTagIntfK>(totalKvSeqLen, kvHeadNum, headDim);
                layoutIntfV = tla::MakeLayout<ElementIntfV, LayoutTagIntfV>(totalKvSeqLen, kvHeadNum, headDim);
            }
        } else {
            layoutIntfK =
                tla::MakeLayout<ElementIntfK, LayoutTagIntfK>(totalPagedBlockNum, pagedBlockSize, kvHeadNum, headDim);
            layoutIntfV =
                tla::MakeLayout<ElementIntfV, LayoutTagIntfV>(totalPagedBlockNum, pagedBlockSize, kvHeadNum, headDim);
        }

        tla::PagedTensor<TensorIntfK, uint32_t> pagedTensorIntfK;
        tla::PagedTensor<TensorIntfV, uint32_t> pagedTensorIntfV;
        if constexpr (ENABLE_PAGED_KV_CACHE) {
            LayoutIntfK layoutPagedK =
                tla::MakeLayout<ElementIntfK, LayoutTagIntfK>(totalPagedBlockNum, pagedBlockSize, kvHeadNum, headDim);
            pagedTensorIntfK.tensor = tla::MakeTensor(gmK, layoutPagedK, Arch::PositionGM{});

            LayoutIntfV layoutPagedV =
                tla::MakeLayout<ElementIntfV, LayoutTagIntfV>(totalPagedBlockNum, pagedBlockSize, kvHeadNum, headDim);
            pagedTensorIntfV.tensor = tla::MakeTensor(gmV, layoutPagedV, Arch::PositionGM{});
        }

        for (uint32_t batchIdx = 0; batchIdx < batch; batchIdx++) {
            uint32_t qSeqlen = gmActualQSeqlen.GetValue(batchIdx + 1) - gmActualQSeqlen.GetValue(batchIdx);
            uint32_t kvSeqlen = gmActualKvSeqlen.GetValue(batchIdx + 1) - gmActualKvSeqlen.GetValue(batchIdx);

            uint32_t qSeqBlockNum = CeilDiv(qSeqlen, Q_BLOCK_SIZE);
            uint32_t taskNum = qSeqBlockNum * qHeadNum;
            uint32_t startTaskId = AccTaskNum;

            for (globalTaskIdx; globalTaskIdx < startTaskId + taskNum; globalTaskIdx += coreNum) {
                uint32_t taskIdxInBatch = globalTaskIdx - startTaskId;
                uint32_t qSeqBlockIdx = taskIdxInBatch / qHeadNum;
                uint32_t qHeadIdx = taskIdxInBatch % qHeadNum;
                uint32_t kvHeadIdx = qHeadIdx;

                uint32_t actualQSeqLen =
                    qSeqBlockIdx < qSeqBlockNum - 1 ? Q_BLOCK_SIZE : qSeqlen - qSeqBlockIdx * Q_BLOCK_SIZE;
                uint32_t qSeqOffset = AccQSeqLen + qSeqBlockIdx * Q_BLOCK_SIZE;
                uint32_t kvSeqOffset = AccKvSeqLen;

#ifdef __DAV_CUBE__
                int64_t qOffset;
                auto qShape = MakeShape(actualQSeqLen, headDim);
                StrideTaskQ qStride;
                if constexpr (std::is_same_v<LayoutTagIntfQ, layout::NTD>) {
                    qOffset = layoutIntfQ(tla::MakeCoord(qHeadIdx, qSeqOffset, 0));
                    qStride = MakeStride((int64_t)headDim, Int<1>{});
                } else {
                    qOffset = layoutIntfQ(tla::MakeCoord(qSeqOffset, qHeadIdx, 0));
                    qStride = MakeStride((int64_t)headDim * qHeadNum, Int<1>{});
                }
                LayoutQ layoutQ = tla::MakeLayout(qShape, qStride, qShape);
                auto tensorQ = tla::MakeTensor(gmQ[qOffset], layoutQ, Arch::PositionGM{});

                TensorK tensorK;
                if constexpr (!ENABLE_PAGED_KV_CACHE) {
                    int64_t kOffset;
                    auto kShape = MakeShape(headDim, kvSeqlen);
                    StrideTaskK kStride;
                    if constexpr (std::is_same_v<LayoutTagIntfK, layout::NTD>) {
                        kOffset = layoutIntfK(tla::MakeCoord(kvHeadIdx, kvSeqOffset, 0));
                        kStride = MakeStride(Int<1>{}, (int64_t)headDim);
                    } else {
                        kOffset = layoutIntfK(tla::MakeCoord(kvSeqOffset, kvHeadIdx, 0));
                        kStride = MakeStride(Int<1>{}, (int64_t)headDim * kvHeadNum);
                    }
                    LayoutK layoutK = tla::MakeLayout(kShape, kStride, kShape);
                    tensorK = tla::MakeTensor(gmK[kOffset], layoutK, Arch::PositionGM{});
                }

                uint32_t l1AListId = 0;
                auto layoutL1Q = tla::MakeLayout<ElementQ, LayoutTagL1Q>(actualQSeqLen, headDim);
                auto tensorL1Q = tla::MakeTensor(l1Q[l1AListId], layoutL1Q, Arch::PositionL1{});

                blockMmadQK.loadQToL1(tensorL1Q, tensorQ, l1AListId);

                TensorV tensorV;
                if constexpr (!ENABLE_PAGED_KV_CACHE) {
                    int64_t vOffset;
                    auto vShape = MakeShape(kvSeqlen, headDim);
                    StrideTaskV vStride;
                    if constexpr (std::is_same_v<LayoutTagIntfV, layout::NTD>) {
                        vOffset = layoutIntfV(tla::MakeCoord(kvHeadIdx, kvSeqOffset, 0));
                        vStride = MakeStride((int64_t)headDim, Int<1>{});
                    } else {
                        vOffset = layoutIntfV(tla::MakeCoord(kvSeqOffset, kvHeadIdx, 0));
                        vStride = MakeStride((int64_t)headDim * kvHeadNum, Int<1>{});
                    }
                    LayoutV layoutV = tla::MakeLayout(vShape, vStride, vShape);
                    tensorV = tla::MakeTensor(gmV[vOffset], layoutV, Arch::PositionGM{});
                }

                int64_t oOffset;
                auto oShape = MakeShape(actualQSeqLen, headDim);
                StrideTaskO oStride;
                if constexpr (std::is_same_v<LayoutTagIntfO, layout::NTD>) {
                    oOffset = layoutIntfO(tla::MakeCoord(qHeadIdx, qSeqOffset, 0));
                    oStride = MakeStride((int64_t)headDim, Int<1>{});
                } else {
                    oOffset = layoutIntfO(tla::MakeCoord(qSeqOffset, qHeadIdx, 0));
                    oStride = MakeStride((int64_t)headDim * qHeadNum, Int<1>{});
                }
                LayoutO layoutO = tla::MakeLayout(oShape, oStride, oShape);
                auto tensorO = tla::MakeTensor(gmO[oOffset], layoutO, Arch::PositionGM{});
#endif
                uint32_t validKvSeqlen = kvSeqlen;
                if (maskType == 1) {
                    uint32_t bottomRowValidKvSeqlen = min(kvSeqlen, Q_BLOCK_SIZE * qSeqBlockIdx + actualQSeqLen);
                    validKvSeqlen = bottomRowValidKvSeqlen;
                }

                uint32_t kvSeqBlockNum = CeilDiv(validKvSeqlen, KV_BLOCK_SIZE);
                for (uint32_t kvSeqBlockIdx = 0; kvSeqBlockIdx < kvSeqBlockNum; kvSeqBlockIdx++) {
                    uint32_t actualKvSeqLen = kvSeqBlockIdx < kvSeqBlockNum - 1 ?
                                                  KV_BLOCK_SIZE :
                                                  validKvSeqlen - kvSeqBlockIdx * KV_BLOCK_SIZE;
                    GemmCoord actualQkCvBlockShape{actualQSeqLen, actualKvSeqLen, headDim};

                    uint32_t firstCvBlockFlag = kvSeqBlockIdx == 0 ? 1 : 0;
                    uint32_t lastCvBlockFlag = kvSeqBlockIdx == kvSeqBlockNum - 1 ? 1 : 0;
                    uint32_t cvBlockFlag = firstCvBlockFlag << 0 | lastCvBlockFlag
                                                                       << 1; // bit0: first block; bit1: last block;

                    uint32_t roundActualKvSeqLen = RoundUp<32>(actualKvSeqLen); // 满足 to UB大小
                    auto layoutUbS = tla::MakeLayout<ElementS, LayoutTagS>(actualQSeqLen, roundActualKvSeqLen);
                    auto tensorUbSCvBlock_ = tla::MakeTensor(ubS[cvStageId], layoutUbS, Arch::PositionUB{});
                    auto tensorUbSCvBlock =
                        GetTile(tensorUbSCvBlock_, tla::MakeCoord(0, 0), tla::MakeShape(actualQSeqLen, actualKvSeqLen));

#ifdef __DAV_CUBE__
                    auto tensorQCvBlock = tensorL1Q;

                    if constexpr (!ENABLE_PAGED_KV_CACHE) {
                        auto tensorKCvBlock = tla::GetTile(
                            tensorK, tla::MakeCoord(0, kvSeqBlockIdx * KV_BLOCK_SIZE),
                            tla::MakeShape(headDim, actualKvSeqLen));
                        blockMmadQK.computeQK(
                            tensorUbSCvBlock, tensorQCvBlock, tensorKCvBlock, actualQkCvBlockShape, qkL1TileShape,
                            qkL0TileShape, cvBlockFlag, l1AListId);
                    } else {
                        MatrixCoord kCoord{0, kvSeqBlockIdx * KV_BLOCK_SIZE};
                        pagedTensorIntfK.headIdx = kvHeadIdx;
                        pagedTensorIntfK.coordBTopLeftDot = kCoord;
                        pagedTensorIntfK.blockTable = gmBlockTable[batchIdx * maxPagedBlockNum];
                        blockMmadQK.computeQK(
                            tensorUbSCvBlock, tensorQCvBlock, pagedTensorIntfK, actualQkCvBlockShape, qkL1TileShape,
                            qkL0TileShape, cvBlockFlag, l1AListId);
                    }

                    QkSetSdataReady(cvStageId);
#endif
                    auto layoutL1P = tla::MakeLayout<ElementP, LayoutTagL1P>(actualQSeqLen, roundActualKvSeqLen);
                    auto tensorL1PCvBlock_ = tla::MakeTensor(l1P[cvStageId], layoutL1P, Arch::PositionL1{});
                    auto tensorL1PCvBlock =
                        GetTile(tensorL1PCvBlock_, tla::MakeCoord(0, 0), tla::MakeShape(actualQSeqLen, actualKvSeqLen));

#ifdef __DAV_VEC__
                    MatrixCoord sShape = actualQkCvBlockShape.GetCoordMN();

                    SiluWaitSdataReady(cvStageId);

                    MatrixCoord topLeftDotCoord{qSeqBlockIdx * Q_BLOCK_SIZE, kvSeqBlockIdx * KV_BLOCK_SIZE};

                    if (maskType == 1) {
                        epilogueSiluScale(
                            tensorL1PCvBlock, tensorUbSCvBlock, sShape, params.siluScale, cvStageId, topLeftDotCoord);
                    } else {
                        epilogueSiluScale(tensorL1PCvBlock, tensorUbSCvBlock, sShape, params.siluScale, cvStageId);
                    }

                    SiluSetPdataReady(cvStageId);
#endif

#ifdef __DAV_CUBE__
                    auto tensorOCvBlock =
                        tla::GetTile(tensorO, tla::MakeCoord(0, 0), tla::MakeShape(actualQSeqLen, headDim));

                    GemmCoord actualPvCvBlockShape{actualQSeqLen, headDim, actualKvSeqLen};

                    PvCastParam* pvCastParam = &pvCastParam_[pvCastNeedProcId % PV_CAST_PARAM_NUM];

                    pvCastParam->coreIdx = coreIdx;
                    pvCastParam->subCoreIdx = subCoreIdx;
                    pvCastParam->batchIdx = batchIdx;
                    pvCastParam->qSeqBlockIdx = qSeqBlockIdx;
                    pvCastParam->kvHeadIdx = kvHeadIdx;
                    pvCastParam->kvSeqBlockIdx = kvSeqBlockIdx;

                    auto tensorVCvBlock = tla::GetTile(
                        tensorV, tla::MakeCoord((int32_t)kvSeqBlockIdx * KV_BLOCK_SIZE, 0),
                        tla::MakeShape(actualKvSeqLen, headDim));

                    pvCastParam->tensorL1P = tensorL1PCvBlock;
                    pvCastParam->tensorGmV = tensorVCvBlock;
                    pvCastParam->tensorGmO = tensorOCvBlock;
                    pvCastParam->actualPvCvBlockShape = actualPvCvBlockShape;
                    pvCastParam->pvL1TileShape = pvL1TileShape;
                    pvCastParam->pvL0TileShape = pvL0TileShape;
                    pvCastParam->cvBlockFlag = cvBlockFlag;
                    pvCastParam->cvStageId = cvStageId;

                    if constexpr (ENABLE_PAGED_KV_CACHE) {
                        MatrixCoord vCoord{kvSeqBlockIdx * KV_BLOCK_SIZE, 0};
                        pagedTensorIntfV.headIdx = kvHeadIdx;
                        pagedTensorIntfV.coordBTopLeftDot = vCoord;
                        pagedTensorIntfV.blockTable = gmBlockTable[batchIdx * maxPagedBlockNum];
                        pvCastParam->pagedTensorIntfV = pagedTensorIntfV;
                    }

                    if (likely(pvCastNeedProcId >= PRE_LAUNCH_NUM)) {
                        PvProcess(blockMmadPV, &pvCastParam_[pvCastRealProcId % PV_CAST_PARAM_NUM]);
                        pvCastRealProcId++;
                    }
#endif
                    pvCastNeedProcId++;
                    cvStageId = (cvStageId + 1 < CV_STAGES) ? (cvStageId + 1) : 0;
                }
            }

            AccQSeqLen += qSeqlen;
            AccKvSeqLen += kvSeqlen;
            AccTaskNum += taskNum;
        }

        for (uint32_t i = pvCastRealProcId; i < pvCastNeedProcId; i++) {
#ifdef __DAV_CUBE__
            PvProcess(blockMmadPV, &pvCastParam_[i % PV_CAST_PARAM_NUM]);
#endif
        }

#ifdef __DAV_CUBE__
        blockMmadQK.ClearEvent();
        blockMmadPV.ClearEvent();
#endif
#ifdef __DAV_VEC__
        epilogueSiluScale.ClearEvent();
#endif
        // AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;

    CATLASS_DEVICE void QkSetSdataReady(uint32_t cvStageId)
    {
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(QK_TO_SILU_FLAG_ID + cvStageId);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(QK_TO_SILU_FLAG_ID + FLAG_ID_MAX + cvStageId);
    }

    CATLASS_DEVICE void SiluWaitSdataReady(uint32_t cvStageId)
    {
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_4, PIPE_V>(QK_TO_SILU_FLAG_ID + cvStageId);
    }

    CATLASS_DEVICE void SiluSetPdataReady(uint32_t cvStageId)
    {
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_4, PIPE_MTE3>(SILU_TO_PV_FLAG_ID + cvStageId);
    }

    CATLASS_DEVICE void PvWaitPdataReady(uint32_t cvStageId)
    {
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(SILU_TO_PV_FLAG_ID + cvStageId);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(SILU_TO_PV_FLAG_ID + FLAG_ID_MAX + cvStageId);
    }

    struct PvWaitPdataReadyFunc {
        using Kernel = HstuInfer;

        CATLASS_DEVICE
        PvWaitPdataReadyFunc() = default;

        CATLASS_DEVICE
        PvWaitPdataReadyFunc(Kernel* kernel, uint32_t cvStageId) : kernelPtr(kernel), cvStageId(cvStageId)
        {}

        CATLASS_DEVICE
        void operator()() const
        {
            kernelPtr->PvWaitPdataReady(cvStageId);
        }

        Kernel* kernelPtr{nullptr};
        uint32_t cvStageId{0};
    };

    // global tensor
    AscendC::GlobalTensor<ElementQ> gmQ;
    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<int64_t> gmActualQSeqlen;
    AscendC::GlobalTensor<int64_t> gmActualKvSeqlen;
    AscendC::GlobalTensor<ElementO> gmO;
    AscendC::GlobalTensor<uint32_t> gmBlockTable;

    AscendC::LocalTensor<ElementQ> l1Q[BlockMmadQK::L1A_STAGES];
    AscendC::LocalTensor<ElementS> ubS[CV_STAGES];
    AscendC::LocalTensor<ElementP> l1P[CV_STAGES];

    static constexpr uint16_t CROSS_CORE_SYNC_MODE_4 = 4;
    static constexpr uint16_t AIV_TO_AIC_FLAG_ID = 0;
    static constexpr uint16_t AIC_TO_AIV_FLAG_ID = 4;

    static constexpr uint16_t QK_TO_SILU_FLAG_ID = AIC_TO_AIV_FLAG_ID + 0;
    static constexpr uint16_t SILU_TO_PV_FLAG_ID = AIV_TO_AIC_FLAG_ID + 0;

    static constexpr uint16_t FLAG_ID_MAX = 16;

    PvWaitPdataReadyFunc pvWaitPdataReadyFuncList[CV_STAGES];
};

} // namespace Catlass::Gemm::Kernel

#endif
