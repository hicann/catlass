/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "fag_common/common_header.h"
#include "fag_common/cube_addr.h"
#include "fag_common/vector_addr.h"

using namespace AscendC;
using namespace Catlass;

template <
    class BlockMmadFAGCube1_,
    class BlockMmadFAGCube2_,
    class BlockMmadFAGCube3_,
    class EpilogueFAGPre_,
    class EpilogueFAGSfmg_,
    class EpilogueFAGOp_,
    class EpilogueFAGPost_
>
class FAGKernel {
public:
    using BlockMmadFAGCube1 = BlockMmadFAGCube1_;
    using ArchTag = typename BlockMmadFAGCube1::ArchTag;
    using L1TileShape = typename BlockMmadFAGCube1::L1TileShape;
    using ElementA1 = typename BlockMmadFAGCube1::ElementA;
    using LayoutA1 = typename BlockMmadFAGCube1::LayoutA;
    using ElementB1 = typename BlockMmadFAGCube1::ElementB;
    using LayoutB1 = typename BlockMmadFAGCube1::LayoutB;
    using ElementC1 = typename BlockMmadFAGCube1::ElementC;
    using LayoutC1 = typename BlockMmadFAGCube1::LayoutC;

    using BlockMmadFAGCube2 = BlockMmadFAGCube2_;
    using ElementA2 = typename BlockMmadFAGCube2::ElementA;
    using LayoutA2 = typename BlockMmadFAGCube2::LayoutA;
    using ElementB2 = typename BlockMmadFAGCube2::ElementB;
    using LayoutB2 = typename BlockMmadFAGCube2::LayoutB;
    using ElementC2 = typename BlockMmadFAGCube2::ElementC;
    using LayoutC2 = typename BlockMmadFAGCube2::LayoutC;

    using BlockMmadFAGCube3 = BlockMmadFAGCube3_;
    using ElementA3 = typename BlockMmadFAGCube3::ElementA;
    using LayoutA3 = typename BlockMmadFAGCube3::LayoutA;
    using ElementB3 = typename BlockMmadFAGCube3::ElementB;
    using LayoutB3 = typename BlockMmadFAGCube3::LayoutB;
    using ElementC3 = typename BlockMmadFAGCube3::ElementC;
    using LayoutC3 = typename BlockMmadFAGCube3::LayoutC;
    
    using EpilogueFAGPre = EpilogueFAGPre_;
    using EpilogueFAGSfmg = EpilogueFAGSfmg_;
    using EpilogueFAGOp = EpilogueFAGOp_;
    using EpilogueFAGPost = EpilogueFAGPost_;



    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR query;
        GM_ADDR key;
        GM_ADDR value;
        GM_ADDR dy;
        GM_ADDR query_right;
        GM_ADDR key_right; 
        GM_ADDR pse_shift;
        GM_ADDR drop_mask;
        GM_ADDR padding_mask; 
        GM_ADDR atten_mask;
        GM_ADDR softmax_max;
        GM_ADDR softmax_sum;
        GM_ADDR softmax_in; 
        GM_ADDR attention_in;
        GM_ADDR prefix;
        GM_ADDR actual_seq_qlen; 
        GM_ADDR actual_seq_kvlen;
        GM_ADDR q_start_idx;
        GM_ADDR kv_start_idx; 
        GM_ADDR dq;
        GM_ADDR dk;
        GM_ADDR dv;
        GM_ADDR dq_right;
        GM_ADDR dk_right;
        GM_ADDR workspace;
        GM_ADDR tiling_data;

        // Methods
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(
            GM_ADDR query_, GM_ADDR key_, GM_ADDR value_, GM_ADDR dy_,
            GM_ADDR query_right_, GM_ADDR key_right_, GM_ADDR pse_shift_,
            GM_ADDR drop_mask_, GM_ADDR padding_mask_, GM_ADDR atten_mask_,
            GM_ADDR softmax_max_, GM_ADDR softmax_sum_, GM_ADDR softmax_in_, 
            GM_ADDR attention_in_, GM_ADDR prefix_, GM_ADDR actual_seq_qlen_, 
            GM_ADDR actual_seq_kvlen_, GM_ADDR q_start_idx_, GM_ADDR kv_start_idx_, 
            GM_ADDR dq_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dq_right_, GM_ADDR dk_right_, 
            GM_ADDR workspace_, GM_ADDR tiling_data_
        ) : query(query_), key(key_), value(value_), dy(dy_),
            query_right(query_right_), key_right(key_right_), pse_shift(pse_shift_),
            drop_mask(drop_mask_), padding_mask(padding_mask_), atten_mask(atten_mask_),
            softmax_max(softmax_max_), softmax_sum(softmax_sum_), softmax_in(softmax_in_), 
            attention_in(attention_in_), prefix(prefix_), actual_seq_qlen(actual_seq_qlen_), 
            actual_seq_kvlen(actual_seq_kvlen_), q_start_idx(q_start_idx_), kv_start_idx(kv_start_idx_), 
            dq(dq_), dk(dk_), dv(dv_), dq_right(dq_right_), dk_right(dk_right_), 
            workspace(workspace_), tiling_data(tiling_data_)
        {
        }    
    };

    // Methods
    CATLASS_DEVICE
    FAGKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        AscendC::GlobalTensor<uint64_t> tilingData;
        tilingData.SetGlobalBuffer((__gm__ uint64_t *)params.tiling_data);
        AscendC::GlobalTensor<uint32_t> tilingDataU32;
        tilingDataU32.SetGlobalBuffer((__gm__ uint32_t *)params.tiling_data);

        int64_t batch = tilingData.GetValue(TILING_B);
        int64_t g = tilingData.GetValue(TILING_G);
        int64_t n2 = tilingData.GetValue(TILING_N2);
        int64_t n1 = n2 * g;
        int64_t headDim = tilingData.GetValue(TILING_D);
        int64_t dqWorkSpaceOffset = tilingData.GetValue(TILING_DQ_WORKSPACE_OFFSET);
        int64_t dkWorkSpaceOffset = tilingData.GetValue(TILING_DK_WORKSPACE_OFFSET);
        int64_t dvWorkSpaceOffset = tilingData.GetValue(TILING_DV_WORKSPACE_OFFSET);
        int64_t mm1WorkspaceOffset = tilingData.GetValue(TILING_MM1_WORKSPACE_OFFSET);
        int64_t mm2WorkspaceOffset = tilingData.GetValue(TILING_MM2_WORKSPACE_OFFSET);
        int64_t pWorkSpaceOffset = tilingData.GetValue(TILING_P_WORKSPACE_OFFSET);
        int64_t dsWorkSpaceOffset = tilingData.GetValue(TILING_DS_WORKSPACE_OFFSET);

        uint32_t coreNum = tilingDataU32.GetValue(TILING_CORE_NUM * CONST_2);
        int64_t mixCoreNum = (coreNum + 1) / 2;

        struct CubeAddrInfo cubeAddrInfo[2];
        int32_t taskId = 0;
        bool running = true;

        CubeAddr cubeAddr;
        cubeAddr.init(batch, n1, g, headDim, GetBlockIdx(), params.actual_seq_qlen, params.actual_seq_kvlen, mixCoreNum);

        uint32_t pingpongFlagL1A = 0;
        uint32_t pingpongFlagL1B = 0;
        uint32_t pingpongFlagL0A = 0;
        uint32_t pingpongFlagL0B = 0;
        uint32_t pingpongFlagC = 0;

        BlockMmadFAGCube1 blockMmadFAGCube1(resource, n1, n2, headDim);
        BlockMmadFAGCube2 blockMmadFAGCube2(resource, n1, n2, headDim);
        BlockMmadFAGCube3 blockMmadFAGCube3(resource, n1, n2, headDim);

        while (running) {
            cubeAddrInfo[taskId % 2].taskId = taskId;
            cubeAddr.addr_mapping(&cubeAddrInfo[taskId % 2]);
            if (cubeAddrInfo[taskId % 2].blockLength > 0) {
                // get start kernel
                SetFlag();
                CubeAddrInfo addrs = cubeAddrInfo[taskId % 2];
                blockMmadFAGCube1(cubeAddrInfo[taskId % 2], (__gm__ half*)(params.query), (__gm__ half*)(params.key), (__gm__ float*)(params.workspace + mm2WorkspaceOffset), 
                    pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B, pingpongFlagC);
                blockMmadFAGCube1(cubeAddrInfo[taskId % 2], (__gm__ half*)(params.dy), (__gm__ half*)(params.value), (__gm__ float*)(params.workspace + mm1WorkspaceOffset), 
                    pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B, pingpongFlagC);
                WaitFlag();
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
            }
            if (taskId > 0 && cubeAddrInfo[(taskId - 1) % 2].blockLength > 0) {
                AscendC::WaitEvent(VEC2CUBE);
                SetFlag();
                blockMmadFAGCube2(cubeAddrInfo[(taskId - 1) % 2], (__gm__ half*)(params.workspace + dsWorkSpaceOffset), (__gm__ half*)(params.key), (__gm__ float*)(params.workspace + dqWorkSpaceOffset), 
                    pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B);
                WaitFlag();
                SetFlag();
                blockMmadFAGCube3(cubeAddrInfo[(taskId - 1) % 2], (__gm__ half*)(params.workspace + pWorkSpaceOffset), (__gm__ half*)(params.dy), (__gm__ float*)(params.workspace + dvWorkSpaceOffset), 
                    pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B, pingpongFlagC);
                WaitFlag();
                SetFlag();
                blockMmadFAGCube3(cubeAddrInfo[(taskId - 1) % 2], (__gm__ half*)(params.workspace + dsWorkSpaceOffset), (__gm__ half*)(params.query), (__gm__ float*)(params.workspace + dkWorkSpaceOffset), 
                    pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B, pingpongFlagC);
                WaitFlag();
            }
            if (cubeAddrInfo[taskId % 2].blockLength == 0) {
                running = false;
            }
            taskId++;
        }
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        AscendC::GlobalTensor<uint64_t> tilingData;
        tilingData.SetGlobalBuffer((__gm__ uint64_t *)params.tiling_data);
        AscendC::GlobalTensor<uint32_t> tilingDataU32;
        tilingDataU32.SetGlobalBuffer((__gm__ uint32_t *)params.tiling_data);

        int64_t batch = tilingData.GetValue(TILING_B);
        int64_t g = tilingData.GetValue(TILING_G);
        int64_t n2 = tilingData.GetValue(TILING_N2);
        int64_t n1 = n2 * g;
        int64_t headDim = tilingData.GetValue(TILING_D); 
        uint32_t coreNum = tilingDataU32.GetValue(TILING_CORE_NUM * CONST_2);
        int64_t mixCoreNum = (coreNum + 1) / 2;
        
        struct VecAddrInfo vecAddrInfo;
        AscendC::TPipe pipePre;
        EpilogueFAGPre epilogueFagPre(resource, &pipePre, params.dq, params.dk, params.dv, params.workspace, params.tiling_data);
        epilogueFagPre();
        pipePre.Destroy();

        // vec SoftmaxGrad
        AscendC::TPipe pipeSoftmaxGrad;
        EpilogueFAGSfmg epilogueFagSfmg(resource, &pipeSoftmaxGrad, params.dy, params.attention_in, params.actual_seq_qlen, params.workspace, batch, params.tiling_data);
        epilogueFagSfmg();
        pipeSoftmaxGrad.Destroy();

        AscendC::SyncAll();

        // vector process
        AscendC::TPipe pipeVec;
        EpilogueFAGOp epilogueFagOp(resource, &pipeVec, params.softmax_max, params.softmax_sum,
            params.atten_mask, params.actual_seq_qlen, params.actual_seq_kvlen, params.workspace, batch, params.tiling_data);

        VectorAddr vector_addr;
        vector_addr.init(batch, n1, g, headDim, GetBlockIdx() / 2, params.actual_seq_qlen, params.actual_seq_kvlen, mixCoreNum);
        int32_t taskId = 0;
        bool running = true;
        while (running) {
            vector_addr.addr_mapping(&vecAddrInfo);
            if (vecAddrInfo.blockLength > 0) {
                AscendC::WaitEvent(CUBE2VEC);
                vecAddrInfo.taskId = taskId;
                epilogueFagOp(vecAddrInfo);
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC2CUBE);
            }
            if (vecAddrInfo.blockLength == 0) {
                running = false;
            }
            taskId++;
        }
        pipeVec.Destroy();
        AscendC::WaitEvent(CUBE2POST);
        AscendC::SyncAll();
        
        // vector post process
        AscendC::TPipe pipePost;
        EpilogueFAGPost epilogueFagPost(resource, &pipePost, params.dq, params.dk, params.dv, params.workspace, params.tiling_data);
        epilogueFagPost();
        pipePost.Destroy(); 
    }

    CATLASS_DEVICE
    void SetFlag()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
    }

    CATLASS_DEVICE
    void WaitFlag()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
    }

private:
    Arch::Resource<ArchTag> resource;
};

CATLASS_GLOBAL
void FAG(uint64_t fftsAddr,
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy,
        GM_ADDR query_right, GM_ADDR key_right, 
        GM_ADDR pse_shift, GM_ADDR drop_mask, GM_ADDR padding_mask, 
        GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR softmax_in, 
        GM_ADDR attention_in, GM_ADDR prefix, GM_ADDR actual_seq_qlen, 
        GM_ADDR actual_seq_kvlen, GM_ADDR q_start_idx, GM_ADDR kv_start_idx, 
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_right, GM_ADDR dk_right, 
        GM_ADDR workspace, GM_ADDR tiling_data)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    // Cube1 计算：左矩阵不转置，右矩阵转置。实现 (Q * K^T) 和 dP = dY * V^T
    using ElementA1 = half;               // query和dy
    using LayoutA1 = layout::RowMajor;
    using ElementB1 = half;               // key和value
    using LayoutB1 = layout::ColumnMajor;
    using ElementC1 = float;
    using LayoutC1 = layout::RowMajor;
    using A1Type = Catlass::Gemm::GemmType<ElementA1, LayoutA1>;
    using B1Type = Catlass::Gemm::GemmType<ElementB1, LayoutB1>;
    using C1Type = Catlass::Gemm::GemmType<ElementC1, LayoutC1>;
    using DispatchPolicyCube1 = Catlass::Gemm::MmadAtlasA2FAGCube1;
    using L1TileShapeCube1 = GemmShape<256, 128, 256>;
    using L0TileShapeCube1 = L1TileShapeCube1;
    using BlockMmadFAGCube1 = Catlass::Gemm::Block::BlockMmad<DispatchPolicyCube1, L1TileShapeCube1, L0TileShapeCube1, A1Type, B1Type, C1Type>;

    // Cube2 计算：左矩阵不转置，右矩阵不转置。实现 dQ = dS * K
    using ElementA2 = half;           // ds
    using LayoutA2 = layout::RowMajor;
    using ElementB2 = half;           // key
    using LayoutB2 = layout::RowMajor;
    using ElementC2 = float;
    using LayoutC2 = layout::RowMajor;

    using A2Type = Catlass::Gemm::GemmType<ElementA2, LayoutA2>;
    using B2Type = Catlass::Gemm::GemmType<ElementB2, LayoutB2>;
    using C2Type = Catlass::Gemm::GemmType<ElementC2, LayoutC2>;
    using DispatchPolicyCube2 = Catlass::Gemm::MmadAtlasA2FAGCube2;
    using L1TileShapeCube2 = GemmShape<128, 128, 128>;
    using L0TileShapeCube2 = L1TileShapeCube2;

    using BlockMmadFAGCube2 = Catlass::Gemm::Block::BlockMmad<DispatchPolicyCube2, L1TileShapeCube2, L0TileShapeCube2, A2Type, B2Type, C2Type>;

    // Cube3 计算：左矩阵转置，右矩阵不转置。 实现 dK = dS^T * Q 和 dV = P^T * dY
    using ElementA3 = half;              // ds和p
    using LayoutA3 = layout::ColumnMajor;
    using ElementB3 = half;              // query和dy
    using LayoutB3 = layout::RowMajor;
    using ElementC3 = float;
    using LayoutC3 = layout::RowMajor;

    using A3Type = Catlass::Gemm::GemmType<ElementA3, LayoutA3>;
    using B3Type = Catlass::Gemm::GemmType<ElementB3, LayoutB3>;
    using C3Type = Catlass::Gemm::GemmType<ElementC3, LayoutC3>;
    using DispatchPolicyCube3 = Catlass::Gemm::MmadAtlasA2FAGCube3;
    using L1TileShapeCube3 = GemmShape<256, 128, 256>;
    using L0TileShapeCube3 = L1TileShapeCube3;

    using BlockMmadFAGCube3 = Catlass::Gemm::Block::BlockMmad<DispatchPolicyCube3, L1TileShapeCube3, L0TileShapeCube3, A3Type, B3Type, C3Type>;

    // Epilogue
    using ElementOutput = float;
    using LayoutOutput = layout::RowMajor;
    using OutputType = Catlass::Gemm::GemmType<ElementOutput, LayoutOutput>;

    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;
    using UpdateType = Catlass::Gemm::GemmType<ElementUpdate, LayoutUpdate>;

    using ElementInput = float;
    using LayoutInput = layout::RowMajor;
    using InputType = Catlass::Gemm::GemmType<ElementInput, LayoutInput>;

    // VEC_Pre ：dQ/dY/dV的workspace清零
    using EpilogueAtlasA2FAGPre = Catlass::Epilogue::EpilogueAtlasA2FAGPre;
    using EpilogueFAGPre = Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGPre, OutputType, UpdateType, InputType>;

    // VEC_Sfmg ：计算 SoftmaxGrad(dY, atten_in)
    using EpilogueAtlasA2FAGSfmg = Catlass::Epilogue::EpilogueAtlasA2FAGSfmg;
    using EpilogueFAGSfmg = Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGSfmg, OutputType, UpdateType, InputType>;

    // VEC_Op：计算S = Mask(Q*K^T)，并完成重计算 P = Softmax(S)，再计算dS = P * Sub(dP, Sfmg)
    using EpilogueAtlasA2FAGOp = Catlass::Epilogue::EpilogueAtlasA2FAGOp;
    using EpilogueFAGOp = Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGOp, OutputType, UpdateType, InputType>;

    // VEC_Post：dQ*scale和dK*scale，并搬运输出dQ/dK/dV
    using EpilogueAtlasA2FAGPost = Catlass::Epilogue::EpilogueAtlasA2FAGPost;
    using EpilogueFAGPost = Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGPost, OutputType, UpdateType, InputType>;

    // Kernel level
    using FAGKernel = FAGKernel<BlockMmadFAGCube1, BlockMmadFAGCube2, BlockMmadFAGCube3, EpilogueFAGPre, EpilogueFAGSfmg, EpilogueFAGOp, EpilogueFAGPost>;
    typename FAGKernel::Params params{
        query, key, value, dy,
        query_right, key_right, pse_shift,
        drop_mask, padding_mask, atten_mask,
        softmax_max, softmax_sum, softmax_in, 
        attention_in, prefix, actual_seq_qlen, 
        actual_seq_kvlen, q_start_idx, kv_start_idx, 
        dq, dk, dv, dq_right, dk_right, 
        workspace, tiling_data};

    // call kernel
    FAGKernel fag;
    fag(params);
}