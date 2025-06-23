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
    class BlockMmadMLACube1_,
    class BlockMmadMLACube2_,
    class BlockMmadMLACube3_,
    class EpilogueFAGPre_,
    class EpilogueFAGSfmg_,
    class EpilogueFAGOp_,
    class EpilogueFAGPost_
>
class FAGKernel {
public:
    using BlockMmadMLACube3 = BlockMmadMLACube3_;
    using ArchTag = typename BlockMmadMLACube3::ArchTag;
    using L1TileShape = typename BlockMmadMLACube3::L1TileShape;
    using ElementdS = typename BlockMmadMLACube3::ElementA;
    using LayoutdS = typename BlockMmadMLACube3::LayoutA;
    using ElementdK = typename BlockMmadMLACube3::ElementC;
    using LayoutdK = typename BlockMmadMLACube3::LayoutC;
    
    using BlockMmadMLACube1 = BlockMmadMLACube1_;
    using ElementQ = typename BlockMmadMLACube1::ElementA;
    using LayoutQ = typename BlockMmadMLACube1::LayoutA;
    using ElementK = typename BlockMmadMLACube1::ElementB;
    using LayoutK = typename BlockMmadMLACube1::LayoutB;
    using ElementS = typename BlockMmadMLACube1::ElementC;
    using LayoutS = typename BlockMmadMLACube1::LayoutC;
    using ElementO = typename BlockMmadMLACube1::ElementA;
    using LayoutO = typename BlockMmadMLACube1::LayoutA;
    using ElementV = typename BlockMmadMLACube1::ElementB;
    using LayoutV = typename BlockMmadMLACube1::LayoutB;
    using ElementP = typename BlockMmadMLACube1::ElementC;
    using LayoutP = typename BlockMmadMLACube1::LayoutC;

    using BlockMmadMLACube2 = BlockMmadMLACube2_;
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

        int64_t batch = tilingData.GetValue(TILING_B);
        int64_t g = tilingData.GetValue(TILING_G);
        int64_t n2 = tilingData.GetValue(TILING_N2);
        int64_t headNum = n2 * g;
        int64_t headDim = tilingData.GetValue(TILING_D);
        int64_t dqWorkSpaceOffset = tilingData.GetValue(TILING_DQ_WORKSPACE_OFFSET);
        int64_t dkWorkSpaceOffset = tilingData.GetValue(TILING_DK_WORKSPACE_OFFSET);
        int64_t dvWorkSpaceOffset = tilingData.GetValue(TILING_DV_WORKSPACE_OFFSET);
        int64_t mm1WorkspaceOffset = tilingData.GetValue(TILING_MM1_WORKSPACE_OFFSET);
        int64_t mm2WorkspaceOffset = tilingData.GetValue(TILING_MM2_WORKSPACE_OFFSET);
        int64_t pWorkSpaceOffset = tilingData.GetValue(TILING_P_WORKSPACE_OFFSET);
        int64_t dsWorkSpaceOffset = tilingData.GetValue(TILING_DS_WORKSPACE_OFFSET);

        AscendC::GlobalTensor<uint32_t> tilingDataU32;
        tilingDataU32.SetGlobalBuffer((__gm__ uint32_t *)params.tiling_data);;
        uint32_t coreNum = tilingDataU32.GetValue(TILING_CORE_NUM * CONST_2);
        int64_t mixCoreNum = (coreNum + 1) / 2;

        struct CubeAddrInfo cubeAddrInfo[2];

        CubeAddr cubeaddr;
        cubeaddr.init(batch, headNum, g, headDim, GetBlockIdx(), params.query, params.key, params.value, params.dy, params.dk_right, params.actual_seq_qlen, params.actual_seq_kvlen, mixCoreNum);

        int32_t taskId = 0;
        uint32_t pingpongFlagL1A = 0;
        uint32_t pingpongFlagL1B = 0;
        uint32_t pingpongFlagL0A = 0;
        uint32_t pingpongFlagL0B = 0;
        uint32_t pingpongFlagC = 0;
        bool running = true;

        BlockMmadMLACube1 blockMmadMLACube1(resource, headNum, n2, headDim);
        BlockMmadMLACube2 blockMmadMLACube2(resource, headNum, n2, headDim);
        BlockMmadMLACube3 blockMmadMLACube3(resource, headNum, n2, headDim);

        while (running) {
            cubeAddrInfo[taskId % 2].taskId = taskId;
            cubeaddr.addr_mapping(&cubeAddrInfo[taskId % 2]);
            if (cubeAddrInfo[taskId % 2].blockLength > 0) {
                // get start kernel
                preset_flag();
                CubeAddrInfo addrs = cubeAddrInfo[taskId % 2];
                blockMmadMLACube1(cubeAddrInfo[taskId % 2], (__gm__ half*)(params.query), (__gm__ half*)(params.key), (__gm__ float*)(params.workspace + mm2WorkspaceOffset), 
                    pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B, pingpongFlagC);
                clear_flag();
                preset_flag();
                blockMmadMLACube1(cubeAddrInfo[taskId % 2], (__gm__ half*)(params.dy), (__gm__ half*)(params.value), (__gm__ float*)(params.workspace + mm1WorkspaceOffset), 
                    pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B, pingpongFlagC);
                clear_flag();
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
            }
            if (taskId > 0 && cubeAddrInfo[(taskId - 1) % 2].blockLength > 0) {
                AscendC::WaitEvent(VEC2CUBE);
                preset_flag();
                blockMmadMLACube2(cubeAddrInfo[(taskId - 1) % 2], (__gm__ half*)(params.workspace + dsWorkSpaceOffset), (__gm__ half*)(params.key), (__gm__ float*)(params.workspace + dqWorkSpaceOffset), 
                    pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B);
                clear_flag();
                preset_flag();
                blockMmadMLACube3(cubeAddrInfo[(taskId - 1) % 2], (__gm__ half*)(params.workspace + pWorkSpaceOffset), (__gm__ half*)(params.dy), (__gm__ float*)(params.workspace + dvWorkSpaceOffset), 
                    pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B, pingpongFlagC);
                clear_flag();
                preset_flag();
                blockMmadMLACube3(cubeAddrInfo[(taskId - 1) % 2], (__gm__ half*)(params.workspace + dsWorkSpaceOffset), (__gm__ half*)(params.query), (__gm__ float*)(params.workspace + dkWorkSpaceOffset), 
                    pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B, pingpongFlagC);
                clear_flag();
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

        int64_t batch = tilingData.GetValue(TILING_B);
        int64_t g = tilingData.GetValue(TILING_G);
        int64_t n2 = tilingData.GetValue(TILING_N2);
        int64_t headNum = n2 * g;
        int64_t headDim = tilingData.GetValue(TILING_D); 
        AscendC::GlobalTensor<uint32_t> tilingDataU32;
        tilingDataU32.SetGlobalBuffer((__gm__ uint32_t *)params.tiling_data);
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
        vector_addr.init(batch, headNum, g, headDim, GetBlockIdx() / 2, params.query, params.key, params.value, params.dy, params.dk_right, params.actual_seq_qlen, params.actual_seq_kvlen, mixCoreNum);
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
    void preset_flag()
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
    void clear_flag()
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
    // Cube1
    using ElementQ = half;
    using LayoutQ = layout::RowMajor;
    using ElementK = half;
    using LayoutK = layout::ColumnMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using QType = Catlass::Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Catlass::Gemm::GemmType<ElementK, LayoutK>;
    using SType = Catlass::Gemm::GemmType<ElementS, LayoutS>;
    using DispatchPolicyCube1 = Catlass::Gemm::MmadAtlasA2FAGCube1;
    using L1TileShapeCube1 = GemmShape<256, 128, 256>;
    using L0TileShapeCube1 = L1TileShapeCube1;
    using BlockMmadMLACube1 = Catlass::Gemm::Block::BlockMmad<DispatchPolicyCube1, L1TileShapeCube1, L0TileShapeCube1, QType, KType, SType>;

    // Cube2
    using ElementdS = half;
    using LayoutdS = layout::RowMajor;
    using ElementKCube2 = half;
    using LayoutKCube2 = layout::RowMajor;
    using ElementdQ = float;
    using LayoutdQ = layout::RowMajor;

    using dsCube2Type = Catlass::Gemm::GemmType<ElementdS, LayoutdS>;
    using kCube2ype = Catlass::Gemm::GemmType<ElementKCube2, LayoutKCube2>;
    using dqCube2Type = Catlass::Gemm::GemmType<ElementdQ, LayoutdQ>;
    using DispatchPolicyCube2 = Catlass::Gemm::MmadAtlasA2FAGCube2;
    using L1TileShapeCube2 = GemmShape<128, 128, 128>;
    using L0TileShapeCube2 = L1TileShapeCube2;

    using BlockMmadMLACube2 = Catlass::Gemm::Block::BlockMmad<DispatchPolicyCube2, L1TileShapeCube2, L0TileShapeCube2, dsCube2Type, kCube2ype, dqCube2Type>;

    // Cube3
    using ElementdSCube3 = half;
    using LayoutdSCube3 = layout::ColumnMajor;
    using ElementQCube3 = half;
    using LayoutQCube3 = layout::RowMajor;
    using ElementdK = float;
    using LayoutdK = layout::RowMajor;

    using dSTypeCube3 = Catlass::Gemm::GemmType<ElementdSCube3, LayoutdSCube3>;
    using QTypeCube3 = Catlass::Gemm::GemmType<ElementQCube3, LayoutQCube3>;
    using dKTypeCube3 = Catlass::Gemm::GemmType<ElementdK, LayoutdK>;
    using DispatchPolicyCube3 = Catlass::Gemm::MmadAtlasA2FAGCube3;
    using L1TileShapeCube3 = GemmShape<256, 128, 256>;
    using L0TileShapeCube3 = L1TileShapeCube3;

    using BlockMmadMLACube3 = Catlass::Gemm::Block::BlockMmad<DispatchPolicyCube3, L1TileShapeCube3, L0TileShapeCube3, dSTypeCube3, QTypeCube3, dKTypeCube3>;

    // epilogue 1
    using ElementdQVector = float;
    using LayoutdQVector = layout::RowMajor;
    using dqVectorType = Catlass::Gemm::GemmType<ElementdQVector, LayoutdQVector>;
    using ElementdKVector = float;
    using LayoutdKVector = layout::RowMajor;
    using dKVectorType = Catlass::Gemm::GemmType<ElementdKVector, LayoutdKVector>;

    using EpilogueAtlasA2FAGPre = Catlass::Epilogue::EpilogueAtlasA2FAGPre;
    using EpilogueFAGPre = Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGPre, dqVectorType, dKVectorType, dKVectorType>;

    using EpilogueAtlasA2FAGSfmg = Catlass::Epilogue::EpilogueAtlasA2FAGSfmg;
    using EpilogueFAGSfmg = Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGSfmg, dqVectorType, dKVectorType, dKVectorType>;

    using EpilogueAtlasA2FAGOp = Catlass::Epilogue::EpilogueAtlasA2FAGOp;
    using EpilogueFAGOp = Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGOp, dqVectorType, dKVectorType, dKVectorType>;

    using EpilogueAtlasA2FAGPost = Catlass::Epilogue::EpilogueAtlasA2FAGPost;
    using EpilogueFAGPost = Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGPost, dqVectorType, dKVectorType, dKVectorType>;

    // Kernel level
    using FAGKernel = FAGKernel<BlockMmadMLACube1, BlockMmadMLACube2, BlockMmadMLACube3, EpilogueFAGPre, EpilogueFAGSfmg, EpilogueFAGOp, EpilogueFAGPost>;
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