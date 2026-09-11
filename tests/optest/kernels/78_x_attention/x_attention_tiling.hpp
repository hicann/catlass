/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_X_ATTENTION_TILING_HPP
#define OPTEST_X_ATTENTION_TILING_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "x_attention_common.hpp"

namespace XAttentionTiling {

constexpr uint32_t UNSHARED_Q_TILE = 128;
constexpr uint32_t UNSHARED_KV_TILE = 256;
constexpr uint32_t Q_S_BLOCK_TILE = 128;
constexpr uint32_t PAGED_BLOCK_SIZE = 128;
constexpr uint32_t FLOAT_BLOCK_SIZE = 8;

struct Context {
    uint32_t batch{0};
    uint32_t beamSize{0};
    uint32_t numHeads{0};
    uint32_t kvHeads{0};
    uint32_t embeddingSize{0};
    uint32_t sharedKvSeqLen{0};
    uint32_t maxDecodeStep{0};
    uint32_t coreNum{0};
    bool sharedPaged{true};
};

inline uint32_t CeilDivHost(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

inline void Validate(const Context& context)
{
    if (context.batch == 0 || context.beamSize == 0 || context.numHeads == 0 || context.kvHeads == 0) {
        throw std::invalid_argument("batch, beamSize, numHeads and kvHeads must be positive");
    }
    if (context.numHeads % context.kvHeads != 0) {
        throw std::invalid_argument("numHeads must be divisible by kvHeads");
    }
    if (context.embeddingSize != 128) {
        throw std::invalid_argument("x_attention currently requires embeddingSize == 128");
    }
    if (context.sharedKvSeqLen == 0 || context.maxDecodeStep == 0 || context.maxDecodeStep > UNSHARED_KV_TILE) {
        throw std::invalid_argument("sharedKvSeqLen must be positive and maxDecodeStep must be in [1, 256]");
    }
    if (context.coreNum < 2) {
        throw std::invalid_argument("x_attention requires at least two cube cores");
    }
    if (context.numHeads / context.kvHeads > UNSHARED_Q_TILE) {
        throw std::invalid_argument("GQA group size must not exceed 128");
    }
}

inline uint64_t GetTiling(
    const Context& context, XAttentionTilingData& tilingData, uint64_t& workspaceSize)
{
    Validate(context);

    tilingData.numHeads = context.numHeads;
    tilingData.kvHeads = context.kvHeads;
    tilingData.embeddingSize = context.embeddingSize;
    tilingData.batch = context.batch;
    tilingData.beamSize = context.beamSize;
    tilingData.scaleValue = 1.0F / std::sqrt(static_cast<float>(context.embeddingSize));
    tilingData.maskType = 0;
    tilingData.numTokens = context.batch * context.beamSize;
    tilingData.blockSize = PAGED_BLOCK_SIZE;
    tilingData.maxDecodeStep = context.maxDecodeStep;
    tilingData.groupSize = context.numHeads / context.kvHeads;

    tilingData.sharedCoreNum = std::min<uint32_t>(12, context.coreNum - 1);
    tilingData.unsharedCoreNum = context.coreNum - tilingData.sharedCoreNum;

    tilingData.maxNumBlocksPerBatch = CeilDivHost(context.sharedKvSeqLen, PAGED_BLOCK_SIZE);
    tilingData.numBlocks = context.batch * tilingData.maxNumBlocksPerBatch;

    uint32_t qNBlockTile = 1;
    uint32_t qNBlockNumPerGroup = CeilDivHost(tilingData.groupSize, qNBlockTile);
    uint32_t qNBlockNum = qNBlockNumPerGroup * context.kvHeads;
    uint32_t qSBlockNum = CeilDivHost(context.beamSize, Q_S_BLOCK_TILE);
    tilingData.firstSharedBatchTaskNum = qNBlockNum * qSBlockNum;
    tilingData.sharedTotalTaskNum = tilingData.firstSharedBatchTaskNum * context.batch;

    uint32_t totalGroupCount = context.beamSize * context.kvHeads;
    bool unsharedPaged = !context.sharedPaged;
    if (!unsharedPaged) {
        totalGroupCount *= context.batch;
    }
    uint32_t groupCountPerLoop = std::min(
        UNSHARED_Q_TILE / tilingData.groupSize, UNSHARED_KV_TILE / context.maxDecodeStep);
    while (groupCountPerLoop > 1 &&
           (totalGroupCount % groupCountPerLoop != 0 || groupCountPerLoop % FLOAT_BLOCK_SIZE != 0)) {
        --groupCountPerLoop;
    }
    if (groupCountPerLoop == 0) {
        throw std::invalid_argument("unsupported group size and maxDecodeStep combination");
    }
    tilingData.unshareGroupCountPerLoop = groupCountPerLoop;

    uint32_t unsharedTasksPerBatch = totalGroupCount / groupCountPerLoop;
    tilingData.unsharedLoopCountPerBatch = std::max<uint32_t>(1, unsharedTasksPerBatch);
    uint32_t totalUnsharedTasks = unsharedTasksPerBatch;
    if (unsharedPaged) {
        totalUnsharedTasks *= context.batch;
    }

    tilingData.unsharedFullCoreNum = tilingData.unsharedCoreNum;
    tilingData.unsharedTaskNumHead = totalUnsharedTasks / tilingData.unsharedCoreNum;
    tilingData.unsharedTaskNumTail = tilingData.unsharedTaskNumHead;
    uint32_t remainingTasks = totalUnsharedTasks % tilingData.unsharedCoreNum;
    if (remainingTasks != 0) {
        tilingData.unsharedFullCoreNum = remainingTasks;
        ++tilingData.unsharedTaskNumHead;
    }

    uint32_t rowNum = tilingData.numTokens * context.numHeads;
    uint32_t combineCoreNum = std::min(rowNum, context.coreNum);
    tilingData.combineFormerCoreNum = rowNum % combineCoreNum;
    tilingData.combineFormerRowNum = rowNum / combineCoreNum + 1;
    tilingData.combineTailRowNum = rowNum / combineCoreNum;
    tilingData.combineCoreNum = combineCoreNum;

    tilingData.mm1OutSize =
        (static_cast<uint64_t>(tilingData.sharedCoreNum) * WORKSPACE_BLOCK_SIZE_DB +
         static_cast<uint64_t>(tilingData.unsharedCoreNum) * UNSHARED_WORKSPACE_BLOCK_SIZE_DB) *
        NUM3 * sizeof(float);
    tilingData.smOnlineOutSize =
        (static_cast<uint64_t>(tilingData.sharedCoreNum) * WORKSPACE_BLOCK_SIZE_DB +
         static_cast<uint64_t>(tilingData.unsharedCoreNum) * UNSHARED_WORKSPACE_BLOCK_SIZE_DB) *
        NUM3 * sizeof(uint16_t);
    tilingData.mm2OutSize =
        static_cast<uint64_t>(tilingData.sharedCoreNum) * WORKSPACE_BLOCK_SIZE_DB * NUM3 * sizeof(float);
    tilingData.updateSize = 0;

    uint64_t outputElements = static_cast<uint64_t>(tilingData.numTokens) * context.numHeads * context.embeddingSize;
    uint64_t sumMaxSize = static_cast<uint64_t>(tilingData.numTokens) * context.numHeads * sizeof(float) * NUM2;
    uint64_t outputBytes = outputElements * sizeof(uint16_t);
    tilingData.sharedWorkspaceSize = sumMaxSize * FLOAT_BLOCK_SIZE + outputBytes * NUM2;
    uint64_t unsharedWorkspaceSize = sumMaxSize + outputBytes * NUM2;
    workspaceSize = tilingData.mm1OutSize + tilingData.smOnlineOutSize + tilingData.mm2OutSize +
                    tilingData.updateSize + tilingData.sharedWorkspaceSize + unsharedWorkspaceSize;

    uint64_t dtypeIndependentKey = context.sharedPaged ? 8 : 4;
    return dtypeIndependentKey;
}

} // namespace XAttentionTiling

#endif // OPTEST_X_ATTENTION_TILING_HPP
