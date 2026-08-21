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

#ifndef RAIN_FUSION_ATTENTION_TILING_DATA_H
#define RAIN_FUSION_ATTENTION_TILING_DATA_H

#include <string>

struct RfaTilingData {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t embeddingSize;
    uint32_t blockSize;
    uint32_t maxNumBlocksPerBatch;
    uint32_t firstBatchTaskNum;
    uint32_t totalTaskNum;
    uint32_t maskType;
    float scaleValue;
    uint32_t totalQBlocks;
    uint32_t firstQBlockNum;

    // 稀疏分块参数 (blockShape)
    uint64_t blockShapeX;  // block的x维度(Q方向)
    uint64_t blockShapeY;  // block的y维度(KV方向)

    uint32_t maxKvBlockNum;  // 最大KV块数量（selectIdx的最后一维）

    uint32_t qInputLayout;  // query Layout: 0=TND, 1=BNSD
    uint32_t kvInputLayout;  // KV Layout: 0=TND, 1=BNSD

    // 当actualSeqLengths为nullptr时，maxQSeqlen也用作统一的qseqlen值
    uint32_t maxQSeqlen;  // BNSD格式Q的第三维（S维度），或统一的qseqlen值
    uint32_t maxKvSeqlen;  // BNSD格式KV的第三维（S维度），或统一的kvseqlen值
    uint32_t isVarLen;  // 是否使用统一的qseqlen值（1=是，0=否）
    int32_t innerPrecise;
    uint64_t workSpaceSize;

    // base tile info
    uint32_t qBaseTile;
    uint32_t kvBaseTile;

    // Getter functions
    uint32_t get_batch() const
    {
        return batch;
    }
    uint32_t get_numHeads() const
    {
        return numHeads;
    }
    uint32_t get_kvHeads() const
    {
        return kvHeads;
    }
    uint32_t get_embeddingSize() const
    {
        return embeddingSize;
    }
    uint32_t get_blockSize() const
    {
        return blockSize;
    }
    uint32_t get_maxNumBlocksPerBatch() const
    {
        return maxNumBlocksPerBatch;
    }
    uint32_t get_firstBatchTaskNum() const
    {
        return firstBatchTaskNum;
    }
    uint32_t get_totalTaskNum() const
    {
        return totalTaskNum;
    }
    uint32_t get_maskType() const
    {
        return maskType;
    }
    float get_scaleValue() const
    {
        return scaleValue;
    }
    uint32_t get_totalQBlocks() const
    {
        return totalQBlocks;
    }
    uint32_t get_firstQBlockNum() const
    {
        return firstQBlockNum;
    }
    uint64_t get_blockShapeX() const
    {
        return blockShapeX;
    }
    uint64_t get_blockShapeY() const
    {
        return blockShapeY;
    }
    uint32_t get_maxKvBlockNum() const
    {
        return maxKvBlockNum;
    }
    uint32_t get_qInputLayout() const
    {
        return qInputLayout;
    }
    uint32_t get_kvInputLayout() const
    {
        return kvInputLayout;
    }
    uint32_t get_maxQSeqlen() const
    {
        return maxQSeqlen;
    }
    uint32_t get_maxKvSeqlen() const
    {
        return maxKvSeqlen;
    }
    uint32_t get_isVarLen() const
    {
        return isVarLen;
    }
    int32_t get_innerPrecise() const
    {
        return innerPrecise;
    }
    uint64_t get_workSpaceSize() const
    {
        return workSpaceSize;
    }
    uint32_t get_qBaseTile() const
    {
        return qBaseTile;
    }
    uint32_t get_kvBaseTile() const
    {
        return kvBaseTile;
    }

    // Setter functions
    void set_batch(uint32_t value)
    {
        batch = value;
    }
    void set_numHeads(uint32_t value)
    {
        numHeads = value;
    }
    void set_kvHeads(uint32_t value)
    {
        kvHeads = value;
    }
    void set_embeddingSize(uint32_t value)
    {
        embeddingSize = value;
    }
    void set_blockSize(uint32_t value)
    {
        blockSize = value;
    }
    void set_maxNumBlocksPerBatch(uint32_t value)
    {
        maxNumBlocksPerBatch = value;
    }
    void set_firstBatchTaskNum(uint32_t value)
    {
        firstBatchTaskNum = value;
    }
    void set_totalTaskNum(uint32_t value)
    {
        totalTaskNum = value;
    }
    void set_maskType(uint32_t value)
    {
        maskType = value;
    }
    void set_scaleValue(float value)
    {
        scaleValue = value;
    }
    void set_totalQBlocks(uint32_t value)
    {
        totalQBlocks = value;
    }
    void set_firstQBlockNum(uint32_t value)
    {
        firstQBlockNum = value;
    }
    void set_blockShapeX(uint64_t value)
    {
        blockShapeX = value;
    }
    void set_blockShapeY(uint64_t value)
    {
        blockShapeY = value;
    }
    void set_maxKvBlockNum(uint32_t value)
    {
        maxKvBlockNum = value;
    }
    void set_qInputLayout(uint32_t value)
    {
        qInputLayout = value;
    }
    void set_kvInputLayout(uint32_t value)
    {
        kvInputLayout = value;
    }
    void set_maxQSeqlen(uint32_t value)
    {
        maxQSeqlen = value;
    }
    void set_maxKvSeqlen(uint32_t value)
    {
        maxKvSeqlen = value;
    }
    void set_isVarLen(uint32_t value)
    {
        isVarLen = value;
    }
    void set_innerPrecise(int32_t value)
    {
        innerPrecise = value;
    }
    void set_workSpaceSize(uint64_t value)
    {
        workSpaceSize = value;
    }
    void set_qBaseTile(uint32_t value)
    {
        qBaseTile = value;
    }
    void set_kvBaseTile(uint32_t value)
    {
        kvBaseTile = value;
    }
};

#endif
