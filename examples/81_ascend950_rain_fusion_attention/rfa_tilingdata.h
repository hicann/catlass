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
    uint32_t batch = 0;
    uint32_t numHeads = 0;
    uint32_t kvHeads = 0;
    uint32_t embeddingSize = 0;
    uint32_t blockSize = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    uint32_t firstBatchTaskNum = 0;
    uint32_t totalTaskNum = 0;
    uint32_t maskType = 0;
    float scaleValue = 0.0f;
    uint32_t totalQBlocks = 0;
    uint32_t firstQBlockNum = 0;

    // 稀疏分块参数 (blockShape)
    int64_t blockShapeX = 128;
    int64_t blockShapeY = 128;

    // 最大KV块数量（selectIdx的最后一维）
    uint32_t maxKvBlockNum = 0;

    // query Layout: 0=TND, 1=BNSD
    uint32_t qInputLayout = 0;
    uint32_t kvInputLayout = 0;

    // 当actualSeqLengths为nullptr时，maxQSeqlen也用作统一的qseqlen值
    // BNSD格式Q的第三维（S维度），或统一的qseqlen值
    uint32_t maxQSeqlen = 0;
    // BNSD格式KV的第三维（S维度），或统一的kvseqlen值
    uint32_t maxKvSeqlen = 0;
    // 是否使用统一的qseqlen值（0=是，1=否）
    uint32_t isVarLen = 0;
    int32_t innerPrecise = 0;
    uint64_t workSpaceSize = 0;

    // base tile info
    uint32_t qBaseTile = 128;
    uint32_t kvBaseTile = 256;

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
    int64_t get_blockShapeX() const
    {
        return blockShapeX;
    }
    int64_t get_blockShapeY() const
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
    void set_blockShapeX(int64_t value)
    {
        blockShapeX = value;
    }
    void set_blockShapeY(int64_t value)
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
