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

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "rfa_tilingdata.h"

namespace optiling {

enum class MaskType : uint32_t
{
    NO_MASK = 0
};

enum class DataType : uint32_t
{
    FP16 = 0,
    BF16 = 1
};

struct RfaContext {
    int32_t batch = 0;
    int32_t numHeads = 0;
    int32_t kvHeads = 0;
    int64_t maxQSeqlen = 0;
    int64_t maxKvSeqlen = 0;
    int32_t embeddingSize = 0;
    int32_t embeddingSizeV = 0;
    int64_t blockShapeX = 128;
    int64_t blockShapeY = 128;
    uint32_t maxKvBlockNum = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    DataType dataType = DataType::FP16;
    std::string qInputLayout = "BNSD";
    std::string kvInputLayout = "BNSD";
    bool isVarLen = false;
    const int64_t* qSeqlenList{nullptr};
    const int64_t* kvSeqlenList{nullptr};
    int32_t innerPrecise = 0;
    MaskType maskType = MaskType::NO_MASK;
    int32_t blockSize = 128;
    float scaleValue = 0.0;
};

class RfaTiling {
public:
    RfaTiling() = default;
    explicit RfaTiling(const RfaContext& faInfo);

    void DoTiling(RfaTilingData& tilingdata);

    void SetCoreNum(uint32_t coreNum)
    {
        this->coreNum_ = coreNum;
    }

    uint32_t GetCoreNum()
    {
        return this->coreNum_;
    }

    uint64_t GetTilingKey();

private:
    uint32_t GetCurQSTileNum(int64_t curQSeqlen);
    void FillBasicTilingData(RfaTilingData& tilingdata);
    void CalculateTaskSplit(RfaTilingData& tilingdata);
    void CalculateWorkSpace(RfaTilingData& tilingdata);

private:
    RfaContext rfaInfo_;
    uint32_t coreNum_;
    uint32_t qBaseTile_ = 128;
    uint32_t kvBaseTile_ = 256;
};

RfaTiling::RfaTiling(const RfaContext& faInfo) : rfaInfo_(faInfo)
{}

void RfaTiling::FillBasicTilingData(RfaTilingData& tilingdata)
{
    tilingdata.set_batch(rfaInfo_.batch);
    tilingdata.set_numHeads(rfaInfo_.numHeads);
    tilingdata.set_kvHeads(rfaInfo_.kvHeads);
    tilingdata.set_embeddingSize(rfaInfo_.embeddingSize);
    tilingdata.set_blockSize(rfaInfo_.blockSize);
    tilingdata.set_maxNumBlocksPerBatch(rfaInfo_.maxNumBlocksPerBatch);
    tilingdata.set_maxKvBlockNum(rfaInfo_.maxKvBlockNum);
    tilingdata.set_maskType(static_cast<uint32_t>(rfaInfo_.maskType));
    tilingdata.set_scaleValue(rfaInfo_.scaleValue);

    tilingdata.set_blockShapeX(rfaInfo_.blockShapeX);
    tilingdata.set_blockShapeY(rfaInfo_.blockShapeY);

    if (rfaInfo_.qInputLayout == "TND") {
        tilingdata.set_qInputLayout(0);
    } else if (rfaInfo_.qInputLayout == "BNSD") {
        tilingdata.set_qInputLayout(1);
    }
    if (rfaInfo_.kvInputLayout == "TND") {
        tilingdata.set_kvInputLayout(0);
    } else if (rfaInfo_.kvInputLayout == "BNSD") {
        tilingdata.set_kvInputLayout(1);
    }
    tilingdata.set_maxQSeqlen(rfaInfo_.maxQSeqlen);
    tilingdata.set_maxKvSeqlen(rfaInfo_.maxKvSeqlen);
    tilingdata.set_isVarLen(rfaInfo_.isVarLen ? 1 : 0);
    tilingdata.set_innerPrecise(rfaInfo_.innerPrecise);

    tilingdata.set_qBaseTile(qBaseTile_);
    tilingdata.set_kvBaseTile(kvBaseTile_);
}

uint32_t RfaTiling::GetCurQSTileNum(int64_t curQSeqlen)
{
    uint32_t fullXBlockNum = curQSeqlen / rfaInfo_.blockShapeX;
    uint32_t tailXBlockSize = curQSeqlen % rfaInfo_.blockShapeX;
    uint32_t qSTileNumPerFullXBlock = (rfaInfo_.blockShapeX + qBaseTile_ - 1) / qBaseTile_;
    uint32_t qSTileNumTailXBlock = (tailXBlockSize + qBaseTile_ - 1) / qBaseTile_;
    uint32_t curQSTileNum = qSTileNumPerFullXBlock * fullXBlockNum + qSTileNumTailXBlock;
    return curQSTileNum;
}

void RfaTiling::CalculateTaskSplit(RfaTilingData& tilingdata)
{
    uint32_t totalTaskNum = 0;
    uint32_t totalQBlocks = 0;
    qBaseTile_ = (rfaInfo_.blockShapeX > 128) ? 128 : rfaInfo_.blockShapeX;
    kvBaseTile_ = 256;

    for (uint32_t bIdx = 0; bIdx < rfaInfo_.batch; bIdx++) {
        int64_t curQSeqlen = rfaInfo_.isVarLen ? rfaInfo_.qSeqlenList[bIdx] : rfaInfo_.maxQSeqlen;

        uint32_t curQBlockNum = (curQSeqlen + rfaInfo_.blockShapeX - 1) / rfaInfo_.blockShapeX * rfaInfo_.numHeads;
        uint32_t curQSTileNum = GetCurQSTileNum(curQSeqlen);
        uint32_t curBatchTaskNum = curQSTileNum * rfaInfo_.numHeads;
        totalTaskNum += curBatchTaskNum;
        totalQBlocks += curQBlockNum;

        if (bIdx == 0) {
            tilingdata.set_firstBatchTaskNum(curBatchTaskNum);
            tilingdata.set_firstQBlockNum(curQBlockNum);
        }
    }
    tilingdata.set_totalTaskNum(totalTaskNum);
    tilingdata.set_totalQBlocks(totalQBlocks);
}

void RfaTiling::CalculateWorkSpace(RfaTilingData& tilingdata)
{
    uint64_t workSpaceSize = 0;
    tilingdata.set_workSpaceSize(workSpaceSize);
}

uint64_t RfaTiling::GetTilingKey()
{
    // RFA基础值（Operator Category = 905）
    uint64_t tilingKey = 9050000000000000ULL;

    // Data Type
    if (rfaInfo_.dataType == DataType::FP16) {
        tilingKey += 0;  // 00 for FP16
    } else if (rfaInfo_.dataType == DataType::BF16) {
        tilingKey += 22220ULL;
    }

    // KV Layout
    if (rfaInfo_.kvInputLayout == "TND") {
        tilingKey += 30000000ULL;
    } else if (rfaInfo_.kvInputLayout == "BNSD") {
        tilingKey += 50000000ULL;
    }

    if (rfaInfo_.innerPrecise == 0) {
        tilingKey += 400000ULL; // 0: low prec online softmax & high prec rescale O
    }

    // Q Layout
    if (rfaInfo_.qInputLayout == "TND") {
        tilingKey += 2ULL;
    } else if (rfaInfo_.qInputLayout == "BNSD") {
        tilingKey += 3ULL;
    }

    return tilingKey;
}

void RfaTiling::DoTiling(RfaTilingData& tilingdata)
{
    FillBasicTilingData(tilingdata);
    CalculateTaskSplit(tilingdata);
    CalculateWorkSpace(tilingdata);
}

} // namespace optiling
