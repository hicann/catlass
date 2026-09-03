/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#define NO_OVERLAP_IN_MULTI_REPEAT

#include <acl/acl.h>

#include "catlass_kernel_prebuilt.h"
#include "kernel_operator.h"

#include "launcher/hstu_infer_launcher.hpp"

namespace CatlassKernel {

static constexpr uint32_t HSTU_AIC_CORE_NUM = 1;

void Ascend950HstuInfer(uint32_t blockNum, aclrtStream stream, const HstuInferParams &params)
{
    uint32_t enablePagedKv = params.pagedBlockSize > 0 ? 1 : 0;
    bool isNTD = (params.layout == "NTD");

    uint8_t *qSeqDevice = params.inputAddr.at(0);
    uint8_t *kvSeqDevice = params.inputAddr.at(1);
    uint8_t *qDevice = params.inputAddr.at(2);
    uint8_t *kDevice = params.inputAddr.at(3);
    uint8_t *vDevice = params.inputAddr.at(4);
    uint8_t *blockTableDevice = params.inputAddr.at(5);
    uint8_t *oDevice = params.outputAddr.at(0);

    uint32_t blockDim = blockNum > 0 ? blockNum : HSTU_AIC_CORE_NUM;

    if (isNTD) {
        if (enablePagedKv == 0) {
            RunHSTUKernel<half, Catlass::layout::NTD, Catlass::layout::NTD, false>(
                qDevice, kDevice, vDevice, oDevice, qSeqDevice, kvSeqDevice, blockTableDevice, params.batch,
                params.numHeads, params.embeddingSize, params.kvHeads, params.maxKvSeqlen, params.numPagedBlocks,
                params.pagedBlockSize, params.siluScale, params.maskType, stream, blockDim);
        } else {
            RunHSTUKernel<half, Catlass::layout::NTD, Catlass::layout::NHD, true>(
                qDevice, kDevice, vDevice, oDevice, qSeqDevice, kvSeqDevice, blockTableDevice, params.batch,
                params.numHeads, params.embeddingSize, params.kvHeads, params.maxKvSeqlen, params.numPagedBlocks,
                params.pagedBlockSize, params.siluScale, params.maskType, stream, blockDim);
        }
    } else {
        if (enablePagedKv == 0) {
            RunHSTUKernel<half, Catlass::layout::TND, Catlass::layout::TND, false>(
                qDevice, kDevice, vDevice, oDevice, qSeqDevice, kvSeqDevice, blockTableDevice, params.batch,
                params.numHeads, params.embeddingSize, params.kvHeads, params.maxKvSeqlen, params.numPagedBlocks,
                params.pagedBlockSize, params.siluScale, params.maskType, stream, blockDim);
        } else {
            RunHSTUKernel<half, Catlass::layout::TND, Catlass::layout::NHD, true>(
                qDevice, kDevice, vDevice, oDevice, qSeqDevice, kvSeqDevice, blockTableDevice, params.batch,
                params.numHeads, params.embeddingSize, params.kvHeads, params.maxKvSeqlen, params.numPagedBlocks,
                params.pagedBlockSize, params.siluScale, params.maskType, stream, blockDim);
        }
    }
}

} // namespace CatlassKernel
