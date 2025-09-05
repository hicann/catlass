#include <acl/acl.h>
#include "catlass_kernel.h"
#include "helper.hpp"
#include "mla_tiling.cpp"

#include "kernel/mla_kernel.cpp"
#include "kernel/mla_kernel_tp1_spec.cpp"

namespace CatlassKernel {

void LaunchMLA(uint32_t blockNum, aclrtStream stream, MLAKernelInfo mlaKernelInfo)
{
    // construct MLAInfo on host
    MLATiling::MLAInfo mlaInfo;
    mlaInfo.numTokens = mlaKernelInfo.numTokens;
    mlaInfo.numHeads = mlaKernelInfo.numHeads;
    mlaInfo.embeddingSize = mlaKernelInfo.embeddingSize;
    mlaInfo.embeddingSizeRope = mlaKernelInfo.embeddingSizeRope;
    mlaInfo.numBlocks = mlaKernelInfo.numBlocks;
    mlaInfo.blockSize = mlaKernelInfo.blockSize;
    mlaInfo.maxKvSeqlen = mlaKernelInfo.maxKvSeqlen;
    mlaInfo.kvHeads = mlaKernelInfo.kvHeads;
    mlaInfo.batch = mlaKernelInfo.batch;
    mlaInfo.qSeqLen = mlaKernelInfo.qSeqLen;
    mlaInfo.kvSeqLen = mlaKernelInfo.kvSeqLen;

    // for convenience later
    int32_t batch = mlaInfo.batch;
    int32_t numTokens = mlaInfo.numTokens;
    int32_t numHeads = mlaInfo.numHeads;
    int32_t embeddingSize = mlaInfo.embeddingSize;
    
    float softmaxScale = mlaKernelInfo.softmaxScale;

    // get MLA tiling on host
    int32_t specStraKey = (numHeads == MLATiling::NUM128) ? 1 : 0;
    uint32_t tilingSize = (MLATiling::TILING_HEAD_SIZE + batch * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    if (specStraKey) {
        tilingSize = (MLATiling::TILING_HEAD_SIZE + numTokens * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    }

    void *tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    MLATiling::GetMLATilingParam(mlaInfo, blockNum, (uint32_t *)tilingHost, softmaxScale);

    // copy tiling info to device
    uint8_t *tilingDevice;
    ACL_CHECK(aclrtMalloc((void **)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t aicCoreNum = blockNum;
    uint32_t kvSplitCoreNum = *((uint32_t *)tilingHost + MLATiling::TILING_KVCORENUM);
    uint64_t oFdSize = embeddingSize * numHeads * numTokens * kvSplitCoreNum * sizeof(float);
    uint64_t lSize = numTokens * numHeads * kvSplitCoreNum * sizeof(float);

    // prepare for device kernel launch
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // 3 bits for tilingKey(specStraKey : 1, dTypeKey : 2)
    int32_t tilingKey = (specStraKey << MLATiling::NUM2) + mlaKernelInfo.dTypeKey;

    // temp variable name for device kernel
    uint32_t blockDim = blockNum;
    uint8_t *qDevice = mlaKernelInfo.inputAddr[0];
    uint8_t *qRopeDevice = mlaKernelInfo.inputAddr[1];
    uint8_t *kDevice = mlaKernelInfo.inputAddr[2];
    uint8_t *kRopeDevice = mlaKernelInfo.inputAddr[3];
    uint8_t *blockTableDevice = mlaKernelInfo.inputAddr[4];
    uint8_t *sDevice = mlaKernelInfo.inputAddr[5];
    uint8_t *pDevice = mlaKernelInfo.inputAddr[6];
    
    uint8_t *oDevice = mlaKernelInfo.outputAddr[0];
    uint8_t *oTmpDevice = mlaKernelInfo.outputAddr[1];
    uint8_t *globaloDevice = mlaKernelInfo.outputAddr[2];
    uint8_t *lDevice = mlaKernelInfo.outputAddr[3];
    uint8_t *oCoreTmpDevice = mlaKernelInfo.outputAddr[4];

    // use Tp1Spec kernel to get better performance when numHeads = 128
    switch (tilingKey) {
        case 0:
            MLAFp16<<<blockDim, nullptr, stream>>>(fftsAddr, qDevice, qRopeDevice, kDevice, kRopeDevice,
                                                   blockTableDevice, oDevice, sDevice, pDevice, oTmpDevice,
                                                   globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        case 1:
            MLABf16<<<blockDim, nullptr, stream>>>(fftsAddr, qDevice, qRopeDevice, kDevice, kRopeDevice,
                                                   blockTableDevice, oDevice, sDevice, pDevice, oTmpDevice,
                                                   globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        case 4:
            MLATp1SpecFp16<<<blockDim, nullptr, stream>>>(fftsAddr, qDevice, qRopeDevice, kDevice, kRopeDevice,
                                                          blockTableDevice, oDevice, sDevice, pDevice, oTmpDevice,
                                                          globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        case 5:
            MLATp1SpecBf16<<<blockDim, nullptr, stream>>>(fftsAddr, qDevice, qRopeDevice, kDevice, kRopeDevice,
                                                          blockTableDevice, oDevice, sDevice, pDevice, oTmpDevice,
                                                          globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        default:
            break;
    }

    // memory clean-up
    aclrtFreeHost(tilingHost);
    aclrtFree(tilingDevice);
}

std::vector<uint64_t> PrepareMLAParams(uint32_t blockNum, aclrtStream stream, MLAKernelInfo mlaKernelInfo)
{
    // construct MLAInfo on host
    MLATiling::MLAInfo mlaInfo;
    mlaInfo.numTokens = mlaKernelInfo.numTokens;
    mlaInfo.numHeads = mlaKernelInfo.numHeads;
    mlaInfo.embeddingSize = mlaKernelInfo.embeddingSize;
    mlaInfo.embeddingSizeRope = mlaKernelInfo.embeddingSizeRope;
    mlaInfo.numBlocks = mlaKernelInfo.numBlocks;
    mlaInfo.blockSize = mlaKernelInfo.blockSize;
    mlaInfo.maxKvSeqlen = mlaKernelInfo.maxKvSeqlen;
    mlaInfo.kvHeads = mlaKernelInfo.kvHeads;
    mlaInfo.batch = mlaKernelInfo.batch;
    mlaInfo.qSeqLen = mlaKernelInfo.qSeqLen;
    mlaInfo.kvSeqLen = mlaKernelInfo.kvSeqLen;

    // for convenience later
    int32_t batch = mlaInfo.batch;
    int32_t numTokens = mlaInfo.numTokens;
    int32_t numHeads = mlaInfo.numHeads;
    int32_t embeddingSize = mlaInfo.embeddingSize;

    // get MLA tiling on host
    int32_t specStraKey = (numHeads == MLATiling::NUM128) ? 1 : 0;
    uint32_t tilingSize = (MLATiling::TILING_HEAD_SIZE + batch * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    if (specStraKey) {
        tilingSize = (MLATiling::TILING_HEAD_SIZE + numTokens * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    }

    void *tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    MLATiling::GetMLATilingParam(mlaInfo, blockNum, (uint32_t *)tilingHost, 0.0);

    uint64_t aicCoreNum = (uint64_t)blockNum;
    uint64_t kvSplitCoreNum = (uint64_t)(*((uint32_t *)tilingHost + MLATiling::TILING_KVCORENUM));
    uint64_t oFdSize = embeddingSize * numHeads * numTokens * kvSplitCoreNum * sizeof(float);
    uint64_t lSize = numTokens * numHeads * kvSplitCoreNum * sizeof(float);

    aclrtFreeHost(tilingHost);

    return {aicCoreNum, kvSplitCoreNum, oFdSize, lSize};
}

} // namespace CatlassKernel