#include <acl/acl.h>
#include "catlass_kernel.h"
#include "helper.hpp"
#include "mla_tiling.cpp"

#include "kernel/mla_kernel.cpp"
#include "kernel/mla_kernel_tp1_spec.cpp"

namespace CatlassKernel {

// using namespace Catlass;

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

    // get MLA tiling on host
    int32_t specStraKey = (numHeads == MLATiling::NUM128) ? 1 : 0;
    uint32_t tilingSize = (MLATiling::TILING_HEAD_SIZE + batch * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    if (specStraKey) {
        tilingSize = (MLATiling::TILING_HEAD_SIZE + numTokens * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    }

    void *tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    MLATiling::GetMLATilingParam(mlaInfo, blockNum, (uint32_t *)tilingHost);

    // copy tiling info to device
    uint8_t *tilingDevice;
    ACL_CHECK(aclrtMalloc((void **)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // temporary variable on device
    // TODO: prepare all temp variables during pytorch preprocessing, to reduce overhead
    uint32_t aicCoreNum = blockNum;
    uint32_t kvSplitCoreNum = *((uint32_t *)tilingHost + MLATiling::TILING_KVCORENUM);
    uint64_t oFdSize = embeddingSize * numHeads * numTokens * kvSplitCoreNum * sizeof(float);
    uint64_t lSize = numTokens * numHeads * kvSplitCoreNum * sizeof(float);

    uint8_t *oCoreTmpDevice;
    ACL_CHECK(aclrtMalloc((void **)(&oCoreTmpDevice), oFdSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *lDevice;
    ACL_CHECK(aclrtMalloc((void **)(&lDevice), lSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *sDevice;
    ACL_CHECK(aclrtMalloc((void **)(&sDevice),
                          aicCoreNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * MLATiling::NUM2,
                          ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *pDevice;
    ACL_CHECK(aclrtMalloc((void **)(&pDevice),
                          aicCoreNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * 2 * MLATiling::NUM2,
                          ACL_MEM_MALLOC_HUGE_FIRST));  // NOTE: sizeof(fp16_t) = 2

    uint8_t *oTmpDevice;
    ACL_CHECK(aclrtMalloc((void **)(&oTmpDevice),
                          aicCoreNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * MLATiling::NUM2,
                          ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *globaloDevice;
    ACL_CHECK(aclrtMalloc((void **)(&globaloDevice), aicCoreNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float),
                          ACL_MEM_MALLOC_HUGE_FIRST));

    // prepare for device kernel launch
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // key to select kernel variant
    std::cout << "dTypeKey : " << mlaKernelInfo.dTypeKey << std::endl; 

    // 3 bits for tilingKey(specStraKey : 1, dTypeKey : 2)
    int32_t tilingKey = (specStraKey << MLATiling::NUM2) + mlaKernelInfo.dTypeKey;
    std::cout << "tilingKey : " << tilingKey << std::endl; 

    // temp variable name for device kernel
    uint32_t blockDim = blockNum;
    uint8_t *qDevice = mlaKernelInfo.inputAddr[0];
    uint8_t *qRopeDevice = mlaKernelInfo.inputAddr[1];
    uint8_t *kDevice = mlaKernelInfo.inputAddr[2];
    uint8_t *kRopeDevice = mlaKernelInfo.inputAddr[3];
    uint8_t *blockTableDevice = mlaKernelInfo.inputAddr[4];
    uint8_t *oDevice = mlaKernelInfo.outputAddr[0];

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
    ACL_CHECK(aclrtSynchronizeStream(stream));  // TODO: remove the sync for timing

    // memory clean-up
    aclrtFreeHost(tilingHost);
    aclrtFree(tilingDevice);
    aclrtFree(oCoreTmpDevice);
    aclrtFree(lDevice);
    aclrtFree(sDevice);
    aclrtFree(pDevice);
    aclrtFree(oTmpDevice);
    aclrtFree(globaloDevice);
}
} // namespace CatlassKernel
