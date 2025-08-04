
#ifndef SHARED_LIB_CATLASS_KERNEL_H
#define SHARED_LIB_CATLASS_KERNEL_H

#include <acl/acl.h>
#include <vector>
#include <string>

namespace CatlassKernel {

struct MLAKernelInfo {
    std::vector<uint8_t *> inputAddr;
    std::vector<uint8_t *> outputAddr;
    int32_t batch;
    int32_t numHeads;
    int32_t embeddingSize;
    int32_t embeddingSizeRope;
    int32_t numTokens;
    int32_t kvHeads;
    int32_t numBlocks;
    int32_t blockSize;
    int32_t maxKvSeqlen;
    int32_t *kvSeqLen{nullptr};
    int32_t *qSeqLen{nullptr};
    int32_t dTypeKey;
};

void LaunchMLA(uint32_t blockNum, aclrtStream stream, MLAKernelInfo mlaKernelInfo);
}

#endif // SHARED_LIB_CATLASS_KERNEL_H
