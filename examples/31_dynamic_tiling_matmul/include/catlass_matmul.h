#ifndef CATLASS_MATMUL_H
#define CATLASS_MATMUL_H

#include "helper.hpp"
#include "tiling.h"

template <class Dtype>
struct CatlassMatmulDescriptor {
    TilingParams tilingParams;
    uint8_t *dTilingParams;
    TilingKey tilingKey;

    CatlassMatmulDescriptor(uint32_t m, uint32_t n, uint32_t k, size_t strideA, size_t strideB, size_t strideC,
        LayoutTag layoutTagA, LayoutTag layoutTagB, LayoutTag layoutTagC)
    {
        ACL_CHECK(aclrtMalloc((void **)&dTilingParams, sizeof(TilingParams), ACL_MEM_MALLOC_HUGE_FIRST));
    }

    void SetMatmulInfo(
        uint32_t m, uint32_t n, uint32_t k, LayoutTag layoutTagA, LayoutTag layoutTagB, LayoutTag layoutTagC)
    {
        size_t strideA = k;
        size_t strideB = n;
        size_t strideC = n;
        if (static_cast<LayoutTag>(layoutTagA) == LayoutTag::TagColumnMajor) {
            strideA = m;
        }
        if (static_cast<LayoutTag>(layoutTagB) == LayoutTag::TagColumnMajor) {
            strideB = k;
        }
        if (static_cast<LayoutTag>(layoutTagC) == LayoutTag::TagColumnMajor) {
            strideC = m;
        }
        tilingParams.SetParams(m, n, k, strideA, strideB, strideC, layoutTagA, layoutTagB, layoutTagC);
        GetTiling<Dtype>(tilingParams, tilingKey);
    }

    void SetMatmulInfo(uint32_t m, uint32_t n, uint32_t k, size_t strideA, size_t strideB, size_t strideC,
        LayoutTag layoutTagA, LayoutTag layoutTagB, LayoutTag layoutTagC)
    {
        tilingParams.SetParams(m, n, k, strideA, strideB, strideC, layoutTagA, layoutTagB, layoutTagC);
        GetTiling<Dtype>(tilingParams, tilingKey);
    }

    ~CatlassMatmulDescriptor()
    {
        ACL_CHECK(aclrtFree(dTilingParams));
    }
};

template <DType>
size_t CatlassMatmulGetWorkspace(CatlassMatmulDescriptor<Dtype> &desc)
{
    return getWorkspaceFuncMap[desc.tilingKey.value](desc.tilingParams);
}

template <class DType>
void ExecuteCatlassMatmul(
    aclrtStream &stream, uint8_t *dA, uint8_t *dB, uint8_t *dC, uint8_t *dW, CatlassMatmulDescriptor<Dtype> &desc)
{
    uint64_t fftsAddr{0};
    uint64_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    ACL_CHECK(aclrtMemcpy(
        dTilingParams, sizeof(TilingParams), &tilingParams, sizeof(TilingParams), ACL_MEMCPY_HOST_TO_DEVICE));

    launchKernelFuncMap[desc.tilingKey.value](stream, fftsAddr, dA, dB, dC, dW, desc.dTilingParams, desc.tilingParams);
}

#endif  // CATLASS_MATMUL_H