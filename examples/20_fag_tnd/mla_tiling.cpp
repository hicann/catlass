#include <acl/acl.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstring>

#include "softmax_tiling.cpp"
#include "helper.hpp"
#include "fp16_t.h"
#include "fag_common/common_header.h"

using namespace std;
using fp16_t = op::fp16_t;

namespace MLATiling {
struct MLAInfo {
    float scaleValue;

    int64_t seqQShapeSize;
    int64_t queryShape_0;
    int64_t queryShape_1;
    int64_t queryShape_2;
    int64_t keyShape_0;
    int64_t keyShape_1;
    int64_t valueShape_0;
    int64_t valueShape_1;
};

void printMLATilingData(int64_t *tilingHost) {
    cout << "MLATilingDATA b: " << tilingHost[TILING_B] << endl;
    cout << "MLATilingDATA: t1: " << tilingHost[TILING_T1] << endl;
    cout << "MLATilingDATA: t2: " << tilingHost[TILING_T2] << endl;
    cout << "MLATilingDATA: N1: " << tilingHost[TILING_N1] << endl;
    cout << "MLATilingDATA: N2: " << tilingHost[TILING_N2] << endl;
    cout << "MLATilingDATA: G: " << tilingHost[TILING_G] << endl;
    cout << "MLATilingDATA: D: " << tilingHost[TILING_D] << endl;
    cout << "MLATilingDATA: Q size: " << tilingHost[TILING_Q_SIZE] << endl;
    cout << "MLATilingDATA: kv size: " << tilingHost[TILING_KV_SIZE] << endl;
    cout << "MLATilingDATA: DQ_WORKSPACE_OFFSET: " << tilingHost[TILING_DQ_WORKSPACE_OFFSET] << endl;
    cout << "MLATilingDATA: DK_WORKSPACE_OFFSET: " << tilingHost[TILING_DK_WORKSPACE_OFFSET] << endl;
    cout << "MLATilingDATA: DV_WORKSPACE_OFFSET: " << tilingHost[TILING_DV_WORKSPACE_OFFSET] << endl;
    cout << "MLATilingDATA: SFMG_WORKSPACE_OFFSET: " << tilingHost[TILING_SFMG_WORKSPACE_OFFSET] << endl;
    cout << "MLATilingDATA: MM1_WORKSPACE_OFFSET: " << tilingHost[TILING_MM1_WORKSPACE_OFFSET] << endl;
    cout << "MLATilingDATA: MM2_WORKSPACE_OFFSET: " << tilingHost[TILING_MM2_WORKSPACE_OFFSET] << endl;
    cout << "MLATilingDATA: P_WORKSPACE_OFFSET: " << tilingHost[TILING_P_WORKSPACE_OFFSET] << endl;
    cout << "MLATilingDATA: DS_WORKSPACE_OFFSET: " << tilingHost[TILING_DS_WORKSPACE_OFFSET] << endl;

    float *tilingHostFp = reinterpret_cast<float *>(tilingHost);
    cout << "MLATilingDATA scale value: " << tilingHostFp[TILING_SCALE_VALUE * CONST_2] << endl;
    
    uint32_t *tilingHostU32 = reinterpret_cast<uint32_t *>(tilingHost);
    cout << "MLATilingDATA coreNum: " << tilingHostU32[TILING_CORE_NUM * CONST_2] << endl;
    cout << "MLATilingDATA srcM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2] << endl;
    cout << "MLATilingDATA srcK: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 1] << endl;
    cout << "MLATilingDATA srcSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 2] << endl;
    cout << "MLATilingDATA outMaxM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 3] << endl;
    cout << "MLATilingDATA outMaxK: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 4] << endl;
    cout << "MLATilingDATA outMaxSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 5] << endl;
    cout << "MLATilingDATA splitM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 6] << endl;
    cout << "MLATilingDATA splitK: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 7] << endl;
    cout << "MLATilingDATA SplitSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 8] << endl;
    cout << "MLATilingDATA reduceM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 9] << endl;
    cout << "MLATilingDATA reduceK: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 10] << endl;
    cout << "MLATilingDATA reduceSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 11] << endl;
    cout << "MLATilingDATA rangeM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 12] << endl;
    cout << "MLATilingDATA tailM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 13] << endl;
    cout << "MLATilingDATA tailSplitSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 14] << endl;
    cout << "MLATilingDATA tailReduceSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 15] << endl;

    // softmax grad data
    cout << "MLATilingDATA srcM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2] << endl;
    cout << "MLATilingDATA srcK: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 1] << endl;
    cout << "MLATilingDATA srcSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 2] << endl;
    cout << "MLATilingDATA outMaxM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 3] << endl;
    cout << "MLATilingDATA outMaxK: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 4] << endl;
    cout << "MLATilingDATA outMaxSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 5] << endl;
    cout << "MLATilingDATA splitM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 6] << endl;
    cout << "MLATilingDATA splitK: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 7] << endl;
    cout << "MLATilingDATA SplitSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 8] << endl;
    cout << "MLATilingDATA reduceM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 9] << endl;
    cout << "MLATilingDATA reduceK: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 10] << endl;
    cout << "MLATilingDATA reduceSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 11] << endl;
    cout << "MLATilingDATA rangeM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 12] << endl;
    cout << "MLATilingDATA tailM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 13] << endl;
    cout << "MLATilingDATA tailSplitSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 14] << endl;
    cout << "MLATilingDATA tailReduceSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 15] << endl;
}

int32_t GetFATilingParam(const MLAInfo mlaInfo, uint32_t &blockDim, int64_t *tilingHost)
{
    float *tilingHostFp = reinterpret_cast<float *>(tilingHost);
    tilingHostFp[TILING_SCALE_VALUE * CONST_2] = mlaInfo.scaleValue;

    tilingHost[TILING_B] = mlaInfo.seqQShapeSize;
    tilingHost[TILING_T1] = mlaInfo.queryShape_0;
    tilingHost[TILING_T2] = mlaInfo.keyShape_0;
    tilingHost[TILING_N1] = mlaInfo.queryShape_1;
    tilingHost[TILING_N2] = mlaInfo.keyShape_1;
    tilingHost[TILING_D] = mlaInfo.queryShape_2;

    uint64_t g = mlaInfo.queryShape_1 / mlaInfo.keyShape_1;
    tilingHost[TILING_G] = mlaInfo.queryShape_1 / mlaInfo.keyShape_1;

    int64_t qSize = mlaInfo.queryShape_0 * mlaInfo.keyShape_1 * g * mlaInfo.queryShape_2;
    int64_t kvSize = mlaInfo.keyShape_0 * mlaInfo.keyShape_1 * 1 * mlaInfo.queryShape_2;
    int64_t sfmgSize = mlaInfo.queryShape_0 * mlaInfo.queryShape_1 * 8;

    tilingHost[TILING_Q_SIZE] = qSize;
    tilingHost[TILING_KV_SIZE] = kvSize;

    // Softmax tiling
    constexpr uint32_t tmpBufferSize = 33 * 1024;
    constexpr uint32_t s1VecSize = 64;
    constexpr uint32_t s2VecSize = 128;
    std::vector<uint32_t> softmaxShape = {s1VecSize, s2VecSize};

    SoftMaxTiling softmaxTilingData;
    SoftMaxTilingFunc(
        softmaxShape, sizeof(float), tmpBufferSize, softmaxTilingData);

    // softmaxGrad tiling
    constexpr uint32_t inputBufferLen = 24 * 1024;
    constexpr uint32_t castBufferLen = 48 * 1024; // castBuffer 48K*2=96K
    uint32_t outputBufferLen = (castBufferLen + mlaInfo.queryShape_2 - 1) /  mlaInfo.queryShape_2 * 8;
    uint32_t tempBufferLen = 40 * 1024 - outputBufferLen;

    int64_t singleLoopNBurstNum = inputBufferLen / sizeof(float) / mlaInfo.queryShape_2;
    std::vector<int64_t> softmaxGradShape = {singleLoopNBurstNum, mlaInfo.queryShape_2};

    SoftMaxTiling softmaxGradTilingData;
    SoftMaxGradTilingFunc(softmaxGradShape, sizeof(float), tempBufferLen, 
        softmaxGradTilingData);

    // put SoftMaxData in Tiling
    uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint32_t vectorCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    uint32_t *tilingHostU32 = reinterpret_cast<uint32_t *>(tilingHost);
    tilingHostU32[TILING_CORE_NUM * CONST_2] = vectorCoreNum;

    uint32_t* softmaxTilingDataPtr = reinterpret_cast<uint32_t *>(&softmaxTilingData);
    memcpy(tilingHostU32 + TILING_SOFTMAX_TILING_DATA * CONST_2, softmaxTilingDataPtr, TILING_SOFTMAX_SIZE);
    softmaxTilingDataPtr = reinterpret_cast<uint32_t *>(&softmaxGradTilingData);
    memcpy(tilingHostU32 + TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2, softmaxTilingDataPtr, TILING_SOFTMAX_SIZE);

    // TODO set workspace offset 
    constexpr size_t WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
    constexpr size_t GM_ALIGN = 512;
    constexpr size_t DB_NUM = 2;
    constexpr size_t matmulSize = 16 * 128 * 128;

    size_t workspaceOffset = WORKSPACE_RSV_BYTE;
    // matmal3 q
    tilingHost[TILING_DQ_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + qSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 k
    tilingHost[TILING_DK_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + kvSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 v
    tilingHost[TILING_DV_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + kvSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // sfmg workspace
    tilingHost[TILING_SFMG_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + sfmgSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    // matmal1/matmal2 workspace size
    tilingHost[TILING_MM1_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * sizeof(float) * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tilingHost[TILING_MM2_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * sizeof(float) * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    constexpr uint32_t size_of_half = 2;
    tilingHost[TILING_P_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * size_of_half * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tilingHost[TILING_DS_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * size_of_half * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    return 0;
}

} // namespace UnpadFATiling