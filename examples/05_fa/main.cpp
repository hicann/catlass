/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Helper methods to check for errors
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <cstring>
#include "helper.hpp"
#include "golden.hpp"
#include "fa_kernel.cpp"
#include "fp16_t.h"

using namespace std;
using fp16_t = op::fp16_t;

namespace UnpadFATiling {
const int32_t TILING_HEAD_NUM = 16;
const int32_t TILING_PARA_NUM = 24;
const int32_t TILING_BATCHSIZE_INDEX = 0;
const int32_t TILING_MAXSEQLEN_INDEX = 1;
const int32_t TILING_QHEAD_INDEX = 2;
const int32_t TILING_EMBEDDING_INDEX = 3;
const int32_t TILING_KVHEAD_INDEX = 4;
const int32_t TILING_TOR_INDEX = 5;
const int32_t TILING_BATCHMASK_INDEX = 6;
const int32_t TILING_MASKSTRIDE_INDEX = 7;
const int32_t TILING_ISTRIUMASK_INDEX = 8;
const int32_t TILING_TOTALQTILENUM_INDEX = 9;
const int32_t TILING_QADDR_INDEX = 2;
const int32_t TILING_KADDR_INDEX = 4;
const int32_t TILING_VADDR_INDEX = 6;
const int32_t TILING_OADDR_INDEX = 8;
const int32_t TILING_CURBATCHTOTALQTILENUM_INDEX = 10;
const int32_t CONST_2 = 2;

struct UnpadFAInfo {
    int32_t batchSize;
    int32_t qHead;
    int32_t embedding;
    int32_t maxSeqlen;
    int32_t kvHead;
    int32_t batchMask;
    int32_t maskStride;
    int32_t *qSeq;
    int32_t *kvSeq;
    int32_t isTriuMask;
};

int32_t GetFATilingParam(const UnpadFAInfo faInfo, uint32_t &blockDim, uint32_t *tilingHost)
{
    if (tilingHost == nullptr || faInfo.qSeq == nullptr || faInfo.kvSeq == nullptr) {
        cerr << "[ERROR] pointer tilingHost or seq is nullptr." << endl;
        return -1;
    }
    const uint32_t MAX_EMBEDDING = 128;
    if (faInfo.embedding > MAX_EMBEDDING) {
        cerr << "[ERROR] embedding > 128 is not supported." << endl;
        return -1;
    }

    uint64_t qAddrOffset = 0;
    uint64_t kAddrOffset = 0;
    uint64_t vAddrOffset = 0;
    uint64_t oAddrOffset = 0;

    float tor = static_cast<float>(1.0 / sqrt(1.0 * faInfo.embedding));
    uint32_t *torPtr = reinterpret_cast<uint32_t *>(&tor);

    tilingHost[TILING_BATCHSIZE_INDEX] = faInfo.batchSize;
    tilingHost[TILING_MAXSEQLEN_INDEX] = faInfo.maxSeqlen;
    tilingHost[TILING_QHEAD_INDEX] = faInfo.qHead;
    tilingHost[TILING_EMBEDDING_INDEX] = faInfo.embedding;
    tilingHost[TILING_KVHEAD_INDEX] = faInfo.kvHead;
    tilingHost[TILING_TOR_INDEX] = *torPtr;
    tilingHost[TILING_BATCHMASK_INDEX] = faInfo.batchMask;
    tilingHost[TILING_MASKSTRIDE_INDEX] = faInfo.maskStride;
    tilingHost[TILING_ISTRIUMASK_INDEX] = faInfo.isTriuMask;

    // Must be same with L1TileShape::M and L1TileShape::N in fa_kernel.cpp
    int32_t mTile = 128;
    int32_t nTile = 128;

    int64_t *tilingHostI64 = reinterpret_cast<int64_t *>(tilingHost);

    int32_t totalQTileNum = 0;
    for (int32_t batchIdx = 0; batchIdx < faInfo.batchSize; batchIdx++) {
        int32_t qSeqlen = *(faInfo.qSeq + batchIdx);
        int32_t kvSeqlen = *(faInfo.kvSeq + batchIdx);

        int32_t curQTileNum = (qSeqlen + mTile - 1) / mTile;
        if (qSeqlen != 0) {
            totalQTileNum += curQTileNum;
        }

        tilingHost[TILING_HEAD_NUM + batchIdx * TILING_PARA_NUM] = qSeqlen;
        tilingHost[TILING_HEAD_NUM + batchIdx * TILING_PARA_NUM + 1] = kvSeqlen;
        tilingHostI64[(TILING_HEAD_NUM + batchIdx * TILING_PARA_NUM + TILING_QADDR_INDEX) / CONST_2] = qAddrOffset;
        tilingHostI64[(TILING_HEAD_NUM + batchIdx * TILING_PARA_NUM + TILING_KADDR_INDEX) / CONST_2] = kAddrOffset;
        tilingHostI64[(TILING_HEAD_NUM + batchIdx * TILING_PARA_NUM + TILING_VADDR_INDEX) / CONST_2] = vAddrOffset;
        tilingHostI64[(TILING_HEAD_NUM + batchIdx * TILING_PARA_NUM + TILING_OADDR_INDEX) / CONST_2] = oAddrOffset;
        tilingHost[TILING_HEAD_NUM + batchIdx * TILING_PARA_NUM + TILING_CURBATCHTOTALQTILENUM_INDEX] = totalQTileNum;

        qAddrOffset += qSeqlen * faInfo.qHead * faInfo.embedding;
        kAddrOffset += faInfo.maxSeqlen * faInfo.kvHead * faInfo.embedding;
        vAddrOffset += faInfo.maxSeqlen * faInfo.kvHead * faInfo.embedding;
        oAddrOffset += qSeqlen * faInfo.qHead * faInfo.embedding;
    }
    tilingHost[TILING_TOTALQTILENUM_INDEX] = totalQTileNum;

    uint32_t processNum = totalQTileNum * faInfo.qHead;
    blockDim = (processNum > blockDim) ? blockDim : processNum;
    return 0;
}

} // namespace UnpadFATiling


/**
 * Function for read file.
 */
bool ReadFile(const string &filePath, void *buffer, size_t bufferSize)
{
    if (buffer == nullptr) {
        printf("Read file %s failed. Buffer is nullptr.\n", filePath.c_str());
        return false;
    }

    // Open file
    ifstream fd(filePath, ios::binary);
    if (!fd) {
        printf("Open file failed. path = %s.\n", filePath.c_str());
        return false;
    }

    // Load file data in buffer
    filebuf *buf = fd.rdbuf();
    size_t size = buf->pubseekoff(0, ios::end, ios::in);
    if (size == 0) {
        printf("File %s size is 0\n", filePath.c_str());
        return false;
    }
    if (size > bufferSize) {
        printf("File %s size is larger than buffer size.\n", filePath.c_str());
        return false;
    }
    buf->pubseekpos(0, ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    return true;
}

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER =
        "Usage: fa batch seqlen qHead groupNum embed maxSeqlen [--datapath DATA_PATH --device DEVICE_ID]\n";
    static constexpr auto MIN_ARGS = 7;

    // Define default value.
    uint32_t batch{0}, seqlen{0}, qHead{0}, groupNum{0}, embed{0}, maxSeqlen{0};
    uint32_t deviceId{0};
    string dataPath = "../../examples/05_fa/data";

    Options() = default;

    // Define function to parse the command-line arguments.
    int Parse(int argc, const char **argv)
    {
        // The number of arguments must >= 7.
        if (argc < MIN_ARGS) {
            printf(HELPER);
            return -1;
        }

        // Allocate arguments to parameters.
        uint32_t argIndex = 1;
        batch = atoi(argv[argIndex++]);
        seqlen = atoi(argv[argIndex++]);
        qHead = atoi(argv[argIndex++]);
        groupNum = atoi(argv[argIndex++]);
        embed = atoi(argv[argIndex++]);
        maxSeqlen = atoi(argv[argIndex++]);
        while (argIndex < argc) {
            string flag = string(argv[argIndex++]);
            if (flag == "--datapath") {
                dataPath = string(argv[argIndex++]);
            } else if (flag == "--device") {
                deviceId = atoi(argv[argIndex++]);
            } else {
                printf(HELPER);
                return -1;
            }
        }

        return 0;
    }

    // Define function to print arguments.
    string ToString() const
    {
        stringstream ss;
        ss << "{ batch: " << batch << ", seqlen: " << seqlen << ", qHead: " << qHead << ", groupNum: " << groupNum <<
           ", embed: " << embed << ", maxSeqlen: " << maxSeqlen << ", deviceId: " << deviceId << " }";
        return ss.str();
    }
};

void AllocMem(uint8_t **host, uint8_t **device, size_t size)
{
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(host), size));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(device), size, ACL_MEM_MALLOC_HUGE_FIRST));
}

void FreeMem(uint8_t *host, uint8_t *device)
{
    ACL_CHECK(aclrtFreeHost(host));
    ACL_CHECK(aclrtFree(device));
}

// Allocate several matrices in NPU device memory and call a
// ACT FA kernel.
void Run(const Options &options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Parameters initialization.
    uint32_t batch = options.batch;
    uint32_t seqlen = options.seqlen;
    uint32_t qHead = options.qHead;
    uint32_t groupNum = options.groupNum;
    uint32_t embed = options.embed;
    uint32_t maxSeqlen = options.maxSeqlen;
    string dataPath = options.dataPath;

    ifstream fd(dataPath + "/q_ntokens.bin", ios::binary);
    if (!fd) {
        printf("No data file in the path, please check the path,"
               "or run [python <SOURCE_DIR>/examples/05_fa/gen_data.py] first!\n");
        // Destroy specified Stream and reset device.
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return;
    }

    printf("Running fa: batch=%d, seqlen=%d, qHead=%d, groupNum=%d, embed=%d, maxSeqlen=%d...\n", \
           batch, seqlen, qHead, groupNum, embed, maxSeqlen);

    // read qNtokens num
    void *qNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t));
    int32_t qNtokensNum = static_cast<int32_t *>(qNtokens)[0];

    // read qSeq
    void *qSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&qSeq, batch * sizeof(int32_t)));
    ReadFile(dataPath + "/q_seqlen.bin", qSeq, batch * sizeof(int32_t));

    // read kvSeq num
    void *kvSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&kvSeq, batch * sizeof(int32_t)));
    ReadFile(dataPath + "/kv_seqlen.bin", kvSeq, batch * sizeof(int32_t));

    uint64_t qoSize = (uint64_t)qNtokensNum * (uint64_t)qHead * (uint64_t)embed  * sizeof(fp16_t);
    uint64_t kvSize = (uint64_t)batch * (uint64_t)maxSeqlen * (uint64_t)groupNum * (uint64_t)embed * sizeof(fp16_t);
    uint64_t maskSize = (uint64_t)maxSeqlen * (uint64_t)maxSeqlen * sizeof(fp16_t);
    uint32_t tilingSize = (UnpadFATiling::TILING_HEAD_NUM + batch * UnpadFATiling::TILING_PARA_NUM) * sizeof(int32_t);

    // Allocate matrices in host and device memory and load Matrix q.
    uint8_t *qHost;
    uint8_t *qDevice;
    AllocMem(&qHost, &qDevice, qoSize);
    ReadFile(dataPath + "/q.bin", qHost, qoSize);
    ACL_CHECK(aclrtMemcpy(qDevice, qoSize, qHost, qoSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix k.
    uint8_t *kHost;
    uint8_t *kDevice;
    AllocMem(&kHost, &kDevice, kvSize);
    ReadFile(dataPath + "/k.bin", kHost, kvSize);
    ACL_CHECK(aclrtMemcpy(kDevice, kvSize, kHost, kvSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load v.
    uint8_t *vHost;
    uint8_t *vDevice;
    AllocMem(&vHost, &vDevice, kvSize);
    ReadFile(dataPath + "/v.bin", vHost, kvSize);
    ACL_CHECK(aclrtMemcpy(vDevice, kvSize, vHost, kvSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load mask.
    uint8_t *maskHost;
    uint8_t *maskDevice;
    AllocMem(&maskHost, &maskDevice, maskSize);
    ReadFile(dataPath + "/mask.bin", maskHost, maskSize);
    ACL_CHECK(aclrtMemcpy(maskDevice, maskSize, maskHost, maskSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in device memory for workspace.
    uint8_t *sDevice;
    ACL_CHECK(
        aclrtMalloc((void **)(&sDevice), aicCoreNum * WORKSPACE_ELENUM * sizeof(fp16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *pDevice;
    ACL_CHECK(
        aclrtMalloc((void **)(&pDevice), aicCoreNum * WORKSPACE_ELENUM * sizeof(fp16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *oTmpDevice;
    ACL_CHECK(
        aclrtMalloc((void **)(&oTmpDevice), aicCoreNum * WORKSPACE_ELENUM * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *oDevice;
    ACL_CHECK(aclrtMalloc((void **)(&oDevice), qoSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *tilingDevice;
    ACL_CHECK(aclrtMalloc((void **)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // get tiling
    void *tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    uint32_t blockDim = aicCoreNum;

    UnpadFATiling::UnpadFAInfo faInfo;
    faInfo.batchSize = batch;
    faInfo.qHead = qHead;
    faInfo.embedding = embed;
    faInfo.maxSeqlen = maxSeqlen;
    faInfo.kvHead = groupNum;
    faInfo.batchMask = 0;
    faInfo.maskStride = maxSeqlen;
    faInfo.qSeq = static_cast<int32_t *>(qSeq);
    faInfo.kvSeq = static_cast<int32_t *>(kvSeq);
    faInfo.isTriuMask = 1;
    UnpadFATiling::GetFATilingParam(faInfo, blockDim, (uint32_t *)tilingHost);

    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    FA<<<blockDim, nullptr, stream>>>(
        fftsAddr, qDevice, kDevice, vDevice, maskDevice, oDevice, sDevice, pDevice, oTmpDevice, tilingDevice);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // Copy the result from device to host
    vector<fp16_t> oHost(qoSize / sizeof(fp16_t));
    ACL_CHECK(aclrtMemcpy(oHost.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // Compute the golden result
    vector<float> goldenHost(qoSize / sizeof(fp16_t));
    const size_t goldenSize = qoSize * 2;
    ReadFile(dataPath + "/golden.bin", goldenHost.data(), goldenSize);

    // Compare the result
    vector<uint64_t> errorIndices = golden::CompareData(oHost, goldenHost, seqlen);
    if (errorIndices.empty()) {
        cout << "Compare success." << endl;
    } else {
        cerr << "Compare failed. Error count: " << errorIndices.size() << endl;
    }

    // Free host memory allocations.
    FreeMem(qHost, qDevice);
    FreeMem(kHost, kDevice);
    FreeMem(vHost, vDevice);
    FreeMem(maskHost, maskDevice);
    aclrtFree(oDevice);
    aclrtFree(tilingDevice);
    aclrtFree(sDevice);
    aclrtFree(pDevice);
    aclrtFree(oTmpDevice);
    aclrtFreeHost(tilingHost);
    aclrtFreeHost(qNtokens);
    aclrtFreeHost(qSeq);
    aclrtFreeHost(kvSeq);

    // Destroy specified Stream and reset device.
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

/// Entry point to fa example.
// usage: fa batch seqlen qHead groupNum embed maxSeqlen [--datapath DATA_PATH --device DEVICE_ID]

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}