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

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

// Helper methods to check for errors
#include "rfa_kernel.cpp"
#include "rfa_tiling.cpp"
#include "golden.hpp"
#include "helper.hpp"

using namespace std;
using namespace optiling;

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER =
        "Usage: rfa batch qSeqlen kvSeqlen numHeads kvHeads embeddingSize blockShapeX blockShapeY "
        "dtype qInputLayout kvInputLayout isVariedLen [--datapath DATA_PATH --device DEVICE_ID]\n";
    static constexpr auto MIN_ARGS = 14;

    // Define default value.
    uint32_t batch{0};
    uint32_t qSeqlen{0};
    uint32_t kvSeqlen{0};
    uint32_t numHeads{0};
    uint32_t kvHeads{0};
    uint32_t embeddingSize{0};
    int64_t blockShapeX{128};
    int64_t blockShapeY{128};
    string dataType = "half";
    string qInputLayout = "BNSD";
    string kvInputLayout = "BNSD";
    uint32_t isVariedLen{0};
    uint32_t deviceId{0};

    string dataPath = "../../examples/81_ascend950_rain_fusion_attention/data";

    Options() = default;

    // Define function to parse the command-line arguments.
    int Parse(int argc, const char** argv)
    {
        // The number of arguments must >= 14.
        std::cout << "argc: " << argc << std::endl;
        if (argc < MIN_ARGS) {
            printf(HELPER);
            return -1;
        }

        // Allocate arguments to parameters.
        uint32_t argIndex = 1;
        batch = atoi(argv[argIndex++]);
        qSeqlen = atoi(argv[argIndex++]);
        kvSeqlen = atoi(argv[argIndex++]);
        numHeads = atoi(argv[argIndex++]);
        kvHeads = atoi(argv[argIndex++]);
        embeddingSize = atoi(argv[argIndex++]);
        blockShapeX = atoi(argv[argIndex++]);
        blockShapeY = atoi(argv[argIndex++]);
        dataType = string(argv[argIndex++]);
        qInputLayout = string(argv[argIndex++]);
        kvInputLayout = string(argv[argIndex++]);
        isVariedLen = atoi(argv[argIndex++]);

        while (argIndex < argc) {
            string flag = string(argv[argIndex++]);
            if (flag == "--datapath") {
                dataPath = string(argv[argIndex++]);
            } else if (flag == "--device") {
                deviceId = atoi(argv[argIndex++]);
            } else {
                std::cout << "flag: " << flag << std::endl;
                printf(HELPER);
                return -1;
            }
        }
        return 0;
    }
};

static void AllocMem(uint8_t** host, uint8_t** device, size_t size)
{
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(host), size));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(device), size, ACL_MEM_MALLOC_HUGE_FIRST));
}

static void FreeMem(uint8_t* host, uint8_t* device)
{
    ACL_CHECK(aclrtFreeHost(host));
    ACL_CHECK(aclrtFree(device));
}

// Allocate several matrices in NPU device memory and call a
// CATLASS RFA kernel.
static void Run(const Options& options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Parameters initialization
    int32_t batch = options.batch;
    int64_t qSeqlen = options.qSeqlen;
    int64_t kvSeqlen = options.kvSeqlen;
    int32_t numHeads = options.numHeads;
    int32_t kvHeads = options.kvHeads;
    int32_t embeddingSize = options.embeddingSize;
    int64_t blockShapeX = options.blockShapeX;
    int64_t blockShapeY = options.blockShapeY;
    string dataType = options.dataType;
    string qInputLayout = options.qInputLayout;
    string kvInputLayout = options.kvInputLayout;
    int32_t isVariedLen = options.isVariedLen;
    string dataPath = options.dataPath;

    int64_t maxKvSeqlen = kvSeqlen;
    int64_t maxKvBlockNum = (maxKvSeqlen + blockShapeY - 1) / blockShapeY;

    if ((dataType != "half") && (dataType != "bf16")) {
        cerr << "[ERROR] dtype must be 'half' or 'bf16'." << endl;
        return;
    }
    if ((qInputLayout != "BNSD") && (qInputLayout != "TND")) {
        cerr << "[ERROR] qInputLayout must be 'BNSD' or 'TND'." << endl;
        return;
    }
    if (qInputLayout != kvInputLayout) {
        cerr << "[ERROR] qInputLayout and kvInputLayout must be the same." << endl;
        return;
    }
    if (isVariedLen == 1 && qInputLayout == "BNSD") {
        cerr << "[ERROR] inputLayout must be 'TND' when isVariedLen is 1." << endl;
        return;
    }
    if (embeddingSize != 64 && embeddingSize != 128) {
        cerr << "[ERROR] embeddingSize must be 64 or 128, got " << embeddingSize << endl;
        return;
    }

    // read qNtokens num
    void* qNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t));
    int32_t numTokens = static_cast<int32_t*>(qNtokens)[0];

    void* kvNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&kvNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/kv_ntokens.bin", kvNtokens, 1 * sizeof(int32_t));
    int32_t kvNumTokens = static_cast<int32_t*>(kvNtokens)[0];

    void* totalQsBlock = nullptr;
    ACL_CHECK(aclrtMallocHost(&totalQsBlock, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/total_qs_block_num.bin", totalQsBlock, 1 * sizeof(int32_t));
    int32_t totalQsBlockNum = static_cast<int32_t*>(totalQsBlock)[0];

    // input size
    uint64_t qSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t kSize = (uint64_t)kvNumTokens * (uint64_t)kvHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t vSize = (uint64_t)kvNumTokens * (uint64_t)kvHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t seqArraySize = batch * sizeof(int64_t);
    uint64_t selectIdxSize = (uint64_t)totalQsBlockNum * (uint64_t)numHeads * maxKvBlockNum * sizeof(int64_t);
    uint64_t selectNumIdxSize = (uint64_t)totalQsBlockNum * (uint64_t)numHeads * sizeof(int64_t);
    // output size
    uint64_t oSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t lseSize = (uint64_t)numTokens * (uint64_t)numHeads * sizeof(float);
    uint32_t tilingSize = sizeof(RfaTilingData);

    // Allocate matrices in host and device memory.
    uint8_t* qSeqHost;
    uint8_t* qSeqDevice;
    AllocMem(&qSeqHost, &qSeqDevice, seqArraySize);
    ReadFile(dataPath + "/q_seqlen_list.bin", qSeqHost, seqArraySize);
    ACL_CHECK(aclrtMemcpy(qSeqDevice, seqArraySize, qSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate kv_seqlen_list in host and device memory.
    uint8_t* kvSeqHost;
    uint8_t* kvSeqDevice;
    AllocMem(&kvSeqHost, &kvSeqDevice, seqArraySize);
    ReadFile(dataPath + "/kv_seqlen_list.bin", kvSeqHost, seqArraySize);
    ACL_CHECK(aclrtMemcpy(kvSeqDevice, seqArraySize, kvSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate and load Matrix q.
    uint8_t* qHost;
    uint8_t* qDevice;
    AllocMem(&qHost, &qDevice, qSize);
    ReadFile(dataPath + "/q.bin", qHost, qSize);
    ACL_CHECK(aclrtMemcpy(qDevice, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate and load Matrix k.
    uint8_t* kHost;
    uint8_t* kDevice;
    AllocMem(&kHost, &kDevice, kSize);
    ReadFile(dataPath + "/k.bin", kHost, kSize);
    ACL_CHECK(aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate and load Matrix v.
    uint8_t* vHost;
    uint8_t* vDevice;
    AllocMem(&vHost, &vDevice, vSize);
    ReadFile(dataPath + "/v.bin", vHost, vSize);
    ACL_CHECK(aclrtMemcpy(vDevice, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // mask and blockTable: not used (passed as nullptr)
    uint8_t* maskDevice{nullptr};
    uint8_t* blockTableDevice{nullptr};

    // Allocate and load select_idx, [totalQsBlockNum, numHeads, maxKvBlockNum]
    uint8_t* selectIdxHost;
    uint8_t* selectIdxDevice;
    AllocMem(&selectIdxHost, &selectIdxDevice, selectIdxSize);
    ReadFile(dataPath + "/select_idx.bin", selectIdxHost, selectIdxSize);
    ACL_CHECK(aclrtMemcpy(selectIdxDevice, selectIdxSize, selectIdxHost, selectIdxSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate and load select_num_idx, [totalQsBlockNum, numHeads]
    uint8_t* selectNumIdxHost;
    uint8_t* selectNumIdxDevice;
    AllocMem(&selectNumIdxHost, &selectNumIdxDevice, selectNumIdxSize);
    ReadFile(dataPath + "/select_num_idx.bin", selectNumIdxHost, selectNumIdxSize);
    ACL_CHECK(aclrtMemcpy(
        selectNumIdxDevice, selectNumIdxSize, selectNumIdxHost, selectNumIdxSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // ========================================================================
    // Set up RfaContext for tiling
    // ========================================================================
    RfaTilingData rfaTilingData;
    RfaContext rfaContext;

    rfaContext.batch = batch;
    rfaContext.numHeads = numHeads;
    rfaContext.kvHeads = kvHeads;
    rfaContext.maxQSeqlen = qSeqlen;
    rfaContext.maxKvSeqlen = kvSeqlen;
    rfaContext.embeddingSize = embeddingSize;
    rfaContext.blockShapeX = blockShapeX;
    rfaContext.blockShapeY = blockShapeY;
    rfaContext.maxKvBlockNum = maxKvBlockNum;
    rfaContext.maxNumBlocksPerBatch = maxKvBlockNum * numHeads;
    rfaContext.dataType = (dataType == "bf16") ? DataType::BF16 : DataType::FP16;
    rfaContext.qInputLayout = qInputLayout;
    rfaContext.kvInputLayout = kvInputLayout;
    rfaContext.isVarLen = (isVariedLen == 1);
    rfaContext.qSeqlenList = reinterpret_cast<int64_t*>(qSeqHost);
    rfaContext.kvSeqlenList = reinterpret_cast<int64_t*>(kvSeqHost);
    rfaContext.scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * embeddingSize));
    rfaContext.blockSize = 128;
    rfaContext.innerPrecise = 0;
    rfaContext.maskType = MaskType::NO_MASK;

    // ========================================================================
    // Tiling
    // ========================================================================
    RfaTiling rfaTiling(rfaContext);
    rfaTiling.SetCoreNum(aicCoreNum);
    rfaTiling.DoTiling(rfaTilingData);
    uint64_t tilingKey = rfaTiling.GetTilingKey();

    // Allocate workspace
    uint32_t workSpaceSize = static_cast<uint32_t>(rfaTilingData.get_workSpaceSize());
    uint8_t* workspaceDevice{nullptr};
    if (workSpaceSize > 0) {
        ACL_CHECK(aclrtMalloc((void**)(&workspaceDevice), workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    uint8_t* oDevice{nullptr};
    ACL_CHECK(aclrtMalloc((void**)(&oDevice), oSize * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t* lseDevice{nullptr};

    // Allocate blockShape device memory: [blockShapeX, blockShapeY]
    uint64_t blockShapeSize = 2 * sizeof(int64_t);
    uint8_t* blockShapeHost;
    uint8_t* blockShapeDevice;
    AllocMem(&blockShapeHost, &blockShapeDevice, blockShapeSize);
    reinterpret_cast<int64_t*>(blockShapeHost)[0] = blockShapeX;
    reinterpret_cast<int64_t*>(blockShapeHost)[1] = blockShapeY;
    ACL_CHECK(aclrtMemcpy(blockShapeDevice, blockShapeSize, blockShapeHost, blockShapeSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Copy tiling data to device
    void* tilingHost = nullptr;
    tilingHost = reinterpret_cast<void*>(&rfaTilingData);
    uint8_t* tilingDevice;
    ACL_CHECK(aclrtMalloc((void**)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t blockDim = aicCoreNum;

    for (int i = 0; i < 1; i++) {
        if (qInputLayout == "TND" && kvInputLayout == "TND") {
            if (dataType == "half") {
                RainFusionAttention950<half, half, Epilogue::LseMode::NONE, 0, 0><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, selectIdxDevice, selectNumIdxDevice, blockShapeDevice, qSeqDevice,
                    kvSeqDevice, maskDevice, blockTableDevice, oDevice, lseDevice, workspaceDevice, tilingDevice);
            } else if (dataType == "bf16") {
                RainFusionAttention950<bfloat16_t, bfloat16_t, Epilogue::LseMode::NONE, 0, 0>
                    <<<blockDim, nullptr, stream>>>(
                        qDevice, kDevice, vDevice, selectIdxDevice, selectNumIdxDevice, blockShapeDevice, qSeqDevice,
                        kvSeqDevice, maskDevice, blockTableDevice, oDevice, lseDevice, workspaceDevice, tilingDevice);
            }
        } else if (qInputLayout == "BNSD" && kvInputLayout == "BNSD") {
            if (dataType == "half") {
                RainFusionAttention950<half, half, Epilogue::LseMode::NONE, 1, 1><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, selectIdxDevice, selectNumIdxDevice, blockShapeDevice, qSeqDevice,
                    kvSeqDevice, maskDevice, blockTableDevice, oDevice, lseDevice, workspaceDevice, tilingDevice);
            } else if (dataType == "bf16") {
                RainFusionAttention950<bfloat16_t, bfloat16_t, Epilogue::LseMode::NONE, 1, 1>
                    <<<blockDim, nullptr, stream>>>(
                        qDevice, kDevice, vDevice, selectIdxDevice, selectNumIdxDevice, blockShapeDevice, qSeqDevice,
                        kvSeqDevice, maskDevice, blockTableDevice, oDevice, lseDevice, workspaceDevice, tilingDevice);
            }
        }

        ACL_CHECK(aclrtSynchronizeStream(stream));
        // Copy the result from device to host
        vector<fp16_t> oHostHalf(oSize / sizeof(fp16_t));
        vector<bfloat16> oHostBf16(oSize / sizeof(bfloat16));
        if (dataType == "half") {
            ACL_CHECK(aclrtMemcpy(oHostHalf.data(), oSize, oDevice, oSize, ACL_MEMCPY_DEVICE_TO_HOST));
        } else if (dataType == "bf16") {
            ACL_CHECK(aclrtMemcpy(oHostBf16.data(), oSize, oDevice, oSize, ACL_MEMCPY_DEVICE_TO_HOST));
        }

        // Compute the golden result
        vector<float> goldenHost(oSize / sizeof(fp16_t));
        const size_t goldenSize = oSize * 2;
        ReadFile(dataPath + "/golden.bin", goldenHost.data(), goldenSize);

        // Compare the result
        vector<uint64_t> errorIndices = (dataType == "half") ? golden::CompareData(oHostHalf, goldenHost, kvSeqlen) :
                                                               golden::CompareData(oHostBf16, goldenHost, kvSeqlen);
        if (errorIndices.empty()) {
            cout << "Compare success." << endl;
        } else {
            if (dataType == "bf16") {
                const float rtolOverThreshold = 1.0f / 128;
                for (uint64_t i = 0; i < errorIndices.size(); ++i) {
                    float actualValue = static_cast<float>(oHostBf16[errorIndices[i]]);
                    float expectValue = goldenHost[errorIndices[i]];
                    float diff = std::fabs(actualValue - expectValue);
                    if (diff <= rtolOverThreshold * std::max(1.0f, std::fabs(expectValue))) {
                        continue;
                    } else {
                        cerr << "Compare failed. Error count: " << errorIndices.size() << endl;
                    }
                }
                cout << "Compare success." << endl;
            } else {
                cerr << "Compare failed. Error count: " << errorIndices.size() << endl;
            }
        }
    }

    // Free host memory allocations.
    FreeMem(qSeqHost, qSeqDevice);
    FreeMem(kvSeqHost, kvSeqDevice);
    FreeMem(qHost, qDevice);
    FreeMem(kHost, kDevice);
    FreeMem(vHost, vDevice);
    FreeMem(selectIdxHost, selectIdxDevice);
    FreeMem(selectNumIdxHost, selectNumIdxDevice);
    FreeMem(blockShapeHost, blockShapeDevice);
    aclrtFree(oDevice);
    aclrtFree(tilingDevice);
    aclrtFree(workspaceDevice);
    aclrtFreeHost(qNtokens);
    aclrtFreeHost(kvNtokens);
    aclrtFreeHost(totalQsBlock);

    // Destroy specified Stream and reset device.
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

// Entry point of the rfa example
int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
