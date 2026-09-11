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

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "golden.hpp"
#include "helper.hpp"
#include "x_attention_device.hpp"
#include "x_attention_tiling.hpp"

using namespace std;

struct Options {
    static constexpr const char* HELP =
        "Usage: 78_x_attention batch beamSize sharedKvSeqLen numHeads kvHeads embeddingSize maxDecodeStep "
        "decodeStep cacheMode [--dtype half|bf16 --datapath DATA_PATH --device DEVICE_ID]\n"
        "  cacheMode=0: shared paged cache + unshared continuous cache\n"
        "  cacheMode=1: shared continuous cache + unshared paged cache\n";
    static constexpr uint32_t MIN_ARGS = 10;

    uint32_t batch{0};
    uint32_t beamSize{0};
    uint32_t sharedKvSeqLen{0};
    uint32_t numHeads{0};
    uint32_t kvHeads{0};
    uint32_t embeddingSize{0};
    uint32_t maxDecodeStep{0};
    uint32_t decodeStep{0};
    uint32_t cacheMode{0};
    uint32_t deviceId{0};
    string dataType{"half"};
    string dataPath{"../../examples/78_x_attention/data"};

    int Parse(int argc, const char** argv)
    {
        if (argc < static_cast<int>(MIN_ARGS)) {
            cerr << HELP;
            return -1;
        }
        uint32_t index = 1;
        batch = static_cast<uint32_t>(stoul(argv[index++]));
        beamSize = static_cast<uint32_t>(stoul(argv[index++]));
        sharedKvSeqLen = static_cast<uint32_t>(stoul(argv[index++]));
        numHeads = static_cast<uint32_t>(stoul(argv[index++]));
        kvHeads = static_cast<uint32_t>(stoul(argv[index++]));
        embeddingSize = static_cast<uint32_t>(stoul(argv[index++]));
        maxDecodeStep = static_cast<uint32_t>(stoul(argv[index++]));
        decodeStep = static_cast<uint32_t>(stoul(argv[index++]));
        cacheMode = static_cast<uint32_t>(stoul(argv[index++]));

        while (index < static_cast<uint32_t>(argc)) {
            string flag = argv[index++];
            if (index >= static_cast<uint32_t>(argc)) {
                cerr << "Missing value for " << flag << '\n' << HELP;
                return -1;
            }
            if (flag == "--dtype") {
                dataType = argv[index++];
            } else if (flag == "--datapath") {
                dataPath = argv[index++];
            } else if (flag == "--device") {
                deviceId = static_cast<uint32_t>(stoul(argv[index++]));
            } else {
                cerr << "Unknown option: " << flag << '\n' << HELP;
                return -1;
            }
        }
        if (dataType != "half" && dataType != "bf16") {
            cerr << "dtype must be 'half' or 'bf16'.\n";
            return -1;
        }
        if (cacheMode > 1) {
            cerr << "cacheMode must be 0 or 1.\n";
            return -1;
        }
        if (decodeStep == 0 || decodeStep > maxDecodeStep) {
            cerr << "decodeStep must be in [1, maxDecodeStep].\n";
            return -1;
        }
        return 0;
    }
};

struct HostDeviceBuffer {
    uint8_t* host{nullptr};
    uint8_t* device{nullptr};
    uint64_t size{0};
};

static void Allocate(HostDeviceBuffer& buffer, uint64_t size)
{
    buffer.size = size;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&buffer.host), size));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&buffer.device), size, ACL_MEM_MALLOC_HUGE_FIRST));
}

static void Load(HostDeviceBuffer& buffer, const string& path, uint64_t size)
{
    Allocate(buffer, size);
    ReadFile(path, buffer.host, size);
    ACL_CHECK(aclrtMemcpy(buffer.device, size, buffer.host, size, ACL_MEMCPY_HOST_TO_DEVICE));
}

static void Release(HostDeviceBuffer& buffer)
{
    if (buffer.host != nullptr) {
        ACL_CHECK(aclrtFreeHost(buffer.host));
    }
    if (buffer.device != nullptr) {
        ACL_CHECK(aclrtFree(buffer.device));
    }
    buffer = {};
}

static bool Run(const Options& options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    bool sharedPaged = options.cacheMode == 0;
    uint32_t blockSize = XAttentionTiling::PAGED_BLOCK_SIZE;
    uint32_t maxBlocksPerBatch =
        XAttentionTiling::CeilDivHost(options.sharedKvSeqLen, XAttentionTiling::PAGED_BLOCK_SIZE);
    uint32_t numBlocks = options.batch * maxBlocksPerBatch;
    uint64_t elementBytes = sizeof(uint16_t);
    uint64_t numTokens = static_cast<uint64_t>(options.batch) * options.beamSize;
    uint64_t outputElements = numTokens * options.numHeads * options.embeddingSize;
    uint64_t outputSize = outputElements * elementBytes;
    uint64_t sharedElements =
        sharedPaged ?
            static_cast<uint64_t>(numBlocks) * blockSize * options.kvHeads * options.embeddingSize :
            static_cast<uint64_t>(options.batch) * options.sharedKvSeqLen * options.kvHeads * options.embeddingSize;
    uint64_t unsharedElements = static_cast<uint64_t>(options.batch) * options.beamSize * options.kvHeads *
                                options.maxDecodeStep * options.embeddingSize;

    HostDeviceBuffer query;
    HostDeviceBuffer sharedKey;
    HostDeviceBuffer sharedValue;
    HostDeviceBuffer unsharedKey;
    HostDeviceBuffer unsharedValue;
    HostDeviceBuffer sharedBlockTable;
    HostDeviceBuffer unsharedBlockTable;
    HostDeviceBuffer sharedKvLens;
    HostDeviceBuffer decodeStep;

    Load(query, options.dataPath + "/query.bin", outputSize);
    Load(sharedKey, options.dataPath + "/shared_key.bin", sharedElements * elementBytes);
    Load(sharedValue, options.dataPath + "/shared_value.bin", sharedElements * elementBytes);
    Load(unsharedKey, options.dataPath + "/unshared_key.bin", unsharedElements * elementBytes);
    Load(unsharedValue, options.dataPath + "/unshared_value.bin", unsharedElements * elementBytes);
    Load(
        sharedBlockTable, options.dataPath + "/shared_block_table.bin",
        static_cast<uint64_t>(options.batch) * maxBlocksPerBatch * sizeof(int32_t));
    Load(
        unsharedBlockTable, options.dataPath + "/unshared_block_table.bin",
        static_cast<uint64_t>(options.batch) * sizeof(int32_t));
    Load(
        sharedKvLens, options.dataPath + "/shared_kv_lens.bin", static_cast<uint64_t>(options.batch) * sizeof(int32_t));
    Load(decodeStep, options.dataPath + "/decode_step.bin", sizeof(int32_t));

    XAttentionTiling::Context context;
    context.batch = options.batch;
    context.beamSize = options.beamSize;
    context.numHeads = options.numHeads;
    context.kvHeads = options.kvHeads;
    context.embeddingSize = options.embeddingSize;
    context.sharedKvSeqLen = options.sharedKvSeqLen;
    context.maxDecodeStep = options.maxDecodeStep;
    context.coreNum = coreNum;
    context.sharedPaged = sharedPaged;

    XAttentionTilingData tilingData;
    uint64_t workspaceSize = 0;
    uint64_t tilingKey = XAttentionTiling::GetTiling(context, tilingData, workspaceSize);
    tilingKey += options.dataType == "bf16" ? 2 : 0;
    cout << "tilingKey: " << tilingKey << ", workspace: " << workspaceSize << " bytes\n";

    uint8_t* outputDevice{nullptr};
    uint8_t* workspaceDevice{nullptr};
    uint8_t* tilingDevice{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&outputDevice), outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&workspaceDevice), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof(tilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(outputDevice, outputSize, 0, outputSize));
    ACL_CHECK(aclrtMemset(workspaceDevice, workspaceSize, 0, workspaceSize));
    ACL_CHECK(
        aclrtMemcpy(tilingDevice, sizeof(tilingData), &tilingData, sizeof(tilingData), ACL_MEMCPY_HOST_TO_DEVICE));

    uint64_t hardwareSyncAddr{0};
    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));

    if (options.dataType == "half") {
        if (sharedPaged) {
            XAttention<half, true, false><<<coreNum, nullptr, stream>>>(
                hardwareSyncAddr, query.device, sharedKey.device, sharedValue.device, unsharedKey.device,
                unsharedValue.device, unsharedBlockTable.device, sharedKvLens.device, decodeStep.device,
                sharedBlockTable.device, outputDevice, workspaceDevice, tilingDevice);
        } else {
            XAttention<half, false, true><<<coreNum, nullptr, stream>>>(
                hardwareSyncAddr, query.device, sharedKey.device, sharedValue.device, unsharedKey.device,
                unsharedValue.device, unsharedBlockTable.device, sharedKvLens.device, decodeStep.device,
                sharedBlockTable.device, outputDevice, workspaceDevice, tilingDevice);
        }
    } else if (sharedPaged) {
        XAttention<bfloat16_t, true, false><<<coreNum, nullptr, stream>>>(
            hardwareSyncAddr, query.device, sharedKey.device, sharedValue.device, unsharedKey.device,
            unsharedValue.device, unsharedBlockTable.device, sharedKvLens.device, decodeStep.device,
            sharedBlockTable.device, outputDevice, workspaceDevice, tilingDevice);
    } else {
        XAttention<bfloat16_t, false, true><<<coreNum, nullptr, stream>>>(
            hardwareSyncAddr, query.device, sharedKey.device, sharedValue.device, unsharedKey.device,
            unsharedValue.device, unsharedBlockTable.device, sharedKvLens.device, decodeStep.device,
            sharedBlockTable.device, outputDevice, workspaceDevice, tilingDevice);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    vector<float> golden(outputElements);
    ReadFile(options.dataPath + "/golden.bin", golden.data(), outputElements * sizeof(float));
    vector<uint64_t> errorIndices;
    if (options.dataType == "half") {
        vector<fp16_t> output(outputElements);
        ACL_CHECK(aclrtMemcpy(output.data(), outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
        errorIndices = golden::CompareData(output, golden, options.sharedKvSeqLen + options.decodeStep);
    } else {
        vector<bfloat16> output(outputElements);
        ACL_CHECK(aclrtMemcpy(output.data(), outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
        errorIndices = golden::CompareData(output, golden, options.sharedKvSeqLen + options.decodeStep);
    }

    bool success = errorIndices.empty();
    cout << (success ? "Compare success." : "Compare failed.") << '\n';
    if (!success) {
        cerr << "Error count: " << errorIndices.size() << '\n';
    }

    Release(query);
    Release(sharedKey);
    Release(sharedValue);
    Release(unsharedKey);
    Release(unsharedValue);
    Release(sharedBlockTable);
    Release(unsharedBlockTable);
    Release(sharedKvLens);
    Release(decodeStep);
    ACL_CHECK(aclrtFree(outputDevice));
    ACL_CHECK(aclrtFree(workspaceDevice));
    ACL_CHECK(aclrtFree(tilingDevice));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
    return success;
}

int main(int argc, const char** argv)
{
    try {
        Options options;
        if (options.Parse(argc, argv) != 0) {
            return EXIT_FAILURE;
        }
        return Run(options) ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const exception& error) {
        cerr << "[ERROR] " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
