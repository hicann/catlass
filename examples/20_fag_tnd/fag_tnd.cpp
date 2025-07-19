/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <cmath>
#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"
#include "fag_tnd_tiling.cpp"
#include "fag_tnd_kernel.cpp"

using namespace std;
using fp16_t = op::fp16_t;

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
        "Usage: mla N1 N2 D dtype element_of_list_seq [--device DEVICE_ID]\n";
    static constexpr auto MIN_ARGS = 5;

    // Define default value.
    uint32_t N1{0}, N2{0}, D{0};
    string dtype{"fp16_t"};
    std::vector<int64_t> list_seq;
    uint32_t deviceId{0};
    string dataPath = "../../examples/20_fag_tnd/data";

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
        N1 = atoi(argv[argIndex++]);
        N2 = atoi(argv[argIndex++]);
        D = atoi(argv[argIndex++]);
        while (argIndex < argc) {
            if (argIndex == argc - 1) {
                deviceId = atoi(argv[argIndex++]);
            } else {
                list_seq.push_back(atoi(argv[argIndex++]));
            };
        }

        return 0;
    }

    // Define function to print arguments.
    string ToString() const
    {
        stringstream ss;
        ss << "{ N1: " << N1 << ", N2: " << N2 << ", D: " << D << ", dtype: " << dtype <<
            ", "  << ", deviceId: " << deviceId << " }";
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
// ACTLASS MLA kernel.
void Run(const Options &options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint32_t N1 = options.N1;
    uint32_t N2 = options.N2;
    uint32_t D = options.D;
    string dtype = options.dtype;
    std::vector<int64_t> list_seq = options.list_seq;
    string dataPath = options.dataPath;

    ifstream fd(dataPath + "/q.bin", ios::binary);
    if (!fd) {
        printf("No data file in the path, please check the path,"
                "or run [python <SOURCE_DIR>/examples/20_fag_tnd/gen_data.py] first!\n");
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return;
    }

    printf("Running mla: N1=%d, N2=%d, D=%d, ...\n", \
            N1, N2, D);
    printf("list_seq = ");
    for (const auto& num : list_seq) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    MLATiling::MLAInfo mlaInfo;
    
    int64_t sum_of_list = 0;
    for (size_t i = 0; i < list_seq.size(); ++i) {
        sum_of_list += list_seq[i];
    }

    mlaInfo.seqQShapeSize = list_seq.size();
    mlaInfo.queryShape_0 = sum_of_list;
    mlaInfo.keyShape_0 = sum_of_list;
    mlaInfo.queryShape_1 = N1;
    mlaInfo.keyShape_1 = N2;
    mlaInfo.queryShape_2 = D;
    mlaInfo.scaleValue = 1.0 / sqrt(D);

    uint64_t g = N1 / N2;
    uint64_t qSize = sum_of_list * N2 * g * D * sizeof(fp16_t);
    uint64_t kSize = sum_of_list * N2 * 1 * D * sizeof(fp16_t);
    uint64_t vSize = sum_of_list * N2 * 1 * D * sizeof(fp16_t);
    uint64_t dySize = sum_of_list * N1 * D * sizeof(fp16_t);
    uint64_t attenMaskSize = 2048 * 2048 * sizeof(fp16_t);
    uint64_t softMaxMaxSize = sum_of_list * N1 * 8 * sizeof(float);
    uint64_t softMaxSumSize = sum_of_list * N1 * 8 * sizeof(float);
    uint64_t attentionInSize = sum_of_list * N1 * D * sizeof(fp16_t);
    uint64_t actualSeqQlenSize = list_seq.size() * sizeof(int64_t);
    uint64_t actualSeqKvlenSize = list_seq.size() * sizeof(int64_t);

    uint64_t dqSize = qSize;
    uint64_t dkSize = kSize;
    uint64_t dvSize = vSize;
    uint64_t dqRightSize = 16 * 128 * 256 * sizeof(fp16_t);
    uint64_t dkRightSize = 16 * 128 * 128 * sizeof(float);

    uint64_t workspaceSize = (2 * aicCoreNum * 16 * 128 * 128 * 8 * N1) * sizeof(float);

    // Allocate matrices in host and device memory and load Matrix.
    uint8_t *qHost;
    uint8_t *qDevice;
    AllocMem(&qHost, &qDevice, qSize);
    ReadFile(dataPath + "/q.bin", qHost, qSize);
    ACL_CHECK(aclrtMemcpy(qDevice, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *kHost;
    uint8_t *kDevice;
    AllocMem(&kHost, &kDevice, kSize);
    ReadFile(dataPath + "/k.bin", kHost, kSize);
    ACL_CHECK(aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *vHost;
    uint8_t *vDevice;
    AllocMem(&vHost, &vDevice, vSize);
    ReadFile(dataPath + "/v.bin", vHost, vSize);
    ACL_CHECK(aclrtMemcpy(vDevice, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *dyHost;
    uint8_t *dyDevice;
    AllocMem(&dyHost, &dyDevice, dySize);
    ReadFile(dataPath + "/dx.bin", dyHost, dySize);
    ACL_CHECK(aclrtMemcpy(dyDevice, dySize, dyHost, dySize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *attenMaskHost;
    uint8_t *attenMaskDevice;
    AllocMem(&attenMaskHost, &attenMaskDevice, attenMaskSize);
    ReadFile(dataPath + "/atten_mask.bin", attenMaskHost, attenMaskSize);
    ACL_CHECK(aclrtMemcpy(attenMaskDevice, attenMaskSize, attenMaskHost, attenMaskSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *softMaxMaxHost;
    uint8_t *softMaxMaxDevice;
    AllocMem(&softMaxMaxHost, &softMaxMaxDevice, softMaxMaxSize);
    ReadFile(dataPath + "/softmax_max.bin", softMaxMaxHost, softMaxMaxSize);
    ACL_CHECK(aclrtMemcpy(softMaxMaxDevice, softMaxMaxSize, softMaxMaxHost, softMaxMaxSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *softMaxSumHost;
    uint8_t *softMaxSumDevice;
    AllocMem(&softMaxSumHost, &softMaxSumDevice, softMaxSumSize);
    ReadFile(dataPath + "/softmax_sum.bin", softMaxSumHost, softMaxSumSize);
    ACL_CHECK(aclrtMemcpy(softMaxSumDevice, softMaxSumSize, softMaxSumHost, softMaxSumSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *attentionInHost;
    uint8_t *attentionInDevice;
    AllocMem(&attentionInHost, &attentionInDevice, attentionInSize);
    ReadFile(dataPath + "/attention_in.bin", attentionInHost, attentionInSize);
    ACL_CHECK(aclrtMemcpy(attentionInDevice, attentionInSize, attentionInHost, attentionInSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *actualSeqQlenHost;
    uint8_t *actualSeqQlenDevice;
    AllocMem(&actualSeqQlenHost, &actualSeqQlenDevice, actualSeqQlenSize);
    ReadFile(dataPath + "/actual_seq_qlen.bin", actualSeqQlenHost, actualSeqQlenSize);
    ACL_CHECK(aclrtMemcpy(actualSeqQlenDevice, actualSeqQlenSize, actualSeqQlenHost, actualSeqQlenSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *actualSeqKvlenHost;
    uint8_t *actualSeqKvlenDevice;
    AllocMem(&actualSeqKvlenHost, &actualSeqKvlenDevice, actualSeqKvlenSize);
    ReadFile(dataPath + "/actual_seq_kvlen.bin", actualSeqKvlenHost, actualSeqKvlenSize);
    ACL_CHECK(aclrtMemcpy(actualSeqKvlenDevice, actualSeqKvlenSize, actualSeqKvlenHost, actualSeqKvlenSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *dqDevice;
    ACL_CHECK(
        aclrtMalloc((void **)(&dqDevice), dqSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *dkDevice;
    ACL_CHECK(
        aclrtMalloc((void **)(&dkDevice), dkSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *dvDevice;
    ACL_CHECK(
        aclrtMalloc((void **)(&dvDevice), dvSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *dq_rightDevice;
    ACL_CHECK(aclrtMalloc((void **)(&dq_rightDevice), dqRightSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *dk_rightDevice;
    ACL_CHECK(aclrtMalloc((void **)(&dk_rightDevice), dkRightSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *workspaceDevice;
    ACL_CHECK(aclrtMalloc((void **)(&workspaceDevice), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // // get tiling
    uint32_t blockDim = aicCoreNum;
    void *tilingHost = nullptr;
    uint32_t tilingSize = TILING_PARA_NUM * sizeof(int64_t);
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    MLATiling::GetFATilingParam(mlaInfo, blockDim, (int64_t *)tilingHost);
    uint8_t *tilingDevice;
    ACL_CHECK(aclrtMalloc((void **)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    FAG<<<blockDim, nullptr, stream>>>(
        fftsAddr, qDevice, kDevice, vDevice, dyDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
        attenMaskDevice, softMaxMaxDevice, softMaxSumDevice, nullptr, attentionInDevice, nullptr, actualSeqQlenDevice, actualSeqKvlenDevice,
        nullptr, nullptr, dqDevice, dkDevice, dvDevice, dq_rightDevice, dk_rightDevice, workspaceDevice, tilingDevice);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    vector<fp16_t> dqHost(dqSize / sizeof(fp16_t));
    ACL_CHECK(aclrtMemcpy(dqHost.data(), dqSize, dqDevice, dqSize, ACL_MEMCPY_DEVICE_TO_HOST));

    vector<float> goldenDqHost(dqSize / sizeof(fp16_t));
    const size_t goldenDqSize = dqSize * 2;
    ReadFile(dataPath + "/dq_golden.bin", goldenDqHost.data(), goldenDqSize);

    vector<fp16_t> dkHost(dkSize / sizeof(fp16_t));
    ACL_CHECK(aclrtMemcpy(dkHost.data(), dkSize, dkDevice, dkSize, ACL_MEMCPY_DEVICE_TO_HOST));

    vector<float> goldenDkHost(dkSize / sizeof(fp16_t));
    const size_t goldenDkSize = dkSize * 2;
    ReadFile(dataPath + "/dk_golden.bin", goldenDkHost.data(), goldenDkSize);

    vector<fp16_t> dvHost(dvSize / sizeof(fp16_t));
    ACL_CHECK(aclrtMemcpy(dvHost.data(), dvSize, dvDevice, dvSize, ACL_MEMCPY_DEVICE_TO_HOST));

    vector<float> goldenDvHost(dvSize / sizeof(fp16_t));
    const size_t goldenDvSize = dvSize * 2;
    ReadFile(dataPath + "/dv_golden.bin", goldenDvHost.data(), goldenDvSize);

    // Compare the result
    vector<uint64_t> dqerrorIndices = Catlass::golden::CompareData(dqHost, goldenDqHost, dqSize);
    if (dqerrorIndices.empty()) {
        cout << "Compare dq success." << endl;
    } else {
        cerr << "Compare dq failed. Error count: " << dqerrorIndices.size() << endl;
    }

    vector<uint64_t> dkerrorIndices = Catlass::golden::CompareData(dkHost, goldenDkHost, dkSize);
    if (dkerrorIndices.empty()) {
        cout << "Compare dk success." << endl;
    } else {
        cerr << "Compare dk failed. Error count: " << dkerrorIndices.size() << endl;
    }

    vector<uint64_t> errorIndices = Catlass::golden::CompareData(dvHost, goldenDvHost, dvSize);
    if (errorIndices.empty()) {
        cout << "Compare dv success." << endl;
    } else {
        cerr << "Compare dv failed. Error count: " << errorIndices.size() << endl;
    }

    // Free host memory allocations.
    FreeMem(qHost, qDevice);
    FreeMem(kHost, kDevice);
    FreeMem(vHost, vDevice);
    FreeMem(dyHost, dyDevice);
    FreeMem(attenMaskHost, attenMaskDevice);
    FreeMem(softMaxMaxHost, softMaxMaxDevice);
    FreeMem(softMaxSumHost, softMaxSumDevice);
    FreeMem(attentionInHost, attentionInDevice);
    FreeMem(actualSeqQlenHost, actualSeqQlenDevice);
    FreeMem(actualSeqKvlenHost, actualSeqKvlenDevice);

    aclrtFree(dqDevice);
    aclrtFree(dkDevice);
    aclrtFree(dvDevice);
    aclrtFree(dq_rightDevice);
    aclrtFree(dk_rightDevice);
    aclrtFree(workspaceDevice);
    aclrtFreeHost(tilingHost);
    aclrtFree(tilingDevice);

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