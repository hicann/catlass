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
        "Usage: FAG nheads nheads_k headdim dtype element_of_list_seq [--device DEVICE_ID]\n";
    static constexpr auto MIN_ARGS = 5;

    // Define default value.
    uint32_t nheads{0}, nheads_k{0}, headdim{0};
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
        nheads = atoi(argv[argIndex++]);
        nheads_k = atoi(argv[argIndex++]);
        headdim = atoi(argv[argIndex++]);
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
        ss << "{ nheads: " << nheads << ", nheads_k: " << nheads_k << ", headdim: " << headdim << ", dtype: " << dtype <<
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
// ACTLASS FAG kernel.
void Run(const Options &options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint32_t nheads = options.nheads;
    uint32_t nheads_k = options.nheads_k;
    uint32_t headdim = options.headdim;
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

    printf("Running fag: nheads=%d, nheads_k=%d, headdim=%d, ...\n", \
            nheads, nheads_k, headdim);
    printf("list_seq = ");
    for (const auto& num : list_seq) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    FAGTiling::FAGInfo fagInfo;
    
    int64_t sum_of_list = 0;
    for (size_t i = 0; i < list_seq.size(); ++i) {
        sum_of_list += list_seq[i];
    }

    fagInfo.seqQShapeSize = list_seq.size();
    fagInfo.queryShape_0 = sum_of_list;
    fagInfo.keyShape_0 = sum_of_list;
    fagInfo.queryShape_1 = nheads;
    fagInfo.keyShape_1 = nheads_k;
    fagInfo.queryShape_2 = headdim;
    fagInfo.scaleValue = 1.0 / sqrt(headdim);

    uint64_t g = nheads / nheads_k;
    uint64_t qSize = sum_of_list * nheads_k * g * headdim * sizeof(fp16_t);
    uint64_t kSize = sum_of_list * nheads_k * 1 * headdim * sizeof(fp16_t);
    uint64_t vSize = sum_of_list * nheads_k * 1 * headdim * sizeof(fp16_t);
    uint64_t dOutSize = sum_of_list * nheads * headdim * sizeof(fp16_t);
    uint64_t attenMaskSize = 2048 * 2048 * sizeof(fp16_t);
    uint64_t softMaxMaxSize = sum_of_list * nheads * 8 * sizeof(float);
    uint64_t softMaxSumSize = sum_of_list * nheads * 8 * sizeof(float);
    uint64_t outSize = sum_of_list * nheads * headdim * sizeof(fp16_t);
    uint64_t cuSeqQlenSize = list_seq.size() * sizeof(int64_t);
    uint64_t cuSeqKvlenSize = list_seq.size() * sizeof(int64_t);

    uint64_t dqSize = qSize;
    uint64_t dkSize = kSize;
    uint64_t dvSize = vSize;
    uint64_t dqRightSize = 16 * 128 * 256 * sizeof(fp16_t);
    uint64_t dkRightSize = 16 * 128 * 128 * sizeof(float);

    uint64_t workspaceSize = (2 * aicCoreNum * 16 * 128 * 128 * 8 * nheads) * sizeof(float);

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

    uint8_t *dOutHost;
    uint8_t *dOutDevice;
    AllocMem(&dOutHost, &dOutDevice, dOutSize);
    ReadFile(dataPath + "/dout.bin", dOutHost, dOutSize);
    ACL_CHECK(aclrtMemcpy(dOutDevice, dOutSize, dOutHost, dOutSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *attenMaskHost;
    uint8_t *attenMaskDevice;
    AllocMem(&attenMaskHost, &attenMaskDevice, attenMaskSize);
    ReadFile(dataPath + "/atten_mask.bin", attenMaskHost, attenMaskSize);
    ACL_CHECK(aclrtMemcpy(attenMaskDevice, attenMaskSize, attenMaskHost, attenMaskSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *softMaxMaxHost;
    uint8_t *softMaxMaxDevice;
    AllocMem(&softMaxMaxHost, &softMaxMaxDevice, softMaxMaxSize);
    ReadFile(dataPath + "/row_max.bin", softMaxMaxHost, softMaxMaxSize);
    ACL_CHECK(aclrtMemcpy(softMaxMaxDevice, softMaxMaxSize, softMaxMaxHost, softMaxMaxSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *softMaxSumHost;
    uint8_t *softMaxSumDevice;
    AllocMem(&softMaxSumHost, &softMaxSumDevice, softMaxSumSize);
    ReadFile(dataPath + "/row_sum.bin", softMaxSumHost, softMaxSumSize);
    ACL_CHECK(aclrtMemcpy(softMaxSumDevice, softMaxSumSize, softMaxSumHost, softMaxSumSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *outHost;
    uint8_t *outDevice;
    AllocMem(&outHost, &outDevice, outSize);
    ReadFile(dataPath + "/out.bin", outHost, outSize);
    ACL_CHECK(aclrtMemcpy(outDevice, outSize, outHost, outSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cuSeqQlenHost;
    uint8_t *cuSeqQlenDevice;
    AllocMem(&cuSeqQlenHost, &cuSeqQlenDevice, cuSeqQlenSize);
    ReadFile(dataPath + "/cu_seq_qlen.bin", cuSeqQlenHost, cuSeqQlenSize);
    ACL_CHECK(aclrtMemcpy(cuSeqQlenDevice, cuSeqQlenSize, cuSeqQlenHost, cuSeqQlenSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cuSeqKvlenHost;
    uint8_t *cuSeqKvlenDevice;
    AllocMem(&cuSeqKvlenHost, &cuSeqKvlenDevice, cuSeqKvlenSize);
    ReadFile(dataPath + "/cu_seq_kvlen.bin", cuSeqKvlenHost, cuSeqKvlenSize);
    ACL_CHECK(aclrtMemcpy(cuSeqKvlenDevice, cuSeqKvlenSize, cuSeqKvlenHost, cuSeqKvlenSize, ACL_MEMCPY_HOST_TO_DEVICE));

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
    FAGTiling::GetFATilingParam(fagInfo, blockDim, (int64_t *)tilingHost);
    uint8_t *tilingDevice;
    ACL_CHECK(aclrtMalloc((void **)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    FAG<half><<<blockDim, nullptr, stream>>>(
        fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
        attenMaskDevice, softMaxMaxDevice, softMaxSumDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice, cuSeqKvlenDevice,
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
    FreeMem(dOutHost, dOutDevice);
    FreeMem(attenMaskHost, attenMaskDevice);
    FreeMem(softMaxMaxHost, softMaxMaxDevice);
    FreeMem(softMaxSumHost, softMaxSumDevice);
    FreeMem(outHost, outDevice);
    FreeMem(cuSeqQlenHost, cuSeqQlenDevice);
    FreeMem(cuSeqKvlenHost, cuSeqKvlenDevice);

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