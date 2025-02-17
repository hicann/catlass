/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include "acl/acl.h"
#include "string"
#include <iomanip>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include "helper.hpp"
#include "golden.hpp"

#include "acot/acot.hpp"
#include "acot/arch/arch.hpp"
#include "acot/gemv/block/block_gemv.hpp"
#include "acot/gemv/block/block_swizzle.hpp"
#include "acot/gemv/dispatch_policy.hpp"
#include "acot/gemv/kernel/gemv_uinversal.hpp"
#include "acot/gemv/gemv_type.hpp"
#include "acot/layout/layout.hpp"

using namespace acot;

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

template <
    class Layoutx,
    class LayoutA,
    class Layouty>
ACOT_GLOBAL void FP32CMGEMV(
    GemvCoord problemShape,
    GM_ADDR gmx, Layoutx layoutx,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmy, Layouty layouty)
{
    constexpr bool enableUnitFlag = true;

    using ArchTag = arch::AtlasA2;
    using DispatchPolicy = gemv::GemvAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = GemvShape<256, 32>;
    using L0TileShape = GemvShape<256, 32>;

    using xType = gemv::GemvType<float, Layoutx>;
    using AType = gemv::GemvType<float, LayoutA>;
    using yType = gemv::GemvType<float, Layouty>;

    using BlockGemv = gemv::block::BlockGemv<DispatchPolicy, L1TileShape, L0TileShape, xType, AType, yType>;
    using BlockEpilogue = void;

    using TileScheduler = typename gemv::block::GemvIdentityBlockSwizzle<3, 0>; // 暂时未使用

    // kernle levels
    using GemvKernel = gemv::kernel::GemvUniversal<BlockGemv, BlockEpilogue, TileScheduler>;

    typename GemvKernel::Params params{problemShape, gmx, layoutx, gmA, layoutA, gmy, layouty};

    // call a kernel
    GemvKernel gemv;
    gemv(params);
}

struct Options
{
    const std::string HELPER = "04_gemv_03_fp32_cm_gemv m n [device_id]";

    GemvCoord problemShape{128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex
        {
            M_INDEX = 1,
            N_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= N_INDEX)
        {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        if (argc > M_INDEX)
        {
            problemShape.m() = std::stoi(argv[M_INDEX]);
        }
        if (argc > N_INDEX)
        {
            problemShape.n() = std::stoi(argv[N_INDEX]);
        }
        if (argc > DEVICE_ID_INDEX)
        {
            deviceId = std::stoi(argv[DEVICE_ID_INDEX]);
        }

        return 0;
    }
};

/**
 * @brief Read data from file
 * @param [in] filePath: file path
 * @param [out] fileSize: file size
 * @return read result
 */
bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1)
    {
        ERROR_LOG("Failed to get file.");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0)
    {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open())
    {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0)
    {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize)
    {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

/**
 * @brief Write data to file
 * @param [in] filePath: file path
 * @param [in] buffer: data to write to file
 * @param [in] size: size to write
 * @return write result
 */
bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr)
    {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0)
    {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void)close(fd);
    if (writeSize != size)
    {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

void Run(Options const &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();

    size_t lenx = static_cast<size_t>(n);
    size_t lenA = static_cast<size_t>(n) * m;
    size_t leny = static_cast<size_t>(m);

    size_t sizex = lenx * sizeof(float);
    size_t sizeA = lenA * sizeof(float);
    size_t sizey = leny * sizeof(float);

    std::cout << "问题向量大小为：" << n << std::endl;
    std::cout << "问题矩阵大小为：" << "m：" << m << " " << "n：" << n << std::endl;

    layout::RowMajor layoutx(1, n);
    // layout::ColumnMajor layoutA(n, m);
    layout::ColumnMajor layoutA(m, n);
    layout::RowMajor layouty(m, 1);

    float *hostx;
    float *hostA;

    ACL_CHECK(aclrtMallocHost((void **)(&hostx), sizex));
    ReadFile("./data/input/X.bin", sizex, hostx, sizex);

    ACL_CHECK(aclrtMallocHost((void **)(&hostA), sizeA));
    ReadFile("./data/input/A.bin", sizeA, hostA, sizeA);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *devicex{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devicex), sizex, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(devicex, sizex, hostx, sizex, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *devicey{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devicey), sizey, ACL_MEM_MALLOC_HUGE_FIRST));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    FP32CMGEMV<<<aicCoreNum, nullptr, stream>>>(
        options.problemShape,
        devicex,
        layoutx,
        deviceA,
        layoutA,
        devicey,
        layouty);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    float *hosty;
    ACL_CHECK(aclrtMallocHost((void **)(&hosty), sizey));
    ACL_CHECK(aclrtMemcpy(hosty, sizey, devicey, sizey, ACL_MEMCPY_DEVICE_TO_HOST));

    WriteFile("./data/output/our_res.bin", hosty, sizey);

    ACL_CHECK(aclrtFree(devicex));
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(devicey));
    ACL_CHECK(aclrtFreeHost(hostx));
    ACL_CHECK(aclrtFreeHost(hostA));
    ACL_CHECK(aclrtFreeHost(hosty));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0)
    {
        return -1;
    }

    Run(options);
    return 0;
}