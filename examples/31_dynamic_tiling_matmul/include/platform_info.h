#ifndef PLATFORM_INFO_H
#define PLATFORM_INFO_H

#include "tiling/platform/platform_ascendc.h"

struct PlatformInfo
{
    uint32_t coreNum{24};
    uint32_t ubSize{192 * 1024};
    uint32_t l1Size{512 * 1024};
    uint32_t l0ASize{64 * 1024};
    uint32_t l0BSize{64 * 1024};
    uint32_t l0CSize{128 * 1024};

    PlatformInfo()
    {
        coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::UB, ubSize);
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L1, l1Size);
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L0_A, l0ASize);
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L0_B, l0BSize);
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L0_C, l0CSize);
    }

    ~PlatformInfo() {}
};

#endif  // PLATFORM_INFO_H