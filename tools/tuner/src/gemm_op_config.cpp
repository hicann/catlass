/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "gemm_op_config.h"
#include "device_memory_manager.h"
#include "metrics.h"
#include "library_helper.h"
#include "catlass/gemm_coord.hpp"

namespace Catlass {

void GemmOpConfig::SaveMetric(Metric &metric)
{
    metric.SetField<ClassicMetric::M>(m_);
    metric.SetField<ClassicMetric::N>(n_);
    metric.SetField<ClassicMetric::K>(k_);
}

bool GemmOpConfig::InitConfig(const Catlass::CommandLineParser &parser)
{
    static const std::vector<std::string> keys = {"m", "n", "k"};
    for (auto &key : keys) {
        if (!parser.HasKey(key)) {
            LOGE("Key %s not exist", key.c_str());
            invalid_ = true;
            return false;
        }
    }
    GET_CHECK(parser.Get<decltype(m_)>("m", m_), "m");
    GET_CHECK(parser.Get<decltype(n_)>("n", n_), "n");
    GET_CHECK(parser.Get<decltype(k_)>("k", k_), "k");
    if (m_ == 0 || n_ == 0 || k_ == 0) {
        invalid_ = true;
        return false;
    }
    tcA_ = GetTensorConfig("A", parser);
    tcB_ = GetTensorConfig("B", parser);
    tcC_ = GetTensorConfig("C", parser);
    return true;
}

bool GemmOpConfig::Filter(Library::Operation *op)
{
    auto &mdesp = static_cast<const Library::GemmOperationDescription&>(op->GetDescription());
    if (UnMatch(tcA_.dataType, mdesp.A.element) || UnMatch(tcA_.layoutType, mdesp.A.layout) ||
        UnMatch(tcB_.dataType, mdesp.B.element) || UnMatch(tcB_.layoutType, mdesp.B.layout) ||
        UnMatch(tcC_.dataType, mdesp.C.element) || UnMatch(tcC_.layoutType, mdesp.C.layout)) {
        return false;
    }
    return true;
}

bool BasicGemmOpConfig::InitConfig(const CommandLineParser &parser)
{
    bool res = GemmOpConfig::InitConfig(parser);
    if (!res) {
        return false;
    }
    config_.m = m_;
    config_.n = n_;
    config_.k = k_;
    return true;
}

bool BasicGemmOpConfig::InitArgument(Library::Operation *op)
{
    auto &mdesp = static_cast<const Library::GemmOperationDescription &>(op->GetDescription());
    size_t lenA;
    size_t lenB;
    size_t lenC;
    constexpr std::string_view log = "Arguments size overflows, please check command line input"
                                     " --m --n --k";
    if (!SafeMul<uint32_t>({config_.m, config_.k}, lenA) ||
        !SafeMul<uint32_t>({config_.k, config_.n}, lenB) ||
        !SafeMul<uint32_t>({config_.m, config_.n}, lenC)) {
        LOGE("%s", log.data());
        return false;
    }

    size_t sizeA;
    size_t sizeB;
    size_t sizeC;
    if (!SafeMul<size_t>({lenA, LibraryHelper::GetDataTypeSize(mdesp.A.element)}, sizeA) ||
        !SafeMul<size_t>({lenB, LibraryHelper::GetDataTypeSize(mdesp.B.element)}, sizeB) ||
        !SafeMul<size_t>({lenC, LibraryHelper::GetDataTypeSize(mdesp.C.element)}, sizeC)) {
        LOGE("%s", log.data());
        return false;
    }
    std::vector<DeviceMemoryParam> params{
        {reinterpret_cast<void**>(&arg_.A), sizeA},
        {reinterpret_cast<void**>(&arg_.B), sizeB},
        {reinterpret_cast<void**>(&arg_.C), sizeC},
    };
    if (!MallocDeviceMemory(params)) {
        return false;
    }
    return true;
}

bool OptimizedGemmOpConfig::InitConfig(const CommandLineParser &parser)
{
    bool res = GemmOpConfig::InitConfig(parser);
    if (!res) {
        return false;
    }
    config_.m = m_;
    config_.n = n_;
    config_.k = k_;
    return true;
}

bool OptimizedGemmOpConfig::InitArgument(Library::Operation *op)
{
    auto &mdesp = static_cast<const Library::GemmOperationDescription &>(op->GetDescription());
    size_t lenA;
    size_t lenB;
    size_t lenC;
    constexpr std::string_view log = "Arguments size overflows, please check command line input"
                                     " --m --n --k";
    if (!SafeMul<uint32_t>({config_.m, config_.k}, lenA) ||
        !SafeMul<uint32_t>({config_.k, config_.n}, lenB) ||
        !SafeMul<uint32_t>({config_.m, config_.n}, lenC)) {
        LOGE("%s", log.data());
        return false;
    }

    size_t sizeA;
    size_t sizeB;
    size_t sizeC;
    if (!SafeMul<size_t>({lenA, LibraryHelper::GetDataTypeSize(mdesp.A.element)}, sizeA) ||
        !SafeMul<size_t>({lenB, LibraryHelper::GetDataTypeSize(mdesp.B.element)}, sizeB) ||
        !SafeMul<size_t>({lenC, LibraryHelper::GetDataTypeSize(mdesp.C.element)}, sizeC)) {
        LOGE("%s", log.data());
        return false;
    }
    std::vector<DeviceMemoryParam> params{
        {reinterpret_cast<void**>(&arg_.A), sizeA},
        {reinterpret_cast<void**>(&arg_.B), sizeB},
        {reinterpret_cast<void**>(&arg_.C), sizeC},
    };
    if (!MallocDeviceMemory(params)) {
        return false;
    }
    return true;
}

void GroupedGemmOpConfig::SaveMetric(Metric &metric)
{
    GemmOpConfig::SaveMetric(metric);
    metric.SetField("group_count", std::to_string(config_.groupCount));
}

bool GroupedGemmOpConfig::InitConfig(const CommandLineParser &parser)
{
    bool res = GemmOpConfig::InitConfig(parser);
    if (!res) {
        return false;
    }
    config_.m = m_;
    config_.n = n_;
    config_.k = k_;
    if (!parser.HasKey("group_count")) {
        config_.groupCount = 1;
    } else {
        GET_CHECK(parser.Get<decltype(config_.groupCount)>("group_count", config_.groupCount), "group_count");
        if (config_.groupCount == 0) {
            LOGE("The --group_count should be a positive integer");
            invalid_ = true;
            return false;
        }
    }
    GenerateGroupList();
    return true;
}

void GroupedGemmOpConfig::GenerateGroupList()
{
    groupList_.resize(config_.groupCount);
    std::srand(std::time(nullptr));
    for (uint32_t i = 0; i < config_.groupCount; ++i) {
        groupList_[i] = rand() % (config_.m + 1);
    }
    std::sort(groupList_.begin(), groupList_.end());
}

bool GroupedGemmOpConfig::CheckArgument(const Library::GemmOperationDescription &mdesp, ArgumentSize &argSize)
{
    argSize.layoutASize = LibraryHelper::GetLayoutSize(mdesp.A.layout);
    argSize.layoutBSize = LibraryHelper::GetLayoutSize(mdesp.B.layout);
    argSize.layoutCSize = LibraryHelper::GetLayoutSize(mdesp.C.layout);
    if (!SafeMul<uint32_t>({config_.m, config_.k}, argSize.lenA) ||
        !SafeMul<uint32_t>({config_.k, config_.n}, argSize.lenB) ||
        !SafeMul<uint32_t>({config_.m, config_.n, config_.groupCount}, argSize.lenC) ||
        !SafeMul<size_t>({argSize.lenA, LibraryHelper::GetDataTypeSize(mdesp.A.element)}, argSize.sizeA) ||
        !SafeMul<size_t>({argSize.lenB, LibraryHelper::GetDataTypeSize(mdesp.B.element)}, argSize.sizeB) ||
        !SafeMul<size_t>({argSize.lenC, LibraryHelper::GetDataTypeSize(mdesp.C.element)}, argSize.sizeC) ||
        !SafeMul<size_t>({config_.groupCount, argSize.layoutASize}, argSize.sizeLayoutAList) ||
        !SafeMul<size_t>({config_.groupCount, argSize.layoutBSize}, argSize.sizeLayoutBList) ||
        !SafeMul<size_t>({config_.groupCount, argSize.layoutCSize}, argSize.sizeLayoutCList) ||
        !SafeMul<size_t>({config_.groupCount, sizeof(GemmCoord)}, argSize.sizeProblemShapeList)) {
        LOGE("Arguments size overflows, please check command line input --m --n --k --group_count");
        return false;
    }
    return true;
}

void GroupedGemmOpConfig::GenerateInput(const Library::GemmOperationDescription &mdesp,
                                        const ArgumentSize &argSize)
{
    std::vector<GemmCoord> problemShapeList(config_.groupCount);
    std::vector<uint8_t> layoutAList(argSize.layoutASize * config_.groupCount);
    std::vector<uint8_t> layoutBList(argSize.layoutBSize * config_.groupCount);
    std::vector<uint8_t> layoutCList(argSize.layoutCSize * config_.groupCount);
    for (uint32_t i = 0, a = 0, b = 0, c = 0;
         i < config_.groupCount;
         ++i, a += argSize.layoutASize, b += argSize.layoutBSize, c += argSize.layoutCSize) {
        uint32_t currentK = (i == 0) ? groupList_[0] : (groupList_[i] - groupList_[i - 1]);
        problemShapeList[i] = GemmCoord{config_.m, config_.n, currentK};
        LibraryHelper::ConstructLayout(mdesp.A.layout, mdesp.A.element, config_.m, currentK, &layoutAList[a]);
        LibraryHelper::ConstructLayout(mdesp.B.layout, mdesp.B.element, currentK, config_.n, &layoutBList[b]);
        LibraryHelper::ConstructLayout(mdesp.C.layout, mdesp.C.element, config_.m, config_.n, &layoutCList[c]);
    }
    DeviceMemoryManager::Instance().FillDeviceData(arg_.problemShapeList, argSize.sizeProblemShapeList,
                                                   problemShapeList.data());
    DeviceMemoryManager::Instance().FillDeviceData(arg_.layoutAList, argSize.sizeLayoutAList, layoutAList.data());
    DeviceMemoryManager::Instance().FillDeviceData(arg_.layoutBList, argSize.sizeLayoutBList, layoutBList.data());
    DeviceMemoryManager::Instance().FillDeviceData(arg_.layoutCList, argSize.sizeLayoutCList, layoutCList.data());
}

bool GroupedGemmOpConfig::InitArgument(Library::Operation *op)
{
    auto &mdesp = static_cast<const Library::GemmOperationDescription &>(op->GetDescription());
    ArgumentSize safeArg{};
    if (!CheckArgument(mdesp, safeArg)) {
        return false;
    }

    std::vector<DeviceMemoryParam> params{
        {reinterpret_cast<void**>(&arg_.problemShapeList), safeArg.sizeProblemShapeList},
        {reinterpret_cast<void**>(&arg_.A), safeArg.sizeA},
        {reinterpret_cast<void**>(&arg_.layoutAList), safeArg.sizeLayoutAList},
        {reinterpret_cast<void**>(&arg_.B), safeArg.sizeB},
        {reinterpret_cast<void**>(&arg_.layoutBList), safeArg.sizeLayoutBList},
        {reinterpret_cast<void**>(&arg_.C), safeArg.sizeC},
        {reinterpret_cast<void**>(&arg_.layoutCList), safeArg.sizeLayoutCList},
    };
    if (!MallocDeviceMemory(params)) {
        return false;
    }

    GenerateInput(mdesp, safeArg);
    return true;
}

bool GroupedSliceMGemmOpConfig::InitConfig(const CommandLineParser &parser)
{
    bool res = GemmOpConfig::InitConfig(parser);
    if (!res) {
        return false;
    }
    config_.m = m_;
    config_.n = n_;
    config_.k = k_;
    if (!parser.HasKey("group_count")) {
        config_.groupCount = 1;
    } else {
        GET_CHECK(parser.Get<decltype(config_.groupCount)>("group_count", config_.groupCount), "group_count");
        if (config_.groupCount == 0) {
            LOGE("The --group_count should be a positive integer");
            invalid_ = true;
            return false;
        }
    }
    return true;
}

bool GroupedSliceMGemmOpConfig::InitArgument(Library::Operation *op)
{
    auto &mdesp = static_cast<const Library::GemmOperationDescription &>(op->GetDescription());
    ArgumentSize argSize{};
    if (!SafeMul<uint32_t>({config_.m, config_.k}, argSize.lenA) ||
        !SafeMul<uint32_t>({config_.k, config_.n, config_.groupCount}, argSize.lenB) ||
        !SafeMul<uint32_t>({config_.m, config_.n}, argSize.lenC) ||
        !SafeMul<size_t>({argSize.lenA, LibraryHelper::GetDataTypeSize(mdesp.A.element)}, argSize.sizeA) ||
        !SafeMul<size_t>({argSize.lenB, LibraryHelper::GetDataTypeSize(mdesp.B.element)}, argSize.sizeB) ||
        !SafeMul<size_t>({argSize.lenC, LibraryHelper::GetDataTypeSize(mdesp.C.element)}, argSize.sizeC) ||
        !SafeMul<size_t>({config_.groupCount, sizeof(GemmCoord)}, argSize.sizeGroupList)) {
        LOGE("Arguments size overflows, please check command line input --m --n --k --group_count");
        return false;
    }

    std::vector<DeviceMemoryParam> params{
        {reinterpret_cast<void**>(&arg_.deviceGroupList), argSize.sizeGroupList},
        {reinterpret_cast<void**>(&arg_.A), argSize.sizeA},
        {reinterpret_cast<void**>(&arg_.B), argSize.sizeB},
        {reinterpret_cast<void**>(&arg_.C), argSize.sizeC},
    };
    if (!MallocDeviceMemory(params)) {
        return false;
    }

    std::vector<int64_t> groupList;
    groupList.resize(config_.groupCount);
    std::srand(std::time(nullptr));
    for (uint32_t i = 0; i < config_.groupCount; ++i) {
        groupList[i] = rand() % (config_.m + 1);
    }
    std::sort(groupList.begin(), groupList.end());
    DeviceMemoryManager::Instance().FillDeviceData(arg_.deviceGroupList, argSize.sizeGroupList,
                                                   groupList.data());
    return true;
}

void GroupedSliceMGemmOpConfig::SaveMetric(Metric &metric)
{
    GemmOpConfig::SaveMetric(metric);
    metric.SetField("group_count", std::to_string(config_.groupCount));
}

} // namespace Catlass