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
#include "library_helper.h"

namespace Catlass {

using namespace Library;

TensorConfig OpConfig::GetTensorConfig(const std::string &key, const CommandLineParser &parser)
{
    TensorConfig config{};
    std::string_view val;
    auto err = parser.Get<std::string_view>(key, val);
    if (err != CommandLineParser::ERROR_CODE::NONE) {
        return config;
    }
    auto i = val.find(':');
    if (i == std::string::npos || i >= val.size() - 1) {
        LOGE("Parse command line inputs failed, value format should be sth like fp16:row, key: %s, value: %.*s",
            key.c_str(), static_cast<int>(val.size()), val.data());
        return config;
    }
    config.dataType = LibraryHelper::GetDataTypeEnum(val.substr(0, i));
    config.layoutType = LibraryHelper::GetLayoutEnum(val.substr(i + 1));
    if (config.dataType == DataType::Invalid || config.layoutType == LayoutType::Invalid) {
        LOGE("Parse command line inputs failed, value is invalid, key: %s, value: %.*s", key.c_str(),
             static_cast<int>(val.size()), val.data());
    }
    return config;
}

std::shared_ptr<OpConfig> GetGemmOpConfig(const OperationDescription &desp)
{
    if (desp.kind != OperationKind::Gemm) {
        LOGE("Operate is not matmul kind");
        return nullptr;
    }
    auto mDesp = static_cast<const GemmOperationDescription&>(desp);
    switch (mDesp.gemmKind) {
        case GemmKind::BasicMatmul:
            return std::make_shared<BasicGemmOpConfig>(desp);
        case GemmKind::GroupedMatmul:
            return std::make_shared<GroupedGemmOpConfig>(desp);
        default:
            LOGE("Matmul op type is invalid %u, config create failed", static_cast<uint32_t>(mDesp.gemmKind));
            break;
    }
    return nullptr;
}

std::shared_ptr<OpConfig> OpConfig::GetOpConfig(const OperationDescription &desp)
{
    using FuncType = std::shared_ptr<OpConfig>(*)(const OperationDescription &desp);
    std::vector<FuncType> func{
        GetGemmOpConfig
    };
    size_t i = static_cast<size_t>(desp.kind);
    if (i >= func.size()) {
        LOGE("description kind invalid %ld", i);
        return nullptr;
    }
    return func[i](desp);
}

void OpConfigPool::Register(Operation *op, const CommandLineParser &parser, const std::string_view kernel)
{
    auto &desp = op->GetDescription();
    std::string_view name = desp.name;
    if (!kernel.empty() && name.find(kernel) == std::string_view::npos) {
        return;
    }
    std::shared_ptr<OpConfig> config = OpConfig::GetOpConfig(desp);
    if (!config) {
        LOGE("Get op config failed, op name %s", desp.name);
        return;
    }
    auto p = pool_.insert({config, {}});
    if (p.second && !config->InitConfig(parser)) {
        LOGE("Initialize config failed, skip all same type op like current: %s", desp.name);
        return;
    }
    config = p.first->first;
    if (!config->Invalid() && config->Filter(op)) {
        p.first->second.emplace_back(op);
    }
}

} // namespace Catlass