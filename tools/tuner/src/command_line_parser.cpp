/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "command_line_parser.h"
#include "log.h"

namespace Catlass {

using ERROR_CODE = CommandLineParser::ERROR_CODE;

template<>
ERROR_CODE CommandLineParser::Get<std::string>(const std::string& key, std::string &target) const
{
    auto it = dataMap_.find(key);
    if (it == dataMap_.end()) {
        return ERROR_CODE::KEY_NOT_EXIST;
    }
    target = it->second;
    return ERROR_CODE::NONE;
}

template<>
ERROR_CODE CommandLineParser::Get<std::string_view>(const std::string& key, std::string_view &target) const
{
    auto it = dataMap_.find(key);
    if (it == dataMap_.end()) {
        return ERROR_CODE::KEY_NOT_EXIST;
    }
    target = it->second;
    return ERROR_CODE::NONE;
}

template<>
ERROR_CODE CommandLineParser::Get<int64_t>(const std::string& key, int64_t &target) const
{
    auto it = dataMap_.find(key);
    if (it == dataMap_.end()) {
        return ERROR_CODE::KEY_NOT_EXIST;
    }

    try {
        target = std::stoll(it->second);
    } catch (...) {
        return ERROR_CODE::CAST_FAILED;
    }
    return ERROR_CODE::NONE;
}

template<>
ERROR_CODE CommandLineParser::Get<uint64_t>(const std::string& key, uint64_t &target) const
{
    auto it = dataMap_.find(key);
    if (it == dataMap_.end()) {
        return ERROR_CODE::KEY_NOT_EXIST;
    }

    if (it->second[0] == '-') {
        return ERROR_CODE::EXPECT_UNSIGNED_INTEGER;
    }
    try {
        target = std::stoull(it->second);
    } catch (...) {
        return ERROR_CODE::CAST_FAILED;
    }
    return ERROR_CODE::NONE;
}

template<>
ERROR_CODE CommandLineParser::Get<int32_t>(const std::string& key, int32_t &target) const
{
    int64_t x = 0;
    ERROR_CODE ret = CommandLineParser::Get<int64_t>(key, x);
    if (ret != ERROR_CODE::NONE) {
        return ret;
    }
    if (x > INT32_MAX || x < INT32_MIN) {
        return ERROR_CODE::INTEGER_OVERFLOW;
    }
    target = static_cast<int32_t>(x);
    return ERROR_CODE::NONE;
}

template<>
ERROR_CODE CommandLineParser::Get<uint32_t>(const std::string& key, uint32_t &target) const
{
    uint64_t x = 0;
    ERROR_CODE ret = CommandLineParser::Get<uint64_t>(key, x);
    if (ret != ERROR_CODE::NONE) {
        return ret;
    }
    if (x > UINT32_MAX) {
        return ERROR_CODE::INTEGER_OVERFLOW;
    }
    target = static_cast<uint32_t>(x);
    return ERROR_CODE::NONE;
}

template<>
ERROR_CODE CommandLineParser::Get<double>(const std::string& key, double &target) const
{
    auto it = dataMap_.find(key);
    if (it == dataMap_.end()) {
        return ERROR_CODE::KEY_NOT_EXIST;
    }

    try {
        target = std::stod(it->second);
    } catch (...) {
        return ERROR_CODE::CAST_FAILED;
    }
    return ERROR_CODE::NONE;
}

template<>
ERROR_CODE CommandLineParser::Get<float>(const std::string& key, float &target) const
{
    auto it = dataMap_.find(key);
    if (it == dataMap_.end()) {
        return ERROR_CODE::KEY_NOT_EXIST;
    }

    try {
        target = std::stof(it->second);
    } catch (...) {
        return ERROR_CODE::CAST_FAILED;
    }
    return ERROR_CODE::NONE;
}

template<>
ERROR_CODE CommandLineParser::Get<bool>(const std::string& key, bool &target) const
{
    auto it = dataMap_.find(key);
    if (it == dataMap_.end()) {
        return ERROR_CODE::KEY_NOT_EXIST;
    }

    std::string val = it->second;
    std::transform(val.begin(), val.end(), val.begin(), ::tolower);
    target = (val == "true" || val == "1" || val == "yes" || val == "on");
    return ERROR_CODE::NONE;
}

void CommandLineParser::Parse(int argc, const char **argv)
{
    constexpr size_t PREFIX_LEN = 2;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        // 检查参数格式是否为 --key=value
        if (arg.substr(0, PREFIX_LEN) == "--" && arg.find('=') != std::string::npos) {
            // 查找等号位置
            size_t pos = arg.find('=');
            std::string key = Trim(arg.substr(PREFIX_LEN, pos - PREFIX_LEN));
            std::string value = Trim(arg.substr(pos + 1));

            if (!key.empty()) {
                dataMap_[key] = value;
            }
        } else if (arg == "--help" || arg == "-h") {
            help_ = true;
        }
    }
}

std::vector<std::string> CommandLineParser::Keys() const
{
    std::vector<std::string> result;
    result.reserve(dataMap_.size());
    for (const auto& pair : dataMap_) {
        result.push_back(pair.first);
    }
    return result;
}

void CommandLineParser::PrintHelp()
{
    LOGM("mstuner_catlass (MindStudio Tuner for CATLASS) is part of MindStudio Operator-dev Tools.\n");
    LOGM("The mstuner_catlass tool provides developers with the capability to efficiently and\n"
         "rapidly perform batch testing on the performance of catlass library operators.\n"
         "It supports developers in selecting optimal tiling parameters for custom problem scenarios.\n");
    LOGM("Options:");
    LOGM("   --help                               <Optional> Help message.");
    LOGM("   --output=<string>                    <Optional> Path to output file containing profiling data.");
    LOGM("   --device=<int>                       <Optional> Device id, a positive integer, default is 0.");
    LOGM("   --m=<int>                            <Required> Specify dimension m for matmul problem shape.");
    LOGM("   --n=<int>                            <Required> Specify dimension n for matmul problem shape.");
    LOGM("   --k=<int>                            <Required> Specify dimension k for matmul problem shape.");
    LOGM("   --kernels=<string>                   <Optional> Filter operations by kernel name.");
    LOGM("   --A=<dtype:layout>                   <Optional> Filter operations by dtype and layout of the tensor A.");
    LOGM("   --B=<dtype:layout>                   <Optional> Filter operations by dtype and layout of the tensor B.");
    LOGM("   --C=<dtype:layout>                   <Optional> Filter operations by dtype and layout of the tensor C.");
    LOGM("   --group_count=<int>                  <Optional> Specify group count for grouped-matmul-like operations.");
}
} // namespace Catlass