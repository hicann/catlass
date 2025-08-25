/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_COMMAND_LINE_PARSER_H
#define CATLASS_TUNER_COMMAND_LINE_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdint>

namespace Catlass {

#define GET_CHECK(stat, key)                                                        \
    do {                                                                            \
        CommandLineParser::ERROR_CODE err = stat;                                   \
        if (err != CommandLineParser::ERROR_CODE::NONE) {                           \
            LOGE("%s:%d get key --" key " failed, err: %s", __FUNCTION__, __LINE__, \
                CommandLineParser::GetErrorStr(err).data());                        \
        }                                                                           \
    } while (false)

class CommandLineParser {
public:
    enum class ERROR_CODE : uint32_t {
        NONE = 0,
        CAST_FAILED,
        KEY_NOT_EXIST,
        INTEGER_OVERFLOW,
        EXPECT_UNSIGNED_INTEGER,
        END,
    };

    static std::string_view GetErrorStr(ERROR_CODE er)
    {
        static const std::string_view strs[] = {
            "none",
            "cast value from string failed",
            "key not exist",
            "integer overflow",
            "expect unsigned integer",
            "unknown error"
        };
        auto idx = std::min(static_cast<int>(er), static_cast<int>(ERROR_CODE::END));
        return strs[idx];
    }

    [[nodiscard]] inline bool HasKey(const std::string& key) const
    {
        return dataMap_.find(key) != dataMap_.end();
    }

    [[nodiscard]] inline bool Help() const { return help_; }

    void Parse(int argc, const char* argv[]);
    void PrintHelp();
    [[nodiscard]] std::vector<std::string> Keys() const;
    template <typename T>
    ERROR_CODE Get(const std::string& key, T &target) const;

private:
    // 去除字符串两端空格
    static std::string Trim(const std::string& str)
    {
        auto start = std::find_if_not(str.begin(), str.end(), [](unsigned char c) {
            return std::isspace(c);
        });
        auto end = std::find_if_not(str.rbegin(), str.rend(), [](unsigned char c) {
            return std::isspace(c);
        }).base();
        return (start < end) ? std::string(start, end) : "";
    }

    std::map<std::string, std::string> dataMap_;
    bool help_{false};
};

#define COMMAND_LINE_TEMPLATE(type) \
template<> \
CommandLineParser::ERROR_CODE CommandLineParser::Get<type>(const std::string& key, type &target) const

COMMAND_LINE_TEMPLATE(std::string);
COMMAND_LINE_TEMPLATE(std::string_view);
COMMAND_LINE_TEMPLATE(int64_t);
COMMAND_LINE_TEMPLATE(uint64_t);
COMMAND_LINE_TEMPLATE(int32_t);
COMMAND_LINE_TEMPLATE(uint32_t);
COMMAND_LINE_TEMPLATE(double);
COMMAND_LINE_TEMPLATE(float);
COMMAND_LINE_TEMPLATE(bool);

#undef COMMAND_LINE_TEMPLATE

} // namespace Catlass
#endif // CATLASS_TUNER_COMMAND_LINE_PARSER_H