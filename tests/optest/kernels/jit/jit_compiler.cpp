/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "jit_compiler.h"
#include "jit_config.h"

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

namespace CatlassKernel {

JitCompiler& JitCompiler::instance()
{
    static JitCompiler inst;
    return inst;
}

JitCompiler::~JitCompiler()
{
    clearCache();
}

void JitCompiler::lazyInit()
{
    std::call_once(initFlag_, [this] {
        std::error_code ec;

        const char* cacheDirEnv = std::getenv(JitConfig::kCacheDirEnv);
        std::string cacheDir = (cacheDirEnv && *cacheDirEnv) ? cacheDirEnv : "";
        if (cacheDir.empty()) {
            const char* home = std::getenv("HOME");
            if (home)
                cacheDir = std::string(home) + "/" + JitConfig::kHomeCacheSubdir;
        }
        if (cacheDir.empty()) {
            cacheDir = JitConfig::kDefaultCacheDir;
        }
        fs::create_directories(cacheDir, ec);
        JIT_CHECK(!ec, "mkdir failed: " + cacheDir + ": " + ec.message());
        cacheDir_ = std::move(cacheDir);

        bishengPath_ = FindCompilerPath();
        JIT_CHECK(!bishengPath_.empty(), "bisheng compiler not found (set ASCEND_HOME_PATH)");

        npuArch_ = GetCurrentNPUArch();

        templateBase_ = ResolveTemplateBase();

        JIT_LOG(
            JitLogLevel::Info, "JIT init: cache=%s compiler=%s arch=%s template=%s", cacheDir_.c_str(),
            bishengPath_.c_str(), npuArch_.c_str(), templateBase_.empty() ? "(none)" : templateBase_.c_str());
    });
}

JitEntryFn JitCompiler::getKernel(const char* templatePath, const MacroMap& macros)
{
    lazyInit();

    const auto nameIt = macros.find("CATLASS_KERNEL_NAME");
    JIT_CHECK(nameIt != macros.end() && !nameIt->second.empty(), "CATLASS_KERNEL_NAME not set in macros");
    const std::string& targetName = nameIt->second;
    const std::string entrySymbol = "run";

    JIT_CHECK(templatePath && templatePath[0], "templatePath is empty");

    auto sanitize = [](const std::string& s) -> std::string {
        std::string out;
        out.reserve(s.size());
        for (unsigned char c : s) {
            if (std::isalnum(c) || c == '_' || c == '-' || c == '.')
                out += static_cast<char>(c);
            else
                out += '_';
        }
        return (out.empty() || out == "." || out == "..") ? "unnamed" : out;
    };

    auto makeCacheKey = [&](const MacroMap& macroValues) -> std::string {
        std::vector<std::pair<std::string, std::string>> sortedMacros;
        sortedMacros.reserve(macroValues.size());
        for (const auto& kv : macroValues) {
            JIT_CHECK(!kv.first.empty(), "empty macro name");
            sortedMacros.emplace_back(kv.first, kv.second);
        }
        std::sort(sortedMacros.begin(), sortedMacros.end());

        std::string key = targetName + "_arch" + npuArch_;
        for (const auto& kv : sortedMacros) {
            key += "_" + kv.first + "_" + kv.second;
        }
        return sanitize(key);
    };

    const std::string cacheKey = makeCacheKey(macros);
    const std::string soPath = cacheDir_ + "/" + sanitize(cacheKey) + ".so";

    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = loaded_.find(cacheKey);
        if (it != loaded_.end()) {
            JIT_LOG(JitLogLevel::Debug, "mem hit: %s %s", cacheKey.c_str(), targetName.c_str());
            return it->second.entry;
        }
    }

    {
        std::error_code ec;
        if (!fs::is_regular_file(soPath, ec)) {
            std::lock_guard<std::mutex> lk(mutex_);

            {
                auto it = loaded_.find(cacheKey);
                if (it != loaded_.end())
                    return it->second.entry;
            }

            if (!fs::is_regular_file(soPath, ec)) {
                JIT_LOG(JitLogLevel::Info, "compiling: %s \xe2\x86\x92 %s", targetName.c_str(), soPath.c_str());

                fs::create_directories(fs::path(soPath).parent_path(), ec);
                JIT_CHECK(!ec, "mkdir failed: " + std::string(fs::path(soPath).parent_path()) + ": " + ec.message());
                compile(targetName, templatePath, macros, soPath);
            }
        }
    }

    SharedLib lib(soPath);
    auto* entry = reinterpret_cast<JitEntryFn>(lib.sym(entrySymbol));

    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = loaded_.find(cacheKey);
        if (it != loaded_.end()) {
            return it->second.entry;
        }
        loaded_.emplace(cacheKey, LoadedKernel{std::move(lib), entry});
    }

    return entry;
}

void JitCompiler::compile(
    std::string_view name, std::string_view templatePath, const MacroMap& macros, const std::string& soPath)
{
    auto args = buildCompilerArgs(name, templatePath, macros, soPath);

    auto cmdJoin = [](const std::vector<std::string>& args) -> std::string {
        std::string cmd;
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0)
                cmd += ' ';
            cmd += args[i];
        }
        return cmd;
    };

    const std::string cmdStr = cmdJoin(args);
    JIT_LOG(JitLogLevel::Debug, "compile: %s", cmdStr.c_str());

    const ProcessResult result = RunProcessCapture(args);
    if (result.exitCode != 0) {
        (void)::unlink(soPath.c_str());
        JIT_LOGE("compile failed (exit=%d): %s", result.exitCode, result.output.c_str());
        JIT_THROW(
            "compile failed (exit=" + std::to_string(result.exitCode) + ")\n" + "command: " + cmdStr + "\n" +
            "output:\n" + result.output);
    }

    {
        std::error_code ec;
        JIT_CHECK(fs::is_regular_file(soPath, ec), "compiler succeeded but output not created: " + soPath);
    }
}

std::vector<std::string> JitCompiler::buildCompilerArgs(
    std::string_view name, std::string_view templatePath, const MacroMap& macros, const std::string& soPath)
{
    std::vector<std::string> args;
    args.reserve(32);

    args.push_back(bishengPath_);

    args.push_back("-x");
    args.push_back("asc");

    {
        auto base = JitConfig::BaseFlags();
        args.insert(args.end(), base.begin(), base.end());
    }

    {
        auto archFlags = JitConfig::ArchFlags(npuArch_);
        args.insert(args.end(), archFlags.begin(), archFlags.end());
    }

    {
        const char* ms = std::getenv(JitConfig::kSanitizeEnv);
        if (ms && std::string(ms) == "1") {
            args.push_back("-g");
            args.push_back("--cce-enable-sanitizer");
            JIT_LOG(JitLogLevel::Info, "msSanitizer ENABLED (MS_SANITIZE_MEMORY=1)");
        }
    }

    {
        const char* ver = std::getenv(JitConfig::kVersionEnv);
        args.push_back(JitConfig::VersionDefine(ver ? ver : "unknown"));
    }

    auto sortedMacros = [](const MacroMap& macros) {
        std::vector<std::pair<std::string, std::string>> result;
        result.reserve(macros.size());
        for (const auto& kv : macros) {
            JIT_CHECK(!kv.first.empty(), "empty macro name");
            result.emplace_back(kv.first, kv.second);
        }
        std::sort(result.begin(), result.end());
        return result;
    };

    for (const auto& kv : sortedMacros(macros)) {
        args.push_back("-D" + kv.first + "=" + kv.second);
    }

    {
        auto it = macros.find("CATLASS_JIT_KERNEL_NAME");
        std::string kn = (it != macros.end() && !it->second.empty()) ? it->second : std::string(name);
        kn += "_arch" + npuArch_;
        args.push_back("-DCATLASS_JIT_KERNEL_NAME=" + kn);
    }

    auto extraIncludes = BuildIncludeArgsFromEnv();
    args.insert(args.end(), extraIncludes.begin(), extraIncludes.end());

    args.push_back(templateBase_.empty() ? std::string(templatePath) : templateBase_ + std::string(templatePath));
    args.push_back("-o");
    args.push_back(soPath);

    return args;
}

void JitCompiler::clearCache()
{
    std::lock_guard<std::mutex> lk(mutex_);
    loaded_.clear();
}

} // namespace CatlassKernel
