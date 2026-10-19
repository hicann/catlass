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

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <thread>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include "jit_config.h"
#include "jit_sha256.h"

namespace fs = std::filesystem;

namespace CatlassKernel {

namespace {

void JitWarn(const std::string& message)
{
    std::fprintf(stderr, "[JIT][WARN] %s\n", message.c_str());
}

class CacheEntryLock {
public:
    explicit CacheEntryLock(const std::string& lockPath)
    {
        using Clock = std::chrono::steady_clock;
        constexpr auto kPollInterval = std::chrono::milliseconds(50);
        constexpr auto kLogInterval = std::chrono::seconds(30);
        // Longer than any expected compile. The env override exists for tests.
        const char* timeoutEnv = std::getenv("CATLASS_JIT_LOCK_TIMEOUT_SEC");
        const auto timeout = std::chrono::seconds(timeoutEnv ? std::atoi(timeoutEnv) : 30 * 60);

        // O_CLOEXEC keeps the compiler subprocess from inheriting the lock, which would hold it
        // past this object's lifetime. O_RDONLY lets a lock file created by another user work on
        // local filesystems; where an exclusive lock needs write access (NFS), flock() fails and
        // the compile continues without the lock.
        lockFd_ = ::open(lockPath.c_str(), O_CREAT | O_RDONLY | O_CLOEXEC, 0666);
        if (lockFd_ < 0) {
            JitWarn(
                "cannot open lock file, compiling without cross-process lock: " + lockPath + ": " +
                std::strerror(errno));
            return;
        }

        const auto waitStart = Clock::now();
        auto nextLogTime = waitStart;
        while (::flock(lockFd_, LOCK_EX | LOCK_NB) != 0) {
            const int error = errno;
            if (error == EINTR) {
                continue;
            }
            if (error != EWOULDBLOCK) {
                release();
                // Any other failure (no flock() on some network mounts, EBADF on NFS) must not stop
                // the compile; atomic publication alone still keeps readers safe.
                JitWarn("flock failed, compiling without cross-process lock: " + lockPath + ": " + std::strerror(error));
                return;
            }

            const auto now = Clock::now();
            const double waitedSeconds = std::chrono::duration<double>(now - waitStart).count();
            if (now - waitStart >= timeout) {
                // A hung compiler on another rank must not hang every other rank with it.
                release();
                JitWarn(
                    "cache lock timeout after " + std::to_string(static_cast<int>(waitedSeconds)) +
                    "s, compiling without cross-process lock: " + lockPath);
                return;
            }
            if (now >= nextLogTime) {
                JitWarn("waiting " + std::to_string(static_cast<int>(waitedSeconds)) + "s for cache lock: " + lockPath);
                nextLogTime = now + kLogInterval;
            }
            std::this_thread::sleep_for(kPollInterval);
        }
    }

    ~CacheEntryLock()
    {
        release();
    }

    CacheEntryLock(const CacheEntryLock&) = delete;
    CacheEntryLock& operator=(const CacheEntryLock&) = delete;

private:
    void release() noexcept
    {
        if (lockFd_ >= 0) {
            (void)::flock(lockFd_, LOCK_UN);
            (void)::close(lockFd_);
            lockFd_ = -1;
        }
    }

    int lockFd_ = -1;
};

std::string MakeTempOutputPath(const std::string& soPath)
{
    char hostname[256] = {};
    if (::gethostname(hostname, sizeof(hostname) - 1) != 0) {
        hostname[0] = '\0';
    }
    std::random_device randomSource;
    return soPath + ".tmp." + hostname + "." + std::to_string(::getpid()) + "." + std::to_string(randomSource()) +
           ".so";
}

} // namespace

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

        const char* ver = std::getenv(JitConfig::kVersionEnv);
        if (ver && *ver) {
            cacheDir += "/" + std::string(ver);
            fs::create_directories(cacheDir, ec);
            JIT_CHECK(!ec, "mkdir failed: " + cacheDir + ": " + ec.message());
        }

        cacheDir_ = std::move(cacheDir);

        bishengPath_ = FindCompilerPath();
        JIT_CHECK(!bishengPath_.empty(), "bisheng compiler not found (set ASCEND_HOME_PATH)");

        includeArgs_ = BuildIncludeArgsFromEnv();
        versionDefine_ = JitConfig::VersionDefine((ver && *ver) ? ver : "unknown");

        npuArch_ = GetCurrentNPUArch();

        templateBase_ = ResolveTemplateBase();

        JIT_LOG(
            JitLogLevel::Info, "JIT init: cache=%s compiler=%s arch=%s template=%s", cacheDir_.c_str(),
            bishengPath_.c_str(), npuArch_.c_str(), templateBase_.empty() ? "(none)" : templateBase_.c_str());
    });
}

JitEntryFn JitCompiler::getKernel(const char* templatePath, const MacroMap& macros, JitKernelType kt)
{
    lazyInit();

    const auto nameIt = macros.find("CATLASS_KERNEL_NAME");
    JIT_CHECK(nameIt != macros.end() && !nameIt->second.empty(), "CATLASS_KERNEL_NAME not set in macros");
    const std::string& targetName = nameIt->second;
    const std::string entrySymbol = "run";

    JIT_CHECK(templatePath && templatePath[0], "templatePath is empty");

    const std::vector<std::string> kernelTypeFlags =
        macros.count("KERNEL_TYPE") ? std::vector<std::string>{} : JitConfig::KernelTypeFlags(kt);

    // Key of loaded_: the compile inputs that can differ between calls. Inputs fixed for the
    // process (compiler, include paths, version) are omitted to keep this lookup cheap.
    auto makeMemoryKey = [&]() -> std::string {
        std::vector<std::pair<std::string, std::string>> inputs;
        inputs.reserve(macros.size() + 5);
        for (const auto& macro : macros) {
            inputs.emplace_back(macro.first, macro.second);
        }
        const char* sanitize = std::getenv(JitConfig::kSanitizeEnv);
        inputs.emplace_back("__ARCH__", npuArch_);
        inputs.emplace_back("__KT__", std::to_string(static_cast<int>(kt)));
        for (const auto& flag : kernelTypeFlags) {
            inputs.emplace_back("__KT_FLAG__", flag);
        }
        inputs.emplace_back("__SANITIZE__", sanitize ? sanitize : "");
        inputs.emplace_back("__TEMPLATE__", templatePath);
        std::sort(inputs.begin(), inputs.end());

        std::string key;
        for (const auto& input : inputs) {
            key += input.first + "=" + input.second + "&";
        }
        return key;
    };

    const std::string memoryKey = makeMemoryKey();

    auto findLoaded = [this, &memoryKey]() -> JitEntryFn {
        std::lock_guard<std::mutex> mapLock(mutex_);
        auto it = loaded_.find(memoryKey);
        return it != loaded_.end() ? it->second.entry : nullptr;
    };

    if (JitEntryFn entry = findLoaded()) {
        JIT_LOG(JitLogLevel::Debug, "mem hit: %s", targetName.c_str());
        return entry;
    }

    const std::vector<std::string> compilerArgs = buildCompilerArgs(targetName, templatePath, macros, kernelTypeFlags);
    const std::string diskKey = [&] {
        std::string input;
        for (const auto& arg : compilerArgs) {
            input += arg;
            input += '\n';
        }
        return Sha256::hash(input);
    }();
    const std::string soPath = cacheDir_ + "/" + diskKey + ".so";

    // Step 1: serialize the threads of this process that want this same entry. mutex_ only guards
    // the map lookup, so threads waiting here never block requests for other kernels.
    std::mutex* entryMutex = nullptr;
    {
        std::lock_guard<std::mutex> mapLock(mutex_);
        entryMutex = &compileMutexes_[diskKey];
    }
    std::lock_guard<std::mutex> entryLock(*entryMutex);

    // Step 2: another thread may have loaded the entry while this one waited.
    if (JitEntryFn entry = findLoaded()) {
        return entry;
    }

    SharedLib lib;

    // Step 3: compile unless the entry is already on disk. The cross-process lock normally lets only
    // one rank of a distributed job compile a given kernel; if it is unavailable or times out, the
    // rank compiles anyway and atomic publication keeps the result safe.
    {
        std::error_code ec;
        if (!fs::is_regular_file(soPath, ec)) {
            const std::string parentDir = fs::path(soPath).parent_path().string();
            fs::create_directories(parentDir, ec);
            JIT_CHECK(!ec, "mkdir failed: " + parentDir + ": " + ec.message());

            CacheEntryLock crossProcessLock(soPath + ".lock");
            // Another process may have published the entry while this one waited for the lock.
            if (!fs::is_regular_file(soPath, ec)) {
                JIT_LOG(JitLogLevel::Info, "compiling: %s \xe2\x86\x92 %s", targetName.c_str(), soPath.c_str());
                lib = compileAndPublish(compilerArgs, soPath);
            }
        }
    }

    // Step 4: retain the validated handle, or load an entry already on disk.
    if (!lib) {
        lib = SharedLib(soPath);
    }
    auto* entry = reinterpret_cast<JitEntryFn>(lib.sym(entrySymbol));

    {
        std::lock_guard<std::mutex> mapLock(mutex_);
        auto it = loaded_.find(memoryKey);
        if (it != loaded_.end()) {
            return it->second.entry;
        }
        loaded_.emplace(memoryKey, LoadedKernel{std::move(lib), entry});
    }

    return entry;
}

SharedLib JitCompiler::compileAndPublish(const std::vector<std::string>& args, const std::string& soPath)
{
    // Compile to a temporary file in the same directory and move it into place with rename(),
    // which is atomic: a reader sees either no entry at all or a complete shared object.
    // A process killed mid-compile leaves its temporary file behind, because deleting stale
    // temporaries is only safe when the lock covers every writer, and it does not on every
    // shared filesystem.
    const std::string tempPath = MakeTempOutputPath(soPath);
    try {
        compile(args, tempPath);
        // Keep the validated handle so library constructors run only once for this compilation.
        SharedLib lib(tempPath);
        (void)lib.sym("run");
        std::error_code ec;
        fs::rename(tempPath, soPath, ec);
        JIT_CHECK(!ec, "publish failed: " + tempPath + " -> " + soPath + ": " + ec.message());
        return lib;
    } catch (...) {
        (void)::unlink(tempPath.c_str());
        throw;
    }
}

void JitCompiler::compile(std::vector<std::string> args, const std::string& soPath)
{
    args.push_back("-o");
    args.push_back(soPath);

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
}

std::vector<std::string> JitCompiler::buildCompilerArgs(
    std::string_view name, std::string_view templatePath, const MacroMap& macros,
    const std::vector<std::string>& kernelTypeFlags)
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

    args.insert(args.end(), kernelTypeFlags.begin(), kernelTypeFlags.end());

    {
        const char* ms = std::getenv(JitConfig::kSanitizeEnv);
        if (ms && std::string(ms) == "1") {
            args.push_back("-g");
            args.push_back("--cce-enable-sanitizer");
            JIT_LOG(JitLogLevel::Info, "msSanitizer ENABLED (MS_SANITIZE_MEMORY=1)");
        }
    }

    args.push_back(versionDefine_);

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
        args.push_back("-DKERNEL_NAME=" + kn);
        kn += "_arch" + npuArch_;
        args.push_back("-DCATLASS_JIT_KERNEL_NAME=" + kn);
    }

    args.insert(args.end(), includeArgs_.begin(), includeArgs_.end());

    args.push_back(templateBase_.empty() ? std::string(templatePath) : templateBase_ + std::string(templatePath));

    return args;
}

void JitCompiler::clearCache()
{
    std::lock_guard<std::mutex> lk(mutex_);
    loaded_.clear();
}

} // namespace CatlassKernel
