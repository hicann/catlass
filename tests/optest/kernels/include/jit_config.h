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

#pragma once

#include <string>
#include <vector>

namespace CatlassKernel {
namespace JitConfig {

// ── 环境变量名称 ──
inline constexpr const char* kLogLevelEnv = "CATLASS_JIT_LOG_LEVEL";
inline constexpr const char* kCacheDirEnv = "TORCH_CATLASS_CACHE_DIR";
inline constexpr const char* kSanitizeEnv = "MS_SANITIZE_MEMORY";
inline constexpr const char* kVersionEnv = "TORCH_CATLASS_VERSION";
inline constexpr const char* kAscendHomeEnv = "ASCEND_HOME_PATH";
inline constexpr const char* kPkgDirEnv = "TORCH_CATLASS_PKG_DIR";

// ── 默认路径 ──
inline constexpr const char* kDefaultCacheDir = "/tmp/catlass_jit";
inline constexpr const char* kHomeCacheSubdir = ".cache/catlass/jit_cache";

// ── 编译器候选路径后缀 ──
inline constexpr const char* kCcecSuffixes[] = {
    "/compiler/bin/ccec",
    "/tools/bisheng_compiler/bin/ccec",
    "/bin/ccec",
};

/**
 * @brief Return compiler flags shared by all JIT kernel compilations.
 */
inline std::vector<std::string> BaseFlags()
{
    return {"-std=c++17", "-O2", "-shared"};
}

/**
 * @brief Return compiler flags derived from the target NPU architecture.
 * @param arch CATLASS architecture id, for example "2201" or "3510".
 */
inline std::vector<std::string> ArchFlags(const std::string& arch)
{
    return {"--npu-arch=dav-" + arch, "-DCATLASS_ARCH=" + arch};
}

/**
 * @brief Build the preprocessor define carrying the package version.
 * @param version Version string exported by the Python package loader.
 */
inline std::string VersionDefine(const std::string& version)
{
    return "-DCATLASS_VERSION_FULL=" + version;
}

} // namespace JitConfig
} // namespace CatlassKernel
