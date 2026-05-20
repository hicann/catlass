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
#include <unordered_map>
#include <vector>

#include <acl/acl.h>

#include "catlass_kernel.h"
#include "jit_util.h"
#include "kernel_utils.h"

namespace CatlassKernel {

/**
 * @brief Convert transpose/NZ flags to a CATLASS layout type token.
 * @param isTranspose Whether the logical matrix is transposed.
 * @param isNz Whether the tensor uses CATLASS NZ block layout.
 * @return Layout type name for JIT macro injection.
 */
inline const char* LayoutToStr(bool isTranspose, bool isNz)
{
    if (isNz)
        return isTranspose ? "nZ" : "zN";
    return isTranspose ? "ColumnMajor" : "RowMajor";
}

/**
 * @brief Default JIT macro generation policy.
 *
 * 主模板不产生任何宏。各 kernel 类型应通过偏特化实现自己的
 * 宏生成逻辑。使用 SFINAE (void_t) 技术，可对特定 TParams 子类型
 * 启用/禁用不同的特化版本。
 *
 * @tparam TParams 模板参数类型（如 MatmulTParams、ConvolutionParams 等）。
 * @tparam TEnable SFINAE 触发参数（默认 void，匹配主模板）。
 *
 * @code
 *   // 调用方式
 *   std::unordered_map<std::string, std::string> macros;
 *   JitMacroGenerator<MatmulTParams>::appendTo(macros, tParams);
 * @endcode
 */
template <typename TParams, typename = void>
struct JitMacroGenerator {
    /**
     * @brief 将模板参数转换为编译宏（基模板：空操作）。
     *
     * 不修改 macros 映射。特化版本将 TParams 中各字段转换为
     * -D 宏定义，供 JIT 编译器模板使用。
     *
     * @param macros [in/out] 宏名→宏值映射，会被追加。
     * @param p      模板参数实例（未被本基模板使用）。
     */
    static void appendTo(std::unordered_map<std::string, std::string>& /*macros*/, const TParams& /*p*/)
    {}

    /**
     * @brief 生成完整宏映射（含 kernel 缓存名和函数名）。
     *
     * 一次性创建并填充 macros 映射，包括：
     *   - CATLASS_KERNEL_NAME       ← kernelName（缓存目录名）
     *   - appendTo 生成的编译宏
     *   - CATLASS_JIT_KERNEL_NAME   ← makeKernelName 生成的描述性函数名
     *
     * @param kernelName  缓存用短名，如 "basic_matmul"。
     * @param p           模板参数引用。
     * @return std::unordered_map<std::string, std::string>  宏名→宏值映射。
     */
    static std::unordered_map<std::string, std::string> generate(const char* kernelName, const TParams& p)
    {
        std::unordered_map<std::string, std::string> macros;
        macros["CATLASS_KERNEL_NAME"] = kernelName;
        JitMacroGenerator::appendTo(macros, p);
        macros["CATLASS_JIT_KERNEL_NAME"] = makeKernelName(macros, p);
        return macros;
    }

    /**
     * @brief 从 macros 映射构造描述性 kernel 函数名（基模板：空）。
     *
     * buildCompilerArgs 会在 CATLASS_JIT_KERNEL_NAME 为空时回退到 target name。
     */
    static std::string makeKernelName(
        const std::unordered_map<std::string, std::string>& /*macros*/, const TParams& /*p*/)
    {
        return {};
    }
};

/**
 * @brief JitMacroGenerator 对 MatmulTParams 的特化。
 *
 * 将 MatmulTParams 中的数据类型和布局信息编码为编译宏：
 *   - CATLASS_JIT_ELEMENT_A / _B / _C → bisheng 类型名
 *   - CATLASS_JIT_LAYOUT_A / _B / _C  → 布局标签
 *
 * 生成的宏定义会被 JIT 编译器拼接为 -D 参数传入 bisheng 编译器，
 * 受 cache key 机制保护——任何宏值变更都会触发重新编译。
 *
 * 使用示例见基模板 @ref JitMacroGenerator。
 */
template <>
struct JitMacroGenerator<MatmulTParams> {
    /**
     * @brief 将 MatmulTParams 各字段写入 macros 映射。
     *
     * @param macros [in/out] 宏名→宏值映射，追加以下条目：
     *   - "CATLASS_JIT_ELEMENT_A" ← AclDtypeToBishengTypeStr(p.elementA)
     *   - "CATLASS_JIT_ELEMENT_B" ← AclDtypeToBishengTypeStr(p.elementB)
     *   - "CATLASS_JIT_ELEMENT_C" ← AclDtypeToBishengTypeStr(p.elementC)
     *   - "CATLASS_JIT_LAYOUT_A"  ← LayoutToStr(p.useNzA, p.transA)
     *   - "CATLASS_JIT_LAYOUT_B"  ← LayoutToStr(p.useNzB, p.transB)
     *   - "CATLASS_JIT_LAYOUT_C"  ← "RowMajor"（固定）
     * @param p MatmulTParams 引用。
     */
    static void appendTo(std::unordered_map<std::string, std::string>& macros, const MatmulTParams& p)
    {
        macros["CATLASS_JIT_ELEMENT_A"] = AclDtypeToBishengTypeStr(p.elementA);
        macros["CATLASS_JIT_ELEMENT_B"] = AclDtypeToBishengTypeStr(p.elementB);
        macros["CATLASS_JIT_ELEMENT_C"] = AclDtypeToBishengTypeStr(p.elementC);
        macros["CATLASS_JIT_LAYOUT_A"] = LayoutToStr(p.transA, p.useNzA);
        macros["CATLASS_JIT_LAYOUT_B"] = LayoutToStr(p.transB, p.useNzB);
        macros["CATLASS_JIT_LAYOUT_C"] = "RowMajor";
    }

    /**
     * @brief 生成完整宏映射（含 kernel 缓存名和函数名），MatmulTParams 特化。
     *
     * @param kernelName  缓存用短名，如 "basic_matmul"。
     * @param p MatmulTParams 引用。
     * @return std::unordered_map<std::string, std::string>  宏名→宏值映射。
     */
    static std::unordered_map<std::string, std::string> generate(const char* kernelName, const MatmulTParams& p)
    {
        std::unordered_map<std::string, std::string> macros;
        macros["CATLASS_KERNEL_NAME"] = kernelName;
        appendTo(macros, p);
        macros["CATLASS_JIT_KERNEL_NAME"] = makeKernelName(macros, p);
        return macros;
    }

    /**
     * @brief 从 appendTo 写入的宏值拼接 kernel 函数名，零冗余调用。
     */
    static std::string makeKernelName(
        const std::unordered_map<std::string, std::string>& macros, const MatmulTParams& /*p*/)
    {
        auto get = [&](const std::string& key) -> const std::string& {
            static const std::string empty;
            auto it = macros.find(key);
            return it != macros.end() ? it->second : empty;
        };
        return "matmul_" + get("CATLASS_JIT_ELEMENT_A") + "_" + get("CATLASS_JIT_ELEMENT_B") + "_" +
               get("CATLASS_JIT_ELEMENT_C") + "_" + get("CATLASS_JIT_LAYOUT_A") + "_" + get("CATLASS_JIT_LAYOUT_B") +
               "_" + get("CATLASS_JIT_LAYOUT_C");
    }
};

} // namespace CatlassKernel
