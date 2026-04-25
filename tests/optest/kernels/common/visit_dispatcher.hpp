/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_LIB_COMMON_VISIT_DISPATCHER_HPP
#define SHARED_LIB_COMMON_VISIT_DISPATCHER_HPP

#include <variant>
#include <type_traits>
#include <acl/acl.h>
#include "common.hpp"

namespace CatlassKernel {
namespace detail {

// 类型标签：用于 variant 中的类型标识
struct FloatTag {};
struct HalfTag {};
struct BFloat16Tag {};
struct Int4Tag {};
struct Int8Tag {};
struct Int32Tag {};
struct Int64Tag {};
struct UInt64Tag {};

// 数据类型 Variant：使用标签而非实际类型，避免存储开销
using DataTypeVariant = std::variant<
    std::monostate,  // 不支持的类型
    FloatTag,
    HalfTag,
    BFloat16Tag,
    Int4Tag,
    Int8Tag,
    Int32Tag,
    Int64Tag,
    UInt64Tag
>;

// 将 aclDataType 转换为 variant
inline DataTypeVariant aclTypeToVariant(aclDataType dt) {
    switch (dt) {
        case ACL_FLOAT:
            return DataTypeVariant{FloatTag{}};
        case ACL_FLOAT16:
            return DataTypeVariant{HalfTag{}};
        case ACL_BF16:
            return DataTypeVariant{BFloat16Tag{}};
        case ACL_INT4:
            return DataTypeVariant{Int4Tag{}};
        case ACL_INT8:
            return DataTypeVariant{Int8Tag{}};
        case ACL_INT32:
            return DataTypeVariant{Int32Tag{}};
        case ACL_INT64:
            return DataTypeVariant{Int64Tag{}};
        case ACL_UINT64:
            return DataTypeVariant{UInt64Tag{}};
        default:
            return DataTypeVariant{std::monostate{}};
    }
}

// 从 variant 标签获取对应的 C++ 类型
template<typename Tag>
struct TagToType;

template<>
struct TagToType<FloatTag> {
    using type = float;
};

template<>
struct TagToType<HalfTag> {
    using type = half;
};

template<>
struct TagToType<BFloat16Tag> {
    using type = bfloat16_t;
};

template<>
struct TagToType<Int4Tag> {
    using type = AscendC::int4b_t;
};

template<>
struct TagToType<Int8Tag> {
    using type = int8_t;
};

template<>
struct TagToType<Int32Tag> {
    using type = int32_t;
};

template<>
struct TagToType<Int64Tag> {
    using type = int64_t;
};

template<>
struct TagToType<UInt64Tag> {
    using type = uint64_t;
};

// 类型组合支持检查器（可特化）
// 默认所有组合都不支持，需要显式特化
template<typename InTag, typename OutTag>
struct TypeCombinationSupported {
    static constexpr bool value = false;
};

// Visit 风格的类型分发器
// 
// 使用示例：
//   visitDataTypePair(kernelInfo.inputDataType, kernelInfo.outputDataType,
//       [&](auto inType, auto outType) {
//           using InType = typename TagToType<decltype(inType)>::type;
//           using OutType = typename TagToType<decltype(outType)>::type;
//           if constexpr (TypeCombinationSupported<decltype(inType), decltype(outType)>::value) {
//               MyKernelImpl<InType, OutType>(...);
//           }
//       });
template<typename Visitor>
void visitDataTypePair(aclDataType inDT, aclDataType outDT, Visitor&& visitor) {
    auto inVariant = aclTypeToVariant(inDT);
    auto outVariant = aclTypeToVariant(outDT);
    
    std::visit([&](auto inTag, auto outTag) {
        // 检查是否为支持的类型组合
        if constexpr (!std::is_same_v<decltype(inTag), std::monostate> &&
                      !std::is_same_v<decltype(outTag), std::monostate>) {
            if constexpr (TypeCombinationSupported<decltype(inTag), decltype(outTag)>::value) {
                visitor(inTag, outTag);
            }
        }
    }, inVariant, outVariant);
}

// 单类型 visit（用于只有一个数据类型的情况）
template<typename Visitor>
void visitDataType(aclDataType dt, Visitor&& visitor) {
    auto variant = aclTypeToVariant(dt);
    
    std::visit([&](auto tag) {
        if constexpr (!std::is_same_v<decltype(tag), std::monostate>) {
            visitor(tag);
        }
    }, variant);
}

// ============================================================================
// 便捷宏：简化类型组合支持定义
// ============================================================================

// 定义单个类型组合支持
#define DEFINE_TYPE_COMBINATION_SUPPORTED(InTag, OutTag) \
    template<> \
    struct TypeCombinationSupported<InTag, OutTag> { \
        static constexpr bool value = true; \
    }

} // namespace detail
} // namespace CatlassKernel

#endif // SHARED_LIB_COMMON_VISIT_DISPATCHER_HPP


