/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_LIB_COMMON_KERNEL_CHOOSER_HPP
#define SHARED_LIB_COMMON_KERNEL_CHOOSER_HPP

#include <tuple>
#include <type_traits>
#include <acl/acl.h>

#include "common.hpp"

namespace CatlassKernel {
namespace detail {

// BoolVariable: 用于存储布尔参数包
template <bool... T>
struct BoolVariable {};

// EnumWrapper: 用于在 tuple 中标记和传递枚举值，作为编译期模板参数
// 使用方式：在 tuple 中使用 EnumWrapper<EnumType, EnumValue> 来传递编译期的枚举值
// 枚举值的运行时分发应在调用处根据实际值直接选择对应的模板实例化，避免使用 if-else 链
// 
// 示例：
//   // 在调用处直接根据枚举值选择
//   if (enumValue == MyEnum::VALUE1) {
//       KernelChooser<std::tuple<..., EnumWrapper<MyEnum, MyEnum::VALUE1>>, ...>::call(...);
//   } else if (enumValue == MyEnum::VALUE2) {
//       KernelChooser<std::tuple<..., EnumWrapper<MyEnum, MyEnum::VALUE2>>, ...>::call(...);
//   }
template <typename EnumType, EnumType EnumValue>
struct EnumWrapper {
    using enum_type = EnumType;
    static constexpr EnumType value = EnumValue;
};

// KernelChooser: 递归展开布尔参数，将运行时布尔值转换为编译期类型
template <typename T0, typename T1>
struct KernelChooser;

// 特化：当还有布尔参数需要处理时
template <typename... Ts0, bool... Ts1>
struct KernelChooser<std::tuple<Ts0...>, BoolVariable<Ts1...>> {
    // 处理布尔参数：递归展开
    template <typename... Args>
    static void call(bool x, Args... args) {
        if (x) {
            KernelChooser<std::tuple<Ts0...>, BoolVariable<Ts1..., true>>::call(args...);
        } else {
            KernelChooser<std::tuple<Ts0...>, BoolVariable<Ts1..., false>>::call(args...);
        }

    }

    // 终止条件：所有布尔参数都已处理，调用最终的内核函数
    template <typename... Args>
    static void call(Args... args) {
        // 这个函数会被特化版本覆盖，用于实际调用内核
    }
};

// 类型选择器：根据 aclDataType 选择对应的 C++ 类型
template <aclDataType DT>
struct DataTypeSelector {
    using type = typename AclType2Type<DT>::type;
};

// 布局选择器：根据转置标志选择布局
// 默认实现：只根据Transpose选择RowMajor或ColumnMajor（向后兼容）
template <bool Transpose>
struct LayoutSelector {
    using type = std::conditional_t<Transpose, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>;
};

// 扩展的布局选择器：根据是否使用分块布局和transpose标志选择布局
// useFormat为true时：使用zN/nZ格式
//   - useFormat=true, trans=false -> zN (分块内行优先，分块间列优先)
//   - useFormat=true, trans=true -> nZ (分块内列优先，分块间行优先)
// useFormat为false时：使用常规RowMajor/ColumnMajor格式
//   - useFormat=false, trans=false -> RowMajor
//   - useFormat=false, trans=true -> ColumnMajor
template <bool UseFormat, bool Transpose>
struct FormatLayoutSelector {
    using type = std::conditional_t<UseFormat,
        std::conditional_t<Transpose, Catlass::layout::nZ, Catlass::layout::zN>,
        std::conditional_t<Transpose, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>
    >;
};

// ============================================================================
// 通用 fp32 TileShape 适配器：统一处理 fp32 类型的 K 维度减半
// ============================================================================
//
// fp32 类型需要将 K 维度减半以避免 L1 空间溢出
// 规律：L1TileShape 的 K 从 256 变为 128，L0TileShape 的 K 从 64 变为 32
//
// 使用方式：
//   template <typename InDType>
//   struct MyKernelTileShapeSelector {
//       using L1TileShape = typename Fp32TileShapeAdapter<InDType, Catlass::GemmShape<128, 256, 256>>::type;
//       using L0TileShape = typename Fp32TileShapeAdapter<InDType, Catlass::GemmShape<128, 256, 64>>::type;
//   };
//
// 或者对于有条件的 TileShape：
//   template <typename InDType, bool Condition>
//   struct MyKernelTileShapeSelector {
//       using BaseL1TileShape = std::conditional_t<Condition, 
//           Catlass::GemmShape<128, 256, 256>, 
//           Catlass::GemmShape<256, 128, 256>>;
//       using BaseL0TileShape = std::conditional_t<Condition,
//           Catlass::GemmShape<128, 256, 64>,
//           Catlass::GemmShape<256, 128, 64>>;
//       using L1TileShape = typename Fp32TileShapeAdapter<InDType, BaseL1TileShape>::type;
//       using L0TileShape = typename Fp32TileShapeAdapter<InDType, BaseL0TileShape>::type;
//   };

// 默认实现：非 fp32 类型，保持原 TileShape
template <typename InDType, typename TileShape>
struct Fp32TileShapeAdapter {
    using type = TileShape;
};

// fp32 特化：将 K 维度减半
template <uint32_t M, uint32_t N, uint32_t K>
struct Fp32TileShapeAdapter<float, Catlass::GemmShape<M, N, K>> {
    using type = Catlass::GemmShape<M, N, K / 2>;
};

// ============================================================================
// 通用数据类型分发器：用于所有 kernel 的数据类型分发
// ============================================================================

// 类型组合检查器：用于检查某个类型组合是否被支持
// 默认实现：所有组合都不支持（需要特化）
template <typename Dispatcher, aclDataType InDT, aclDataType OutDT>
struct DataTypeCombinationChecker {
    static constexpr bool is_supported = false;
};

// 前向声明
template <typename Dispatcher, aclDataType InDT>
struct DataTypeDispatcherByInput;

// 根据输入数据类型特化的分发器：处理输出数据类型分发
// 必须先定义，因为 DataTypeDispatcher 需要使用它
template <typename Dispatcher, aclDataType InDT>
struct DataTypeDispatcherByInput {
    template <typename... Args>
    static void dispatch(aclDataType outDT, Args... args) {
        // 使用类型组合检查器，只调用支持的类型组合
        if (outDT == ACL_FLOAT16) {
            if constexpr (DataTypeCombinationChecker<Dispatcher, InDT, ACL_FLOAT16>::is_supported) {
                Dispatcher::template dispatch<InDT, ACL_FLOAT16>(args...);
            }
        } else if (outDT == ACL_BF16) {
            if constexpr (DataTypeCombinationChecker<Dispatcher, InDT, ACL_BF16>::is_supported) {
                Dispatcher::template dispatch<InDT, ACL_BF16>(args...);
            }
        } else if (outDT == ACL_FLOAT) {
            if constexpr (DataTypeCombinationChecker<Dispatcher, InDT, ACL_FLOAT>::is_supported) {
                Dispatcher::template dispatch<InDT, ACL_FLOAT>(args...);
            }
        } else if (outDT == ACL_INT32) {
            if constexpr (DataTypeCombinationChecker<Dispatcher, InDT, ACL_INT32>::is_supported) {
                Dispatcher::template dispatch<InDT, ACL_INT32>(args...);
            }
        }
    }
};

// 通用的数据类型分发器模板
// 
// 使用步骤：
//   1. 定义自己的 Dispatcher 结构体，实现 dispatch 方法
//   2. 特化 DataTypeCombinationChecker，定义支持的类型组合
//   3. 在主函数中使用 DataTypeDispatcher
//
// 完整示例：
//   // 步骤1：定义 Dispatcher
//   struct MyKernelDataTypeDispatcher {
//       template <aclDataType InDT, aclDataType OutDT>
//       static void dispatch(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo) {
//           // 实际的 kernel 调用逻辑
//           using InType = typename DataTypeSelector<InDT>::type;
//           using OutType = typename DataTypeSelector<OutDT>::type;
//           MyKernelImpl<InType, OutType>(blockNum, stream, kernelInfo);
//       }
//   };
//
//   // 步骤2：特化类型组合检查器，定义支持的类型组合
//   // 例如：只支持 ACL_INT8 -> ACL_FLOAT16
//   template <>
//   struct DataTypeCombinationChecker<MyKernelDataTypeDispatcher, ACL_INT8, ACL_FLOAT16> {
//       static constexpr bool is_supported = true;
//   };
//   // 或者支持多个组合：
//   template <aclDataType InDT, aclDataType OutDT>
//   struct DataTypeCombinationChecker<MyKernelDataTypeDispatcher, InDT, OutDT> {
//       static constexpr bool is_supported = 
//           (InDT == ACL_INT8 && OutDT == ACL_FLOAT16) ||
//           (InDT == ACL_FLOAT16 && OutDT == ACL_FLOAT16);
//   };
//
//   // 步骤3：在主函数中使用
//   void MyKernel(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo) {
//       DataTypeDispatcher<MyKernelDataTypeDispatcher>::dispatch(
//           kernelInfo.inputDataType, kernelInfo.outputDataType, blockNum, stream, kernelInfo
//       );
//   }
//
// 注意：
//   - 如果不特化 DataTypeCombinationChecker，默认所有组合都不支持（不会编译错误，但不会执行）
//   - 类型组合检查在编译期完成，不会产生运行时开销
//   - 支持的类型组合可以根据每个 kernel 的实际需求灵活定义
template <typename Dispatcher>
struct DataTypeDispatcher {
    // 根据输入数据类型分发到输出数据类型分发器
    template <typename... Args>
    static void dispatch(aclDataType inDT, aclDataType outDT, Args... args) {
        if (inDT == ACL_FLOAT16) {
            DataTypeDispatcherByInput<Dispatcher, ACL_FLOAT16>::dispatch(outDT, args...);
        } else if (inDT == ACL_BF16) {
            DataTypeDispatcherByInput<Dispatcher, ACL_BF16>::dispatch(outDT, args...);
        } else if (inDT == ACL_FLOAT) {
            DataTypeDispatcherByInput<Dispatcher, ACL_FLOAT>::dispatch(outDT, args...);
        } else if (inDT == ACL_INT8) {
            DataTypeDispatcherByInput<Dispatcher, ACL_INT8>::dispatch(outDT, args...);
        }
    }
};

} // namespace detail
} // namespace CatlassKernel

#endif // SHARED_LIB_COMMON_KERNEL_CHOOSER_HPP


