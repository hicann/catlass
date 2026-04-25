/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>

#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "common/utils.h"

namespace CatlassKernelWrapper {

torch::Tensor GetOutputTensor(const std::vector<int64_t>& shape, const torch::Dtype dtype)
{
    at::TensorOptions options = at::TensorOptions();
    options =
        options.dtype(dtype).layout(at::kStrided).requires_grad(false).device(torch_npu::utils::get_npu_device_type());
    return at_npu::native::empty_with_format(shape, options, ACL_FORMAT_ND);
}

static const std::vector<std::string> typeStrVec = {
    "float",            // ACL_FLOAT = 0, fp32
    "float16",          // ACL_FLOAT16 = 1
    "int8",             // ACL_INT8 = 2
    "int32",            // ACL_INT32 = 3
    "uint8",            // ACL_UINT8 = 4
    "int16",            // ACL_INT16 = 6
    "uint16",           // ACL_UINT16 = 7
    "uint32",           // ACL_UINT32 = 8
    "int64",            // ACL_INT64 = 9
    "uint64",           // ACL_UINT64 = 10
    "double",           // ACL_DOUBLE = 11
    "bool",             // ACL_BOOL = 12
    "string",           // ACL_STRING = 13
    "complex64",        // ACL_COMPLEX64 = 16
    "complex128",       // ACL_COMPLEX128 = 17
    "bfloat16",         // ACL_BF16 = 27
    "int4",             // ACL_INT4 = 29
    "uint1",            // ACL_UINT1 = 30
    "complex32",        // ACL_COMPLEX32 = 33
    "_hifloat8",        // ACL_HIFLOAT8 = 34, 当前不支持该类型
    "float8_e5m2",      // ACL_FLOAT8_E5M2 = 35
    "float8_e4m3fn",    // ACL_FLOAT8_E4M3FN = 36
    "float8_e8m0fnu",   // ACL_FLOAT8_E8M0 = 37
    "_float6_e3m2",     // ACL_FLOAT6_E3M2 = 38
    "_float6_e2m3",     // ACL_FLOAT6_E2M3 = 39
    "float4_e2m1fn_x2", // ACL_FLOAT4_E2M1 = 40
    "_float4_e1m2",     // ACL_FLOAT4_E1M2 = 41
};
// torch 支持的类型用真实 dtype，其余预留占位
static const std::vector<torch::Dtype> typeTorchVec = {
    torch::kFloat32,
    torch::kFloat16,
    torch::kInt8,
    torch::kInt32,
    torch::kUInt8,
    torch::kInt16,
    torch::kUInt16,
    torch::kUInt32,
    torch::kInt64,
    torch::kUInt64,
    torch::kDouble,
    torch::kBool,
    torch::kFloat16, // string 无 dtype，预留
    torch::kComplexFloat,
    torch::kComplexDouble,
    torch::kBFloat16,
    torch::kInt4,
    torch::kUInt1,
    torch::kComplexHalf, // complex32 -> ComplexHalf 预留
    torch::kFloat16,     // hifloat8 预留
    torch::kFloat8_e5m2,
    torch::kFloat8_e4m3fn,
    torch::kFloat8_e8m0fnu,
    torch::kFloat16,
    torch::kFloat16, // float6 暂无，预留
    torch::kFloat4_e2m1fn_x2,
    torch::kFloat16, // float4_e1m2 暂无，预留
};
static const std::vector<aclDataType> typeAclVec = {
    ACL_FLOAT,       ACL_FLOAT16,     ACL_INT8,        ACL_INT32,         ACL_UINT8,       ACL_INT16,
    ACL_UINT16,      ACL_UINT32,      ACL_INT64,       ACL_UINT64,        ACL_DOUBLE,      ACL_BOOL,
    ACL_STRING,      ACL_COMPLEX64,   ACL_COMPLEX128,  ACL_BF16,          ACL_INT4,        ACL_UINT1,
    ACL_COMPLEX32,   ACL_HIFLOAT8,    ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT6_E3M2,
    ACL_FLOAT6_E2M3, ACL_FLOAT4_E2M1, ACL_FLOAT4_E1M2,
};

template <typename Vec, typename Val>
static int64_t FindTypeIndex(const Vec& vec, const Val& val)
{
    for (size_t i = 0; i < vec.size(); ++i)
        if (vec[i] == val)
            return static_cast<int64_t>(i);
    return -1;
}

template <typename S, typename T>
T TypeCast(const S& src)
{
    if constexpr (std::is_same_v<S, T>) {
        return src;
    }
    int64_t idx = -1;
    if constexpr (std::is_same_v<S, std::string>) {
        idx = FindTypeIndex(typeStrVec, src);
    } else if constexpr (std::is_same_v<S, torch::Dtype>) {
        idx = FindTypeIndex(typeTorchVec, src);
    } else if constexpr (std::is_same_v<S, aclDataType>) {
        idx = FindTypeIndex(typeAclVec, src);
    }
    if (idx == -1) {
        throw std::runtime_error("Invalid type conversion");
    }
    if constexpr (std::is_same_v<T, std::string>) {
        return typeStrVec[idx];
    }
    if constexpr (std::is_same_v<T, torch::Dtype>) {
        return typeTorchVec[idx];
    }
    if constexpr (std::is_same_v<T, aclDataType>) {
        return typeAclVec[idx];
    }

    throw std::runtime_error("Invalid type conversion");
}

torch::Dtype TypeStrToTorchDtype(const std::string& typeStr)
{
    return TypeCast<std::string, torch::Dtype>(typeStr);
}

aclDataType TorchDtypeToAclDtype(const torch::Dtype torchDtype)
{
    return TypeCast<torch::Dtype, aclDataType>(torchDtype);
};

aclDataType TypeStrToAclDtype(const std::string& typeStr)
{
    return TypeCast<std::string, aclDataType>(typeStr);
};

torch::Dtype AclDtypeToTorchDtype(const aclDataType aclDtype)
{
    return TypeCast<aclDataType, torch::Dtype>(aclDtype);
};
[[nodiscard]] TransposeStatus GetTransposeStatus(const at::Tensor& mat)
{
    constexpr int64_t kFormatND = 2;
    constexpr int64_t kFormatNZ = 29;

    int64_t npuFormat = at_npu::native::get_npu_format(mat);

    if (!mat.is_contiguous() && npuFormat == kFormatND) {
        return TransposeStatus::NON_CONTINUOUS;
    }

    if (mat.dim() < 2) {
        return TransposeStatus::NO_TRANSPOSE;
    }

    const auto& strides = mat.strides();
    const auto& shape = mat.sizes();

    int64_t dimA = shape[mat.dim() - 2];
    int64_t dimB = shape[mat.dim() - 1];
    int64_t strideA = strides[mat.dim() - 2];
    int64_t strideB = strides[mat.dim() - 1];

    if (npuFormat == kFormatND) {
        if (strideA == dimB && strideB == 1) {
            return TransposeStatus::NO_TRANSPOSE;
        }
        if (strideA == 1 && strideB == dimA) {
            return TransposeStatus::TRANSPOSE;
        }
        return TransposeStatus::NON_CONTINUOUS;
    }

    if (npuFormat == kFormatNZ) {
        if (strideA == dimB && strideB == 1) {
            return TransposeStatus::LAYOUT_ZN;
        }
        if (strideA == 1 && strideB == dimA) {
            return TransposeStatus::LAYOUT_NZ;
        }
        return TransposeStatus::NON_CONTINUOUS;
    }

    return TransposeStatus::NON_CONTINUOUS;
}

} // namespace CatlassKernelWrapper
