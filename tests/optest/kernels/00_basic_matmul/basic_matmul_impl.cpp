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

/**
 * JIT kernel 模板 — basic_matmul
 * 由运行时 bisheng -D... 编译，所有可变参数通过宏注入
 */

#include <cstddef>
using std::size_t;

#include <kernel_operator.h>

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A half
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B half
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C half
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif

/** ---- 类型/布局别名 ---- */
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutA = Catlass::layout::CATLASS_JIT_LAYOUT_A;
using LayoutB = Catlass::layout::CATLASS_JIT_LAYOUT_B;
using LayoutC = Catlass::layout::CATLASS_JIT_LAYOUT_C;

/** ---- 固定编译期参数 ---- */
using ArchTag = Catlass::Arch::AtlasA2;
using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;

/** ---- TileShape（仅依赖 JitInDType） ---- */
/**
 * @brief Rebuild a tile shape while scaling the K tile by a compile-time factor.
 */
template <uint32_t M, uint32_t N, uint32_t K, uint32_t KScale>
struct TileShapeKScaler {
    static_assert(K % KScale == 0, "K tile size must be divisible by KScale");
    using type = Catlass::GemmShape<M, N, K / KScale>;
};

/**
 * @brief Keep the default tile shape unless a dtype-specific specialization overrides it.
 */
template <typename T, typename Shape>
struct ShapeAdapter {
    using type = Shape;
};

/**
 * @brief Keep the full K tile for fp16 inputs.
 */
template <uint32_t M, uint32_t N, uint32_t K>
struct ShapeAdapter<half, Catlass::GemmShape<M, N, K>> : TileShapeKScaler<M, N, K, 1> {
};

/**
 * @brief Keep the full K tile for bf16 inputs.
 */
template <uint32_t M, uint32_t N, uint32_t K>
struct ShapeAdapter<bfloat16_t, Catlass::GemmShape<M, N, K>> : TileShapeKScaler<M, N, K, 1> {
};

/**
 * @brief Halve the K tile for fp32 inputs to account for larger element width.
 */
template <uint32_t M, uint32_t N, uint32_t K>
struct ShapeAdapter<float, Catlass::GemmShape<M, N, K>> : TileShapeKScaler<M, N, K, 2> {
};

using L1TileShape = typename ShapeAdapter<ElementA, Catlass::GemmShape<128, 256, 256>>::type;
using L0TileShape = typename ShapeAdapter<ElementA, Catlass::GemmShape<128, 256, 64>>::type;

/** ---- Block 组件 ---- */
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"

using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue = void;
using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

using MatmulKernel = typename Catlass::Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

/** ---- Kernel 函数（名称由 JIT 编译器注入，便于 prof 识别） ---- */
#ifndef CATLASS_JIT_KERNEL_NAME
#define CATLASS_JIT_KERNEL_NAME BasicMatmulKernel
#endif

extern "C" CATLASS_GLOBAL __attribute__((aic)) void CATLASS_JIT_KERNEL_NAME(
    Catlass::GemmCoord shape, GM_ADDR ptrA, LayoutA layoutA, GM_ADDR ptrB, LayoutB layoutB, GM_ADDR ptrC,
    LayoutC layoutC)
{
    typename MatmulKernel::Params params{shape, ptrA, layoutA, ptrB, layoutB, ptrC, layoutC};
    MatmulKernel matmul;
    matmul(params);
}

/** ---- C ABI 入口 ---- */
#include "catlass_kernel.h"

/**
 * @brief Stable C ABI called by the JIT loader after symbol resolution.
 */
extern "C" void run(
    uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulTParams* /*tParams*/,
    const CatlassKernel::MatmulParams* params)
{
    Catlass::GemmCoord shape{params->m, params->n, params->k};

    LayoutA la = LayoutA::template MakeLayout<ElementA>(shape.m(), shape.k());
    LayoutB lb = LayoutB::template MakeLayout<ElementB>(shape.k(), shape.n());
    LayoutC lc = LayoutC::template MakeLayout<ElementC>(shape.m(), shape.n());

    CATLASS_JIT_KERNEL_NAME<<<blockNum, nullptr, stream>>>(
        shape, params->inputAddr[0], la, params->inputAddr[1], lb, params->outputAddr[0], lc);
}
