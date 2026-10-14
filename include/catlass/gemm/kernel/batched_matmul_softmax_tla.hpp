/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 */

#ifndef CATLASS_GEMM_KERNEL_BATCHED_MATMUL_SOFTMAX_TLA_HPP
#define CATLASS_GEMM_KERNEL_BATCHED_MATMUL_SOFTMAX_TLA_HPP

#include <cstddef>
#include <cstdint>

#include "catlass/epilogue/block/block_epilogue_batched_matmul_softmax.hpp"
#include "catlass/gemm/kernel/batched_matmul_softmax_tla_tuning.hpp"

#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/layout.hpp"

namespace Catlass::Gemm::Kernel {

namespace BatchedMatmulSoftmaxTuning {

// The completion flag is published after the full N dimension is written.
// AIV can then consume complete rows without a per-tile synchronization race.
constexpr uint32_t CROSS_CORE_BARRIER_MODE = 0x0;
constexpr uint32_t CROSS_CORE_COMPLETION_MODE = 0x2;

} // namespace BatchedMatmulSoftmaxTuning

// Fuses batched matmul and row-wise softmax in one MIX kernel launch.
template <class MatmulKernel_, class LayoutTagA_, class LayoutTagB_, class LayoutTagC_>
class BatchedMatmulSoftmaxTla {
public:
    using MatmulKernel = MatmulKernel_;
    using ElementA = typename MatmulKernel::ElementA;
    using ElementB = typename MatmulKernel::ElementB;
    using ElementC = typename MatmulKernel::ElementC;
    using ArchTag = typename MatmulKernel::ArchTag;
    using LayoutTagA = LayoutTagA_;
    using LayoutTagB = LayoutTagB_;
    using LayoutTagC = LayoutTagC_;
    using SoftmaxEpilogue = Epilogue::Block::BlockEpilogueBatchedMatmulSoftmax<ArchTag, ElementC>;

    struct Arguments {
        uint32_t batch;
        GemmCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
    };

    struct Params {
        uint32_t batch;
        uint32_t m;
        uint32_t n;
        uint32_t k;
        uint32_t alignedN;
        uint32_t rowsPerTile;
        uint32_t useBasicBlock;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
    };

    static bool CanImplement(Arguments const& args)
    {
        return args.batch > 0 && args.problemShape.m() > 0 && args.problemShape.n() > 0 && args.problemShape.k() > 0 &&
               args.ptrA != nullptr && args.ptrB != nullptr && args.ptrC != nullptr;
    }

    static size_t GetWorkspaceSize(Arguments const&)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(Arguments const& args, uint8_t*)
    {
        uint32_t alignedN = BatchedMatmulSoftmaxTuning::AlignSoftmaxN(args.problemShape.n());
        uint32_t rowsPerTile = BatchedMatmulSoftmaxTuning::SelectSoftmaxRowsPerTile(alignedN);
        return Params{
            args.batch,
            args.problemShape.m(),
            args.problemShape.n(),
            args.problemShape.k(),
            alignedN,
            rowsPerTile,
            BatchedMatmulSoftmaxTuning::SelectSoftmaxBasicBlock(args.problemShape.n(), alignedN, rowsPerTile),
            args.ptrA,
            args.ptrB,
            args.ptrC};
    }

    CATLASS_DEVICE
    BatchedMatmulSoftmaxTla() : aicDone(AIC_DONE_FLAG)
    {}

    CATLASS_DEVICE void operator()(Params const& params)
    {
        if ASCEND_IS_AIC {
            auto problemShape = GemmCoord{params.m, params.n, params.k};
            auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(params.m, params.k);
            auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(params.k, params.n);
            auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(params.m, params.n);
            typename MatmulKernel::Params matmulParams{
                params.batch,
                problemShape,
                params.ptrA,
                layoutA,
                static_cast<int64_t>(params.m) * params.k,
                params.ptrB,
                layoutB,
                static_cast<int64_t>(params.k) * params.n,
                params.ptrC,
                layoutC,
                static_cast<int64_t>(params.m) * params.n};
            MatmulKernel matmul;
            matmul.template operator()<AscendC::AIC>(matmulParams);
            // Softmax reduces the complete N dimension, so AIV starts only
            // after the whole batched matmul has completed.
            Arch::CrossCoreBarrier<BatchedMatmulSoftmaxTuning::CROSS_CORE_BARRIER_MODE, PIPE_FIX>();
            Arch::CrossCoreSetFlag<BatchedMatmulSoftmaxTuning::CROSS_CORE_COMPLETION_MODE, PIPE_FIX>(aicDone);
        }
        if ASCEND_IS_AIV {
            Arch::CrossCoreWaitFlag<BatchedMatmulSoftmaxTuning::CROSS_CORE_COMPLETION_MODE, PIPE_MTE2>(aicDone);
            SoftmaxEpilogue epilogue(resource);
            epilogue(
                params.ptrC, params.batch, params.m, params.n, params.alignedN, params.rowsPerTile,
                params.useBasicBlock);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID AIC_DONE_FLAG = 0;
    Arch::CrossCoreFlag aicDone;
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_BATCHED_MATMUL_SOFTMAX_TLA_HPP
