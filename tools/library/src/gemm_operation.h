/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_LIBRARY_GEMM_OPERATION_H
#define CATLASS_LIBRARY_GEMM_OPERATION_H

#include <type_traits>
#include "catlass/library/library.h"
#include "library_utils.h"

namespace Catlass {
namespace Library {

template <typename Operator_>
class GemmOperationBase : public Operation {
public:
    using Operator = Operator_;
    using OperatorArguments = typename Operator::Arguments;
    using OperatorKernel = typename Operator::Kernel;

    using ElementA = typename OperatorKernel::ElementA;
    using ElementB = typename OperatorKernel::ElementB;
    using ElementC = typename OperatorKernel::ElementC;
    using LayoutA = typename OperatorKernel::LayoutA;
    using LayoutB = typename OperatorKernel::LayoutB;
    using LayoutC = typename OperatorKernel::LayoutC;
    using BlockMmad = typename OperatorKernel::BlockMmad;
    using ArchTag = typename OperatorKernel::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;
    using BlockScheduler = typename OperatorKernel::BlockScheduler;

    GemmOperationBase(char const *name = "")
    {
        this->description_.name = name;
        this->description_.kind = OperationKind::Gemm;

        this->description_.A = MakeTensorDescription<ElementA, LayoutA>();
        this->description_.B = MakeTensorDescription<ElementB, LayoutB>();
        this->description_.C = MakeTensorDescription<ElementC, LayoutC>();

        this->description_.tileDescription.L1TileShape =
            GemmShapeDescription(L1TileShape::M, L1TileShape::N, L1TileShape::K);
        this->description_.tileDescription.L0TileShape =
            GemmShapeDescription(L0TileShape::M, L0TileShape::N, L0TileShape::K);
    }

    virtual OperationDescription const &GetDescription() const override
    {
        return this->description_;
    }

    virtual Status CanImplement(void *argsPtr, void *configPtr) override
    {
        BuildArgs(argsPtr, configPtr);
        return op_.CanImplement(this->args_);
    }

    virtual size_t GetWorkspaceSize(void *argsPtr, void *configPtr) override
    {
        BuildArgs(argsPtr, configPtr);
        return op_.GetWorkspaceSize(this->args_);
    }

    virtual Status Initialize(
        void *argsPtr,
        void *configPtr,
        uint8_t *workspace,
        aclrtStream stream
    ) override
    {
        BuildArgs(argsPtr, configPtr);
        return op_.Initialize(this->args_, workspace, stream);
    }

    virtual Status Run(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr) override
    {
        return op_.Run(stream, blockDim, fftsAddr);
    }

protected:
    virtual void BuildArgs(void *argsPtr, void *configPtr) = 0;

    GemmOperationDescription description_;
    OperatorArguments args_{};
    Operator op_;
};

/********************* basic matmul *********************/
template <typename Operator_>
class BasicMatmulGemmOperation : public GemmOperationBase<Operator_> {
public:
    BasicMatmulGemmOperation(char const *name = "") : GemmOperationBase<Operator_>(name)
    {
        this->description_.gemmKind = GemmKind::BasicMatmul;
    }

    virtual Status Run(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr) override
    {
        (void)fftsAddr;
        return this->op_.Run(stream, blockDim, 0);
    }

private:
    virtual void BuildArgs(void *argsPtr, void *configPtr) override
    {
        BasicMatmulGemmArguments *arguments = (BasicMatmulGemmArguments *)argsPtr;
        BasicMatmulGemmConfiguration *config = (BasicMatmulGemmConfiguration *)configPtr;
        this->args_.problemShape = GemmCoord{config->m, config->n, config->k};
        this->args_.ptrA = arguments->A;
        this->args_.ptrB = arguments->B;
        this->args_.ptrC = arguments->C;
    }
};
/********************* basic matmul end *********************/

/********************* grouped matmul *********************/
template <typename Operator_>
class GroupedMatmulGemmOperation : public GemmOperationBase<Operator_> {
public:
    GroupedMatmulGemmOperation(char const *name = "") : GemmOperationBase<Operator_>(name)
    {
        this->description_.gemmKind = GemmKind::GroupedMatmul;
    }

private:
    virtual void BuildArgs(void *argsPtr, void *configPtr) override
    {
        GroupedMatmulGemmArguments *arguments = (GroupedMatmulGemmArguments *)argsPtr;
        GroupedMatmulGemmConfiguration *config = (GroupedMatmulGemmConfiguration *)configPtr;

        this->args_.problemCount = config->groupCount;
        this->args_.ptrProblemShape = arguments->problemShapeList;
        this->args_.ptrA = arguments->A;
        this->args_.ptrLayoutA = arguments->layoutAList;
        this->args_.ptrB = arguments->B;
        this->args_.ptrLayoutB = arguments->layoutBList;
        this->args_.ptrC = arguments->C;
        this->args_.ptrLayoutC = arguments->layoutCList;
    }

    virtual Status Run(
        aclrtStream stream,
        uint32_t blockDim,
        uint64_t fftsAddr
    ) override
    {
        (void)fftsAddr;
        return this->op_.Run(stream, blockDim, 0);
    }
};
/********************* grouped matmul end *********************/

/********************* grouped matmul slice M *********************/
template <typename Operator_>
class GroupedMatmulSliceMGemmOperation : public GemmOperationBase<Operator_> {
public:
    GroupedMatmulSliceMGemmOperation(char const *name = "") : GemmOperationBase<Operator_>(name)
    {
        this->description_.gemmKind = GemmKind::GroupedMatmulSliceM;
    }

private:
    virtual void BuildArgs(void *argsPtr, void *configPtr) override
    {
        GroupedMatmulSliceMGemmArguments *arguments = (GroupedMatmulSliceMGemmArguments *)argsPtr;
        GroupedMatmulSliceMGemmConfiguration *config = (GroupedMatmulSliceMGemmConfiguration *)configPtr;

        this->args_.problemShape = GemmCoord{config->m, config->n, config->k};
        this->args_.problemCount = config->groupCount;
        this->args_.ptrGroupList = arguments->deviceGroupList;
        this->args_.ptrA = arguments->A;
        this->args_.ptrB = arguments->B;
        this->args_.ptrC = arguments->C;
    }

    virtual Status Run(
        aclrtStream stream,
        uint32_t blockDim,
        uint64_t fftsAddr
    ) override
    {
        (void)fftsAddr;
        return this->op_.Run(stream, blockDim, 0);
    }
};
/********************* grouped matmul slice M end *********************/

/********************* optimized matmul *********************/
template <typename Operator_>
class OptimizedMatmulGemmOperation : public GemmOperationBase<Operator_> {
    using Operator = Operator_;
    using OperatorKernel = typename Operator::Kernel;
    using ElementA = typename OperatorKernel::ElementA;
    using LayoutWA = typename OperatorKernel::LayoutWA;
    using LayoutWB = typename OperatorKernel::LayoutWB;
    using L1TileShape = typename OperatorKernel::BlockMmad::L1TileShape;
public:
    OptimizedMatmulGemmOperation(char const *name = "") : GemmOperationBase<Operator_>(name)
    {
        this->description_.gemmKind = GemmKind::OptimizedMatmul;
    }

private:
    virtual void BuildArgs(void *argsPtr, void *configPtr) override
    {
        BasicMatmulGemmArguments *arguments = (BasicMatmulGemmArguments *)argsPtr;
        BasicMatmulGemmConfiguration *config = (BasicMatmulGemmConfiguration *)configPtr;

        constexpr uint32_t alignByByte = 512;

        this->args_.problemShape = GemmCoord{config->m, config->n, config->k};
        this->args_.align = alignByByte / sizeof(ElementA);
        this->args_.elementSize = sizeof(float);
        this->args_.ptrA = arguments->A;
        this->args_.ptrB = arguments->B;
        this->args_.ptrC = arguments->C;

        // extract padding configuration from OperatorKernel::LayoutWA/B
        if constexpr (std::is_same<LayoutWA, Catlass::layout::PaddingRowMajor>::value ||
            std::is_same<LayoutWA, Catlass::layout::PaddingColumnMajor>::value) {

            // construct layoutWA in Kernel::Arguments with the passed template kernel
            this->args_.layoutWA = LayoutWA(config->m, config->k, L1TileShape::M, L1TileShape::K);

            // save the padding configuration that will be used in CanImplement()
            isPaddingA_ = true;
            using LayoutA = typename OperatorKernel::LayoutA;
            isNeedPaddingA_ = IsNeedPadding(LayoutA(config->m, config->k), this->args_.align);
        } else {
            this->args_.layoutWA = LayoutWA(config->m, config->k);
            isNeedPaddingA_ = IsNeedPadding(this->args_.layoutWA, this->args_.align);
        }

        // do the same for layoutWB
        if constexpr (std::is_same<LayoutWB, Catlass::layout::PaddingRowMajor>::value ||
            std::is_same<LayoutWB, Catlass::layout::PaddingColumnMajor>::value) {
            this->args_.layoutWB = LayoutWB(config->k, config->n, L1TileShape::K, L1TileShape::N);
            isPaddingB_ = true;
            using LayoutB = typename OperatorKernel::LayoutB;
            isNeedPaddingB_ = IsNeedPadding(LayoutB(config->k, config->n), this->args_.align);
        } else {
            this->args_.layoutWB = LayoutWB(config->k, config->n);
            isNeedPaddingB_ = IsNeedPadding(this->args_.layoutWB, this->args_.align);
        }
    }

    virtual Status CanImplement(void *argsPtr, void *configPtr) override
    {
        // check the operator's padding configuration is correct under the given problem shape
        BuildArgs(argsPtr, configPtr);

        if ((isPaddingA_ != isNeedPaddingA_) || (isPaddingB_ != isNeedPaddingB_)) {
            return Catlass::Status::kInvalid;
        }

        return this->op_.CanImplement(this->args_);
    }

    bool IsNeedPadding(Catlass::layout::RowMajor layout, uint32_t align)
    {
        // If the stride is greater than 65536, padding is required to reduce the stride.
        if (layout.stride(0) < 65536) {
            return layout.stride(0) % align != 0;
        } else {
            return true;
        }
    }

    bool IsNeedPadding(Catlass::layout::ColumnMajor layout, uint32_t align)
    {
        // If the stride is greater than 65536, padding is required to reduce the stride.
        if (layout.stride(1) < 65536) {
            return layout.stride(1) % align != 0;
        } else {
            return true;
        }
    }

    bool isPaddingA_{false};
    bool isPaddingB_{false};
    bool isNeedPaddingA_{false};
    bool isNeedPaddingB_{false};
};
/********************* optimized matmul end *********************/

}
}

#endif // CATLASS_LIBRARY_GEMM_OPERATION_H