/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR dataA PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXAMPLES_COMMON_GOLDEN_MATMUL_HPP
#define EXAMPLES_COMMON_GOLDEN_MATMUL_HPP

#include <vector>

#include "AscendCT/layout/layout.hpp"

namespace AscendCT::golden {

// simple matmul
template<class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeMatmul(
    const GemmCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementB> &dataB, const LayoutB &layoutB,
    std::vector<ElementGolden> &dataGolden, const LayoutGolden &layoutGolden
)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            ElementGolden accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            dataGolden[offsetGolden] = static_cast<ElementGolden>(accumulator);
        }
    }
}

// simple batched matmul
template<class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeBatchedMatmul(
    const uint32_t batchedCount, const GemmCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementB> &dataB, const LayoutB &layoutB,
    std::vector<ElementGolden> &dataC, const LayoutGolden &layoutGolden
)
{
    for (uint32_t batchId = 0; batchId < batchedCount; ++batchId) {
        size_t batchOffsetA = static_cast<size_t>(problemShape.m()) * problemShape.k() * batchId;
        size_t batchOffsetB = static_cast<size_t>(problemShape.k()) * problemShape.n() * batchId;
        size_t batchoffsetGolden = static_cast<size_t>(problemShape.m()) * problemShape.n() * batchId;
        for (uint32_t i = 0; i < problemShape.m(); ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j)) + batchoffsetGolden;
                ElementGolden accumulator = 0;
                for (uint32_t k = 0; k < problemShape.k(); ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, k)) + batchOffsetA;
                    size_t offsetB = layoutB.GetOffset(MakeCoord(k, j)) + batchOffsetB;
                    accumulator += static_cast<ElementGolden>(dataA[offsetA]) *
                        static_cast<ElementGolden>(dataB[offsetB]);
                }
                dataC[offsetGolden] = static_cast<ElementGolden>(accumulator);
            }
        }
    }
}

// simple grouped matmul
template<class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeGroupedMatmul(
    uint32_t problemCount,
    const std::vector<GemmCoord> &problemShapeList,
    const std::vector<ElementA> &dataA, const std::vector<LayoutA> &layoutAList,
    const std::vector<ElementB> &dataB, const std::vector<LayoutB> &layoutBList,
    std::vector<ElementGolden> &dataGolden, const std::vector<LayoutGolden> &layoutGoldenList
)
{
    size_t inGroupOffsetA = 0;
    size_t inGroupOffsetB = 0;
    size_t inGroupOffsetGolden = 0;
    for (uint32_t inGroupId = 0; inGroupId < problemCount; ++inGroupId) {
        GemmCoord problemShape = problemShapeList[inGroupId];
        for (uint32_t i = 0; i < problemShape.m(); ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = inGroupOffsetGolden + layoutGoldenList[inGroupId].GetOffset(MakeCoord(i, j));
                ElementGolden accumulator = 0;
                for (uint32_t k = 0; k < problemShape.k(); ++k) {
                    size_t offsetA = inGroupOffsetA + layoutAList[inGroupId].GetOffset(MakeCoord(i, k));
                    size_t offsetB = inGroupOffsetB + layoutBList[inGroupId].GetOffset(MakeCoord(k, j));
                    accumulator += static_cast<ElementGolden>(dataA[offsetA]) *
                        static_cast<ElementGolden>(dataB[offsetB]);
                }
                dataGolden[offsetGolden] = static_cast<ElementGolden>(accumulator);
            }
        }
        inGroupOffsetA += static_cast<size_t>(problemShape.m()) * problemShape.k();
        inGroupOffsetB += static_cast<size_t>(problemShape.k()) * problemShape.n();
        inGroupOffsetGolden += static_cast<size_t>(problemShape.m()) * problemShape.n();
    }
}

// matmul add
template<
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementX,  // Layout X must be same as LayoutGolden
    class ElementGolden, class LayoutGolden
>
void ComputeMatmulElemWiseAdd(
    const GemmCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementB> &dataB, const LayoutB &layoutB,
    const std::vector<ElementX> &dataX,  // layoutX must be same as layoutGolden
    std::vector<ElementGolden> &dataGolden, const LayoutGolden &layoutGolden
)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            ElementGolden accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            dataGolden[offsetGolden] = accumulator + static_cast<ElementGolden>(dataX[offsetGolden]);
        }
    }
}

template <
    class ElementGroupList, class ElementScale,
    class LayoutB, class LayoutScale, class LayoutPerTokenScale
>
void ComputeGroupedMatmulPerTokenDequant(
    const GemmCoord &problemShape, uint32_t problemCount, const std::vector<ElementGroupList> &groupList,
    const std::vector<int8_t> &dataA, const layout::RowMajor &layoutA,
    const std::vector<int8_t> &dataB, const LayoutB &layoutB,
    const std::vector<ElementScale> &dataScale, const LayoutScale &,
    const std::vector<ElementScale> &dataPerTokenScale, const LayoutPerTokenScale &,
    std::vector<float> &dataGolden, const layout::RowMajor &layoutGolden
)
{
    size_t groupOffsetB = 0;
    size_t groupOffsetScale = 0;
    uint32_t startRow = 0;
    for (uint32_t inGroupId = 0; inGroupId < problemCount; ++inGroupId) {
        for (uint32_t i = startRow; i < groupList[inGroupId]; ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
                int32_t accumulator = 0;
                for (uint32_t k = 0; k < problemShape.k(); ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                    size_t offsetB = groupOffsetB + layoutB.GetOffset(MakeCoord(k, j));
                    accumulator += static_cast<int32_t>(dataA[offsetA]) * static_cast<int32_t>(dataB[offsetB]);
                }
                dataGolden[offsetGolden] = static_cast<float>(accumulator) *
                    static_cast<float>(dataScale[groupOffsetScale + j]) *
                    static_cast<float>(dataPerTokenScale[i]);
            }
        }
        groupOffsetB += static_cast<size_t>(problemShape.k()) * problemShape.n();
        groupOffsetScale += static_cast<size_t>(problemShape.n());
        startRow = groupList[inGroupId];
    }
}

template <
    class LayoutA,
    class LayoutB,
    class ElementScale
>
void QuantMatmul(
    const GemmCoord &problemShape,
    const std::vector<int8_t> &dataA, const LayoutA &layoutA,
    const std::vector<int8_t> &dataB, const LayoutB &layoutB,
    const std::vector<ElementScale> &dataScale, const layout::VectorLayout &layoutScale,
    const std::vector<ElementScale> &dataPerTokenScale, const layout::VectorLayout &layoutPerTokenScale,
    std::vector<float> &dataGolden, const layout::RowMajor &layoutGolden
)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            int32_t accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<int32_t>(dataA[offsetA]) * static_cast<int32_t>(dataB[offsetB]);
            }
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            dataGolden[offsetGolden] = static_cast<float>(accumulator) *
                static_cast<float>(dataScale[j]) *
                static_cast<float>(dataPerTokenScale[i]);
        }
    }
}

template <
    class ElementGroupList, class ElementScale, class LayoutScale, class LayoutPerTokenScale
>
void ComputeGroupedMatmulKPerTokenDequant(
    const GemmCoord &problemShape, uint32_t problemCount, const std::vector<ElementGroupList> &groupList,
    const std::vector<int8_t> &dataA, const layout::ColumnMajor &layoutA,
    const std::vector<int8_t> &dataB, const layout::RowMajor &layoutB,
    const std::vector<ElementScale> &dataScale, const LayoutScale &,
    const std::vector<ElementScale> &dataPerTokenScale, const LayoutPerTokenScale &,
    std::vector<float> &dataGolden, const layout::RowMajor &layoutGolden
)
{
    size_t groupOffsetD = 0;
    size_t groupOffsetScale = 0;
    size_t groupOffsetPerTokenScale = 0;
    uint32_t startRow = 0;
    for (uint32_t inGroupId = 0; inGroupId < problemCount; ++inGroupId) {
        for (uint32_t i = 0; i < problemShape.m(); ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
                int32_t accumulator = 0;
                for (uint32_t k = startRow; k < groupList[inGroupId]; ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                    size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                    accumulator += static_cast<int32_t>(dataA[offsetA]) * static_cast<int32_t>(dataB[offsetB]);
                }
                dataGolden[groupOffsetD+offsetGolden] = static_cast<float>(accumulator) *
                    static_cast<float>(dataScale[groupOffsetScale + j]) *
                    static_cast<float>(dataPerTokenScale[groupOffsetPerTokenScale + i]);
            }
        }

        groupOffsetD += static_cast<size_t>(problemShape.m()) * problemShape.n();
        groupOffsetScale += static_cast<size_t>(problemShape.n());
        groupOffsetPerTokenScale += static_cast<size_t>(problemShape.m());
        startRow = groupList[inGroupId];
    }
}

} // namespace AscendCT::golden

#endif // EXAMPLES_COMMON_GOLDEN_MATMUL_HPP
