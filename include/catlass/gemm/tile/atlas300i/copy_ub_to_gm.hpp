/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ATLAS300I_COPY_UB_TO_GM_HPP
#define CATLASS_GEMM_TILE_ATLAS300I_COPY_UB_TO_GM_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Gemm::Tile {

constexpr uint32_t STRIDE_LIMIT = 65536;

template <class ArchTag, class GmType>
struct CopyUb2Gm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm, can not find the specialization.");
};

template <class ArchTag, typename Element>
struct CopyUb2Gm<ArchTag, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::zN;

    CATLASS_DEVICE
    CopyUb2Gm() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        layout::RowMajor const &layoutDst,
        layout::zN const &layoutSrc
    )
    {
        if (layoutDst.shape(0) == layoutSrc.orgShape(0) && layoutDst.shape(1) == layoutSrc.orgShape(1)) {
            AscendC::Nz2NdParamsFull intriParams;
            intriParams.ndNum = 1;
            intriParams.nValue = layoutSrc.orgShape(0);
            intriParams.dValue = layoutSrc.orgShape(1);
            intriParams.srcNStride = layoutSrc.orgShape(0);
            intriParams.dstDStride = layoutDst.stride(0);
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else if (layoutDst.shape(1) % C0_NUM_PER_FRACTAL == 0) {
            for (int i = 0; i < layoutDst.shape(0); i++) {
                AscendC::DataCopyParams params;
                params.blockCount = layoutDst.shape(1) / C0_NUM_PER_FRACTAL;
                params.blockLen = C0_NUM_PER_FRACTAL * sizeof(Element) / BYTE_PER_BLK;
                params.srcGap = (layoutSrc.orgShape(0) - 1) * C0_NUM_PER_FRACTAL * sizeof(Element) / BYTE_PER_BLK;
                params.dstGap = 0;
                AscendC::DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * C0_NUM_PER_FRACTAL], params);
            }
        } else {
            if (layoutDst.shape(1) / C0_NUM_PER_FRACTAL) {
                for (int i = 0; i < layoutDst.shape(0); i++) {
                    AscendC::DataCopyParams params;
                    params.blockCount = layoutDst.shape(1) / C0_NUM_PER_FRACTAL;
                    params.blockLen = C0_NUM_PER_FRACTAL * sizeof(Element) / BYTE_PER_BLK;
                    params.srcGap = (layoutSrc.orgShape(0) - 1) * C0_NUM_PER_FRACTAL * sizeof(Element) / BYTE_PER_BLK;
                    params.dstGap = 0;
                    AscendC::DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * C0_NUM_PER_FRACTAL], params);
                }
            }

            for (int i = 0; i < layoutDst.shape(0); i++) {

                uint64_t mask0 = (1ul << C0_NUM_PER_FRACTAL) - (1ul << (layoutDst.shape(1) % C0_NUM_PER_FRACTAL));
                if (SizeOfBits<Element>::value == 16) {
                    uint64_t mask[2] = {mask0, 0};
                    AscendC::Duplicate<Element>(
                        srcTensor
                            [layoutSrc.orgShape(0) * (layoutDst.shape(1) / C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL)
                             + i * C0_NUM_PER_FRACTAL],
                        0, mask, 1, 1, 1
                    );
                } else {
                    uint64_t mask[1] = {mask0};
                    AscendC::Duplicate<Element>(
                        srcTensor
                            [layoutSrc.orgShape(0) * (layoutDst.shape(1) / C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL)
                             + i * C0_NUM_PER_FRACTAL],
                        0, mask, 1, 1, 1
                    );
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::DataCopyParams params;
                params.blockCount = 1;
                params.blockLen = C0_NUM_PER_FRACTAL * sizeof(Element) / BYTE_PER_BLK;
                params.srcGap = 0;
                params.dstGap = 0;
                AscendC::SetAtomicAdd<Element>();
                AscendC::DataCopy(
                    dstTensor[i * layoutDst.stride(0) + (layoutDst.shape(1) / C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL)],
                    srcTensor
                        [layoutSrc.orgShape(0) * (layoutDst.shape(1) / C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL)
                         + i * C0_NUM_PER_FRACTAL],
                    params
                );
                AscendC::SetAtomicNone();
            }
        }
    }
};

template <class ArchTag, typename Element>
struct CopyUb2Gm<ArchTag, Gemm::GemmType<Element, layout::zN>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    CATLASS_DEVICE
    CopyUb2Gm() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        layout::zN const &layoutDst,
        layout::zN const &layoutSrc,
        layout::zN const &layoutC
    )
    {
        uint32_t blockCount = min(layoutSrc.orgShape(1), layoutDst.orgShape(1)) / C0_NUM_PER_FRACTAL;
        uint16_t blockLen = min(layoutSrc.orgShape(0), layoutDst.orgShape(0)) * C0_NUM_PER_FRACTAL * sizeof(Element)
                            / BYTE_PER_BLK;
        uint16_t srcGap = (layoutSrc.orgShape(0) - layoutDst.orgShape(0)) * C0_NUM_PER_FRACTAL * sizeof(Element)
                          / BYTE_PER_BLK;
        uint32_t dstGap = (layoutC.orgShape(0) - min(layoutDst.orgShape(0), layoutSrc.orgShape(0))) * C0_NUM_PER_FRACTAL
                          * sizeof(Element) / BYTE_PER_BLK;
        AscendC::DataCopyParams params;
        if (dstGap < STRIDE_LIMIT) {
            params.blockCount = blockCount;
            params.blockLen = blockLen;
            params.srcGap = srcGap;
            params.dstGap = dstGap;
            AscendC::DataCopy(dstTensor, srcTensor, params);
        } else {
            for (uint32_t i = 0; i < blockCount; i++) {
                params.blockCount = 1;
                params.blockLen = blockLen;
                params.srcGap = 0;
                params.dstGap = 0;
                AscendC::DataCopy(
                    dstTensor[i * layoutC.orgShape(0) * C0_NUM_PER_FRACTAL],
                    srcTensor[i * layoutSrc.orgShape(0) * C0_NUM_PER_FRACTAL], params
                );
            }
        }
    }
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_UB_TO_GM_HPP