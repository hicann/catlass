/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_MATMUL_TILE_TILE_COPY_HPP
#define ACOT_MATMUL_TILE_TILE_COPY_HPP

#include "acot/acot.hpp"
#include "acot/detail/layout.hpp"

namespace acot::matmul::tile {

template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class Enable = void
>
struct TileCopyV2 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyV2, can not find the specialization.");
};

template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class LayoutTagSrc,
    class LayoutTagDst
>
struct TileCopyExt {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyExt, can not find the specialization.");
};

} // namespace acot::matmul::tile

#include "acot/matmul/tile/copy_gm_to_l1.hpp"
#include "acot/matmul/tile/copy_l0c_to_gm.hpp"
#include "acot/matmul/tile/copy_l1_to_l0a.hpp"
#include "acot/matmul/tile/copy_l1_to_l0b.hpp"
#include "acot/matmul/tile/copy_gm_to_ub.hpp"
#include "acot/matmul/tile/copy_ub_to_gm.hpp"
#include "acot/matmul/helper.hpp"


namespace acot::matmul::tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for B matrix operand
    class BType,
    /// MatmulType type for C matrix operand
    class CType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopy {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename matmul::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = matmul::tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = matmul::tile::CopyGmToL1<ArchTag, BType>;
    using CopyL1ToL0A = matmul::tile::CopyL1ToL0A<
        ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = matmul::tile::CopyL1ToL0B<
        ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = matmul::tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
};

template <
    /// Tag indicating architecture
    class ArchTag,
    class TensorA,
    class LayoutTagA,
    class TensorB,
    class LayoutTagB,
    class TensorC,
    class LayoutTagC,
    class TensorBias = void,
    class LayoutTagBias = void
>
struct PackedTileCopyV2 {
    using ElementA = typename TensorA::Element;
    using ElementB = typename TensorB::Element;
    using ElementAccumulator =
        typename matmul::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using LayoutL1A = detail::TagToLayout_t<ElementA,
        typename helper::L1ATypeSelector<matmul::MatmulType<ElementA, LayoutTagA>>::L1AType::Layout>;
    using LayoutL1B = detail::TagToLayout_t<ElementB,
        typename helper::L1BTypeSelector<matmul::MatmulType<ElementB, LayoutTagB>>::L1BType::Layout>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, layout::zZ>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, layout::nZ>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL1A = Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, AscendC::TPosition::A1>;
    using TensorL1B = Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, AscendC::TPosition::A1>;
    using TensorL0A = Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, AscendC::TPosition::A2>;
    using TensorL0B = Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, AscendC::TPosition::B2>;
    using TensorL0C = Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, AscendC::TPosition::CO1>;

    using L1AAlignHelper = matmul::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = matmul::helper::L1AlignHelper<ElementB, LayoutTagB>;

    using CopyGmToL1A = matmul::tile::TileCopyV2<ArchTag, TensorA, TensorL1A>;
    using CopyGmToL1B = matmul::tile::TileCopyV2<ArchTag, TensorB, TensorL1B>;
    using CopyL1ToL0A = matmul::tile::TileCopyV2<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = matmul::tile::TileCopyV2<ArchTag, TensorL1B, TensorL0B>;
    using CopyL0CToGm = matmul::tile::CopyL0CToGmV2<ArchTag, TensorL0C, TensorC>;
};

template <
    /// Tag indicating architecture
    class ArchTag,
    class TensorA,
    class LayoutTagA,
    class TensorB,
    class LayoutTagB,
    class TensorC,
    class LayoutTagC,
    class TensorBias = void,
    class LayoutTagBias = void,
    bool IS_PADDING_A = false,
    bool IS_PADDING_B = false
>
struct PaddingPackedTileCopy {
    static_assert(std::is_same_v<LayoutTagA, layout::RowMajor> || std::is_same_v<LayoutTagA, layout::ColumnMajor>,
        "Unsupported layout, only can be RowMajor and ColumnMajor");
    static_assert(std::is_same_v<LayoutTagB, layout::RowMajor> || std::is_same_v<LayoutTagB, layout::ColumnMajor>,
        "Unsupported layout, only can be RowMajor and ColumnMajor");
    using ElementA = typename TensorA::Element;
    using ElementB = typename TensorB::Element;
    using ElementAccumulator =
        typename matmul::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using LayoutTagL1A = typename helper::L1ATypeSelector<matmul::MatmulType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename helper::L1BTypeSelector<matmul::MatmulType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, layout::zZ>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, layout::nZ>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL1A = Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, AscendC::TPosition::A1>;
    using TensorL1B = Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, AscendC::TPosition::A1>;
    using TensorL0A = Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, AscendC::TPosition::A2>;
    using TensorL0B = Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, AscendC::TPosition::B2>;
    using TensorL0C = Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, AscendC::TPosition::CO1>;

    using L1AAlignHelper = matmul::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = matmul::helper::L1AlignHelper<ElementB, LayoutTagB>;

    using LayoutPaddingTagA = std::conditional_t<std::is_same_v<LayoutTagA, layout::RowMajor>, 
        layout::PaddingRowMajor, layout::PaddingColumnMajor>;
    using LayoutPaddingTagB = std::conditional_t<std::is_same_v<LayoutTagB, layout::RowMajor>,
        layout::PaddingRowMajor, layout::PaddingColumnMajor>;

    using CopyGmToL1A = std::conditional_t<
        IS_PADDING_A,
        matmul::tile::TileCopyExt<ArchTag, TensorA, TensorL1A, LayoutPaddingTagA, LayoutTagL1A>,
        matmul::tile::TileCopyV2<ArchTag, TensorA, TensorL1A>
    >;
    using CopyGmToL1B = std::conditional_t<
        IS_PADDING_B,
        matmul::tile::TileCopyExt<ArchTag, TensorB, TensorL1B, LayoutPaddingTagB, LayoutTagL1B>,
        matmul::tile::TileCopyV2<ArchTag, TensorB, TensorL1B>
    >;

    using CopyL1ToL0A = matmul::tile::TileCopyV2<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = matmul::tile::TileCopyV2<ArchTag, TensorL1B, TensorL0B>;
    using CopyL0CToGm = matmul::tile::CopyL0CToGmV2<ArchTag, TensorL0C, TensorC>;
};

} // namespace acot::matmul::tile

#endif // ACOT_MATMUL_TILE_TILE_COPY_HPP
