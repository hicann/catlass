#ifndef ACOT_GEMM_TILE_TILE_COPY_HPP
#define ACOT_GEMM_TILE_TILE_COPY_HPP

#include "acot/gemm/tile/copy_gm_to_l1.hpp"
#include "acot/gemm/tile/copy_l0c_to_gm.hpp"
#include "acot/gemm/tile/copy_l1_to_l0.hpp"
#include "acot/gemm/helper.hpp"

namespace acot::gemm::tile{
template<
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class BiasType = void
>
struct TileCopy{
    using ElementA = typename AType::Element;
    using LayoutA = typename AType::Layout;
    using ElementB = typename BType::Element;
    using LayoutB = typename BType::Layout;
    using ElementAccumulator =
        typename gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = gemm::tile::CopyGmToL1A<ArchTag, AType>;
    using CopyGmToL1B = gemm::tile::CopyGmToL1B<ArchTag, BType>;
    using CopyL1ToL0A = gemm::tile::CopyL1ToL0A<ArchTag, AType>;
    using CopyL1ToL0B = gemm::tile::CopyL1ToL0B<ArchTag, BType>;
    using CopyL0CToGm = gemm::tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
};
}

#endif // ACOT_GEMM_TILE_TILE_COPY_HPP