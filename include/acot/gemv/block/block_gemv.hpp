#ifndef ACOT_GEMV_BLOCK_BLOCK_GEMV_HPP
#define ACOT_GEMV_BLOCK_BLOCK_GEMV_HPP

#include "acot/acot.hpp"
#include "acot/gemv/tile/tile_copy.hpp"
#include "acot/gemv/tile/tile_mmad.hpp"

namespace acot::gemv::block
{

    template <
        class DispatchPolicy,
        class L1TileShape,
        class L0TileShape,
        class xType,
        class AType,
        class yType,
        class BiasType = void,
        class TileCopy = gemv::tile::TileCopy<typename DispatchPolicy::ArchTag, xType, AType, yType, BiasType>, //
        class TileMmad = gemv::tile::TileMmad<typename DispatchPolicy::ArchTag, xType, AType, BiasType>>        //
    struct BlockGemv
    {
        static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockGemv is not implemented for this DispatchPolicy");
    };
} // namespace acot::gemv::block

#include "acot/gemv/block/block_gemv_pingpong_PL.hpp"
#include "acot/gemv/block/block_swizzle.hpp"

#endif // ACOT_GEMV_BLOCK_BLOCK_GEMV_HPP