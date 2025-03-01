#ifndef ACOT_GEMM_BLOCK_BLOCK_GEMM_HPP
#define ACOT_GEMM_BLOCK_BLOCK_GEMM_HPP

#include "acot/acot.hpp"
#include "acot/gemm/helper.hpp"
#include "acot/gemm/tile/tile_copy.hpp"
#include "acot/gemm/tile/tile_mmad.hpp"
#include "acot/matmul_coord.hpp"

namespace acot::gemm::block{
template<
    class DispatchPolicy_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_ = void, 
    class TileCopy_ = acot::gemm::tile::TileCopy<typename DispatchPolicy_::ArchTag, AType_, BType_, CType_, BiasType_>,
    class TileMmad_ = acot::gemm::tile::TileMmad<typename DispatchPolicy_::ArchTag, AType_, BType_, CType_, BiasType_>
>
struct BlockGemm{
    static_assert(DEPENDENT_FALSE<DispatchPolicy_>, "BlockMmad is not implemented for this DispatchPolicy");
};

}

#include "acot/gemm/block/block_gemm_pingpong.hpp"
#include "acot/gemm/block/block_gemm_PP_PL.hpp"

#endif // ACOT_GEMM_BLOCK_BLOCK_GEMM_HPP