#ifndef ACOT_GEMM_DISPATCH_POLICY_HPP
#define ACOT_GEMM_DISPATCH_POLICY_HPP

#include "acot/acot.hpp"

namespace acot::gemm {
// 单独开启双缓冲的控制
template <bool ENABLE_UNIT_FLAG_ = false>
struct GemmAscendC910B3Pingpong{
    using ArchTag = arch::AscendC910B3;
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};
}

#endif // ACOT_GEMM_DISPATCH_POLICY_HPP