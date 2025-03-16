#ifndef ACOT_GEMM_HELPER_HPP
#define ACOT_GEMM_HELPER_HPP

#include "acot/acot.hpp"
#include "acot/layout/layout.hpp"

namespace acot::gemm::helper{
template<class ElementA, class ElementB>
struct ElementAccumulatorSelector {};

template<>
struct ElementAccumulatorSelector<half, half> {
    using ElementAccumulator = float;
};

template<>
struct ElementAccumulatorSelector<int8_t, int8_t> {
    using ElementAccumulator = int32_t;
};

template<>
struct ElementAccumulatorSelector<int32_t, int32_t> {
    using ElementAccumulator = int32_t;
};

template<>
struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
    using ElementAccumulator = float;
};

template<>
struct ElementAccumulatorSelector<float, float> {
    using ElementAccumulator = float;
};
}

#endif // ACOT_GEMM_HELPER_HPP