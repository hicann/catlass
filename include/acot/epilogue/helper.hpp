#ifndef ACOT_EPILOGUE_HELPER_HPP
#define ACOT_EPILOGUE_HELPER_HPP

#include "acot/acot.hpp"
#include "acot/layout/layout.hpp"

namespace acot::epilogue::helper{
// 正常的模板特例化过程
template<class ElementA, class ElementB>
struct ElementAccumulatorSelector {};

template<>
struct ElementAccumulatorSelector<half, half> {
    using ElementAccumulator = float;
};

template<>
struct ElementAccumulatorSelector<int8_t, int8_t> {
    using ElementAccumulator = half;
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

#endif // ACOT_EPILOGUE_HELPER_HPP