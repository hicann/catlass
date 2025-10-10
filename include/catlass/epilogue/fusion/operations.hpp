#ifndef CATLASS_EPILOGUE_FUSION_OPERATIONS_HPP
#define CATLASS_EPILOGUE_FUSION_OPERATIONS_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Fusion {

// 类型转换操作符（3参数版本）
template <
    typename T,
    typename S,
    AscendC::RoundMode RoundMode = AscendC::RoundMode::CAST_NONE
>
struct NumericArrayConverter {
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        AscendC::LocalTensor<S> const& src,
        uint32_t compute_length
    ) const {
        if constexpr (std::is_same_v<T, S>) {
            AscendC::DataCopy(dst, src, compute_length);
        } else {
            AscendC::Cast(dst, src, RoundMode, compute_length);
        }
    }
};

// 加法操作符
template <typename T>
struct Plus {
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        uint32_t compute_length
    ) const {
        AscendC::Add(dst, src0, src1, compute_length);
    }
};

} // namespace Catlass::Epilogue::Fusion

#endif
