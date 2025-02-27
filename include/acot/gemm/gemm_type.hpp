#ifndef ACOT_GEMM_GEMM_TYPE_HPP
#define ACOT_GEMM_GEMM_TYPE_HPP

namespace acot::gemm{

// 完全看齐基本数据结构
template <class Element_, class Layout_, AscendC::TPosition POSITION_ = AscendC::TPosition::GM>
struct GemmType {
    using Element = Element_;
    using Layout = Layout_;
    static constexpr AscendC::TPosition POSITION = POSITION_;
};
}

#endif // ACOT_GEMM_GEMM_TYPE_HPP