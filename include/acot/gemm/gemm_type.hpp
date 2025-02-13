#ifndef ACOT_GEMM_GEMM_TYPE_HPP
#define ACOT_GEMM_GEMM_TYPE_HPP

namespace acot::gemm{

// 只考虑数据类型加排布方式
template <class Element_, class Layout_>
struct GemmType {
    using Element = Element_;
    using Layout = Layout_;
};
}

#endif // ACOT_GEMM_GEMM_TYPE_HPP