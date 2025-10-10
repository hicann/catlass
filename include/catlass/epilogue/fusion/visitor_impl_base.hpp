#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_IMPL_BASE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_IMPL_BASE_HPP

#include "catlass/catlass.hpp"
#include "catlass/status.hpp"
#include "tla/int_tuple.hpp"

namespace Catlass::Epilogue::Fusion {

template <class... Ops>
struct VisitorImplBase {
    using Arguments = tla::tuple<typename Ops::Arguments...>;
    using Params = tla::tuple<typename Ops::Params...>;

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
        uint8_t* op_workspace = reinterpret_cast<uint8_t*>(workspace);
        return tla::transform_apply(
            tla::tuple<Ops...>{}, args,
            [&](auto&& op_tag, auto const& op_args) {
                using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                auto ret = Op::to_underlying_arguments(problem_shape, op_args, op_workspace);
                if (op_workspace != nullptr) {
                    size_t sz = Op::get_workspace_size(problem_shape, op_args);
                    op_workspace += (sz + 15) & ~15; // 16 字节对齐
                }
                return ret;
            },
            [](auto&&... op_params) -> tla::tuple<tla::remove_cvref_t<decltype(op_params)>...> { 
                return tla::tuple<tla::remove_cvref_t<decltype(op_params)>...>(op_params...); 
            }
        );
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static bool
    can_implement(ProblemShape const& problem_shape, Arguments const& args) {
        return tla::transform_apply(
            tla::tuple<Ops...>{}, args,
            [&](auto&& op_tag, auto const& op_args) {
                using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                return Op::can_implement(problem_shape, op_args);
            },
            [](auto&&... ok) { return (true && ... && ok); }
        );
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static size_t
    get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
        return tla::transform_apply(
            tla::tuple<Ops...>{}, args,
            [&](auto&& op_tag, auto const& op_args) {
                using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                size_t sz = Op::get_workspace_size(problem_shape, op_args);
                return (sz + 15) & ~15; // 16 字节对齐
            },
            [](auto&&... rounded_sizes) { return (size_t{0} + ... + rounded_sizes); }
        );
    }

    CATLASS_HOST_DEVICE
    VisitorImplBase() {}

    CATLASS_HOST_DEVICE
    VisitorImplBase(Params const& params)
        : ops(
            tla::transform_apply(
                tla::tuple<Ops...>{}, params,
                [](auto&& op_tag, auto const& op_params) {
                    using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                    return Op(op_params);
                },
                [](auto&&... built_ops) -> tla::tuple<tla::remove_cvref_t<decltype(built_ops)>...> { 
                    return tla::tuple<tla::remove_cvref_t<decltype(built_ops)>...>(built_ops...); 
                }
            )
        )
    {}

    // 使用 tla::tuple 存储
    tla::tuple<Ops...> ops;
};

} // namespace Catlass::Epilogue::Fusion

#endif
