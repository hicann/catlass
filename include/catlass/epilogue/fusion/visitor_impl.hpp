#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_IMPL_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_IMPL_HPP

#include "catlass/epilogue/fusion/visitor_impl_base.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/int_tuple.hpp"

namespace Catlass::Epilogue::Fusion {

// 空回调（用于叶子节点）
struct EmptyCallbacks {
    CATLASS_DEVICE void begin_epilogue() {}
    CATLASS_DEVICE void end_epilogue() {}
    CATLASS_DEVICE void begin_row(int) {}
    CATLASS_DEVICE void end_row(int) {}
};

template <class... Ops>
struct VisitorImpl : VisitorImplBase<Ops...> {
    using VisitorImplBase<Ops...>::VisitorImplBase;
    using VisitorImplBase<Ops...>::ops;

    template <class CallbacksTuple>
    struct Callbacks {
        CallbacksTuple callbacks_tuple;

        CATLASS_DEVICE
        Callbacks(CallbacksTuple&& cbs) : callbacks_tuple(static_cast<CallbacksTuple&&>(cbs)) {}

        CATLASS_DEVICE void begin_epilogue() {
            tla::for_each(callbacks_tuple, [](auto& cb) { cb.begin_epilogue(); });
        }

        CATLASS_DEVICE void end_epilogue() {
            tla::for_each(callbacks_tuple, [](auto& cb) { cb.end_epilogue(); });
        }

        CATLASS_DEVICE void begin_row(int row_idx) {
            tla::for_each(callbacks_tuple, [&](auto& cb) { cb.begin_row(row_idx); });
        }

        CATLASS_DEVICE void end_row(int row_idx) {
            tla::for_each(callbacks_tuple, [&](auto& cb) { cb.end_row(row_idx); });
        }
    };

    template <class ArchTag, int... Is>
    CATLASS_DEVICE auto get_callbacks_impl(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length,
        GemmCoord const& blockShapeMNK,
        GemmCoord const& blockCoordMNK,
        MatrixCoord const& subblockShape,
        MatrixCoord const& subblockCoord,
        AscendC::GlobalTensor<half> const& gmSubblockC,
        layout::RowMajor const& layoutSubblockC,
        tla::seq<Is...>
    ) {
        auto tuple_cbs = tla::tuple<
            decltype(tla::get<Is>(ops).get_callbacks(resource, ub_offset, compute_length,
                                                     blockShapeMNK, blockCoordMNK,
                                                     subblockShape, subblockCoord,
                                                     gmSubblockC, layoutSubblockC))...
        >(
            tla::get<Is>(ops).get_callbacks(resource, ub_offset, compute_length,
                                           blockShapeMNK, blockCoordMNK,
                                           subblockShape, subblockCoord,
                                           gmSubblockC, layoutSubblockC)...
        );
        return Callbacks<decltype(tuple_cbs)>(static_cast<decltype(tuple_cbs)&&>(tuple_cbs));
    }

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length,
        GemmCoord const& blockShapeMNK,
        GemmCoord const& blockCoordMNK,
        MatrixCoord const& subblockShape,
        MatrixCoord const& subblockCoord,
        AscendC::GlobalTensor<half> const& gmSubblockC,
        layout::RowMajor const& layoutSubblockC
    ) {
        return get_callbacks_impl(
            resource, ub_offset, compute_length,
            blockShapeMNK, blockCoordMNK,
            subblockShape, subblockCoord,
            gmSubblockC, layoutSubblockC,
            tla::make_seq<sizeof...(Ops)>{}
        );
    }
};

} // namespace Catlass::Epilogue::Fusion

#endif
