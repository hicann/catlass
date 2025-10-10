#ifndef CATLASS_EPILOGUE_FUSION_TREE_VISITOR_HPP
#define CATLASS_EPILOGUE_FUSION_TREE_VISITOR_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"

namespace Catlass::Epilogue::Fusion {

template <class NodeOp, class... ChildOps>
struct TreeVisitor : VisitorImpl<ChildOps..., NodeOp> {
    using VisitorImpl<ChildOps..., NodeOp>::VisitorImpl;

    template<class CallbacksImpl>
    struct Callbacks : CallbacksImpl {
        CATLASS_DEVICE
        Callbacks(CallbacksImpl&& impl)
            : CallbacksImpl(static_cast<CallbacksImpl&&>(impl)) {}

        using CallbacksImpl::callbacks_tuple;

        // 辅助函数：收集子节点输出
        template <typename... Args, int... Is>
        CATLASS_DEVICE auto collect_child_outputs(
            MatrixCoord const& tileOffset,
            MatrixCoord const& tileShape,
            uint32_t calCount,
            tla::seq<Is...>,
            Args const&... args
        ) {
            return tla::tuple<decltype(tla::get<Is>(callbacks_tuple).visit(
                tileOffset, tileShape, calCount, args...))...>(
                tla::get<Is>(callbacks_tuple).visit(tileOffset, tileShape, calCount, args...)...
            );
        }

        // 辅助函数：调用父节点
        template <typename ChildOutputs, int... Is>
        CATLASS_DEVICE auto call_parent_with_outputs(
            MatrixCoord const& tileOffset,
            MatrixCoord const& tileShape,
            uint32_t calCount,
            ChildOutputs const& child_outputs,
            tla::seq<Is...>
        ) {
            constexpr int Rm1 = sizeof...(ChildOps);
            return tla::get<Rm1>(callbacks_tuple).visit(
                tileOffset, tileShape, calCount, 
                tla::get<Is>(child_outputs)...
            );
        }

        // 统一的 visit 签名：可变参数
        template <typename... Args>
        CATLASS_DEVICE auto visit(
            MatrixCoord const& tileOffset,
            MatrixCoord const& tileShape,
            uint32_t calCount,
            Args const&... args
        ) {
            constexpr int Rm1 = sizeof...(ChildOps);
            
            // 访问所有子节点，收集输出
            auto child_outputs = collect_child_outputs(
                tileOffset, tileShape, calCount,
                tla::make_seq<Rm1>{},
                args...
            );
            
            // 将子节点输出传给父节点
            return call_parent_with_outputs(
                tileOffset, tileShape, calCount,
                child_outputs,
                tla::make_seq<Rm1>{}
            );
        }
    };

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
        auto base_callbacks = this->VisitorImpl<ChildOps..., NodeOp>::get_callbacks(
            resource, ub_offset, compute_length,
            blockShapeMNK, blockCoordMNK, subblockShape, subblockCoord,
            gmSubblockC, layoutSubblockC
        );
        return Callbacks<decltype(base_callbacks)>(
            static_cast<decltype(base_callbacks)&&>(base_callbacks)
        );
    }
};

template <class NodeOp, class... ChildOps>
using EVT = TreeVisitor<NodeOp, ChildOps...>;

} // namespace Catlass::Epilogue::Fusion

#endif
