#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_AUX_LOAD_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_AUX_LOAD_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Fusion {

template<class Element, class Layout>
struct VisitorAuxLoad : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    struct Arguments {
        GM_ADDR ptr_aux = nullptr;
        Element null_default = Element(0);
        Layout layout = {};
    };

    struct Params {
        GM_ADDR ptr_aux;
        Element null_default;
        Layout layout;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptr_aux_, Element null_default_, Layout const& layout_)
            : ptr_aux(ptr_aux_), null_default(null_default_), layout(layout_) {}
    };

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return Params(args.ptr_aux, args.null_default, args.layout);
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static bool
    can_implement(ProblemShape const&, Arguments const& args) {
        return true;  // ptr_aux 可以为 nullptr（用于 AccLoad）
    }

    CATLASS_HOST_DEVICE
    VisitorAuxLoad() {}

    CATLASS_HOST_DEVICE
    VisitorAuxLoad(Params const& params_) : params(params_) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<Element> ubAux;
        Params const* params_ptr;
        uint32_t compute_length;
        AscendC::GlobalTensor<Element> gmSubblockC;
        layout::RowMajor layoutSubblockC;

        CATLASS_DEVICE
        Callbacks(AscendC::LocalTensor<Element> ubAux_,
                 Params const* params_ptr_,
                 uint32_t compute_length_,
                 AscendC::GlobalTensor<Element> const& gmSubblockC_,
                 layout::RowMajor const& layoutSubblockC_)
            : ubAux(ubAux_), params_ptr(params_ptr_), compute_length(compute_length_),
              gmSubblockC(gmSubblockC_), layoutSubblockC(layoutSubblockC_) {}

        template <typename... Args>
        CATLASS_DEVICE AscendC::LocalTensor<Element> const& visit(
            MatrixCoord const& globalTileOffset,
            MatrixCoord const& localTileOffset,  // 新增参数
            MatrixCoord const& tileShape,
            uint32_t calCount,
            Args const&... /*unused*/
        ) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            // AscendC::PipeBarrier<PIPE_ALL>();
            auto layoutUb = layout::RowMajor::MakeLayoutInUb<Element>(tileShape);
            using CopyGm2UbT = Epilogue::Tile::CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>>;
            CopyGm2UbT copyGm2Ub{};

            if (params_ptr->ptr_aux != nullptr) {
                // 从用户提供的 ptr_aux 加载（使用全局坐标，tile 封装处理跨距）
                AscendC::GlobalTensor<Element> gmAux;
                gmAux.SetGlobalBuffer((__gm__ Element*)(params_ptr->ptr_aux));
                auto gmTile = gmAux[params_ptr->layout.GetOffset(globalTileOffset)];
                auto layoutSrc = params_ptr->layout.GetTileLayout(tileShape);
                copyGm2Ub(ubAux, gmTile, layoutUb, layoutSrc);
            } else {
                // 从 gmSubblockC 加载（AccLoad 场景，使用局部坐标，tile 封装处理跨距）
                auto gmTile = gmSubblockC[layoutSubblockC.GetOffset(localTileOffset)];
                auto layoutSrc = layoutSubblockC.GetTileLayout(tileShape);
                copyGm2Ub(ubAux, gmTile, layoutUb, layoutSrc);
            }
            // 同步 MTE2: GM->UB 完成
            // AscendC::PipeBarrier<PIPE_ALL>();

            // AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            // AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            return ubAux;
            // AscendC::PipeBarrier<PIPE_ALL>();
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length,
        GemmCoord const&,
        GemmCoord const&,
        MatrixCoord const&,
        MatrixCoord const&,
        AscendC::GlobalTensor<Element> const& gmSubblockC,
        layout::RowMajor const& layoutSubblockC
    ) {
        auto ubAux = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        return Callbacks(ubAux, &params, compute_length, gmSubblockC, layoutSubblockC);
    }

    Params params;
};

// AccLoad: 加载累加器 C 的便捷别名（ptr_aux 为 nullptr 时从 gmSubblockC 读取）
template<class Element, class Layout>
using VisitorAccLoad = VisitorAuxLoad<Element, Layout>;

} // namespace Catlass::Epilogue::Fusion

#endif



