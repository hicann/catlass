#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_AUX_STORE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_AUX_STORE_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/fusion/operations.hpp"

namespace Catlass::Epilogue::Fusion {

template<
  class Element,
  AscendC::RoundMode RoundStyle,
  class Layout
>
struct VisitorAuxStore : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    struct Arguments {
        GM_ADDR ptr_aux = nullptr;
        Layout layout = {};
    };

    struct Params {
        GM_ADDR ptr_aux;
        Layout layout;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptr_aux_, Layout const& layout_)
            : ptr_aux(ptr_aux_), layout(layout_) {}
    };

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return Params(args.ptr_aux, args.layout);
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static bool
    can_implement(ProblemShape const&, Arguments const& args) {
        return args.ptr_aux != nullptr;
    }

    CATLASS_HOST_DEVICE
    VisitorAuxStore() {}

    CATLASS_HOST_DEVICE
    VisitorAuxStore(Params const& params_) : params(params_) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<Element> ubAux;
        Params const* params_ptr;
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(AscendC::LocalTensor<Element> ubAux_,
                 Params const* params_ptr_,
                 uint32_t compute_length_)
            : ubAux(ubAux_), params_ptr(params_ptr_), compute_length(compute_length_) {}

        template <typename ElementInput>
        CATLASS_DEVICE void visit(
            MatrixCoord const& globalTileOffset,
            MatrixCoord const& localTileOffset,  // 新增参数（不使用）
            MatrixCoord const& tileShape,
            uint32_t calCount,
            AscendC::LocalTensor<ElementInput> const& input
        ) {
            // AscendC::PipeBarrier<PIPE_ALL>();
            // 类型转换
            if constexpr (!std::is_same_v<ElementInput, Element>) {
                NumericArrayConverter<Element, ElementInput, RoundStyle>{}(ubAux, input, calCount);
            } else {
                AscendC::DataCopy(ubAux, input, calCount);
            }

            // 同步 V->MTE3: 向量计算完成
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // 写回 GM（使用全局坐标，逐行写回处理跨距）
            if (params_ptr->ptr_aux != nullptr) {
                AscendC::GlobalTensor<Element> gmAux;
                gmAux.SetGlobalBuffer((__gm__ Element*)(params_ptr->ptr_aux));
                uint32_t cols = tileShape.column();
                uint32_t rows = (cols == 0) ? 0 : (calCount / cols);
                for (uint32_t r = 0; r < rows; ++r) {
                    auto rowGlobalOffset = params_ptr->layout.GetOffset(globalTileOffset + MatrixCoord{r, 0});
                    auto gmRow = gmAux[rowGlobalOffset];
                    auto ubRow = ubAux[r * cols];
                    AscendC::DataCopy(gmRow, ubRow, cols);
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
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
        AscendC::GlobalTensor<Element> const&,
        layout::RowMajor const&
    ) {
        auto ubAux = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        return Callbacks(ubAux, &params, compute_length);
    }

    Params params;
};

} // namespace Catlass::Epilogue::Fusion

#endif



