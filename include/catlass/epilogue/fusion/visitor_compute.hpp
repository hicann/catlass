#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_COMPUTE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_COMPUTE_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/fusion/operations.hpp"

namespace Catlass::Epilogue::Fusion {

// 类型萃取：提取 LocalTensor<T> 中的 T
template <typename T>
struct LocalTensorElementType;

template <typename T>
struct LocalTensorElementType<AscendC::LocalTensor<T>> {
    using type = T;
};

template <typename T>
struct LocalTensorElementType<AscendC::LocalTensor<T> const> {
    using type = T;
};

template <typename T>
struct LocalTensorElementType<AscendC::LocalTensor<T>&> {
    using type = T;
};

template <typename T>
struct LocalTensorElementType<AscendC::LocalTensor<T> const&> {
    using type = T;
};

template<
  template <class> class ComputeFn,
  class ElementOutput,
  class ElementCompute,
  AscendC::RoundMode RoundStyle,
  int NumInputs
>
struct VisitorCompute : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    struct Arguments {};
    struct Params {};

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const&, void*) {
        return Params();
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static bool
    can_implement(ProblemShape const&, Arguments const&) {
        return true;
    }

    CATLASS_HOST_DEVICE
    VisitorCompute() {}

    CATLASS_HOST_DEVICE
    VisitorCompute(Params const&) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<ElementOutput> ubOut;
        AscendC::LocalTensor<ElementCompute> ubOutCompute;
        AscendC::LocalTensor<ElementCompute> ubConverted[NumInputs];
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(AscendC::LocalTensor<ElementOutput> ubOut_,
                 AscendC::LocalTensor<ElementCompute> ubOutCompute_,
                 AscendC::LocalTensor<ElementCompute> ubConverted_[NumInputs],
                 uint32_t compute_length_)
            : ubOut(ubOut_), ubOutCompute(ubOutCompute_), compute_length(compute_length_) {
            for (int i = 0; i < NumInputs; ++i) {
                ubConverted[i] = ubConverted_[i];
            }
        }

        template <int I, typename InputsTuple>
        CATLASS_DEVICE void convert_input_helper_from_tuple(
            InputsTuple const& inputs_tuple,
            uint32_t calCount
        ) {
            using InRef = decltype(tla::get<I>(inputs_tuple));
            using ElementInput = typename LocalTensorElementType<typename std::remove_reference<InRef>::type>::type;
            auto const& in = tla::get<I>(inputs_tuple);
            
            if constexpr (!std::is_same_v<ElementInput, ElementCompute>) {
                NumericArrayConverter<ElementCompute, ElementInput, RoundStyle>{}(
                    ubConverted[I], in, calCount);
            } else {
                AscendC::DataCopy(ubConverted[I], in, calCount);
            }
        }

        template <int... Is>
        CATLASS_DEVICE void call_compute_fn(tla::seq<Is...>) {
            ComputeFn<ElementCompute> compute_fn{};
            compute_fn(ubOutCompute, ubConverted[Is]..., compute_length);
        }

        template <typename... ElementInputs>
        CATLASS_DEVICE AscendC::LocalTensor<ElementOutput> const& visit(
            MatrixCoord const& globalTileOffset,    // 不使用
            MatrixCoord const& localTileOffset,     // 新增参数（不使用）
            MatrixCoord const& tileShape,           // 不使用
            uint32_t calCount,
            AscendC::LocalTensor<ElementInputs> const&... inputs
        ) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            static_assert(sizeof...(ElementInputs) == NumInputs, "Input count mismatch");

            auto inputs_tuple = tla::tuple<AscendC::LocalTensor<ElementInputs> const&...>(inputs...);

            // 类型转换：将输入转换到 ElementCompute
            convert_inputs_from_tuple(inputs_tuple, calCount, tla::make_seq<NumInputs>{});
            // AscendC::PipeBarrier<PIPE_ALL>();
            // 执行计算
            call_compute_fn(tla::make_seq<NumInputs>{});
            // AscendC::PipeBarrier<PIPE_ALL>();
            // 输出类型转换
            if constexpr (!std::is_same_v<ElementOutput, ElementCompute>) {
                NumericArrayConverter<ElementOutput, ElementCompute, RoundStyle>{}(
                    ubOut, ubOutCompute, calCount);
            } else {
                AscendC::DataCopy(ubOut, ubOutCompute, calCount);
            }
            // AscendC::PipeBarrier<PIPE_ALL>();
            return ubOut;
        }

        template <typename InputsTuple, int... Is>
        CATLASS_DEVICE void convert_inputs_from_tuple(
            InputsTuple const& inputs_tuple,
            uint32_t calCount,
            tla::seq<Is...>
        ) {
            ((convert_input_helper_from_tuple<Is>(inputs_tuple, calCount)), ...);
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
        AscendC::GlobalTensor<half> const&,
        layout::RowMajor const&
    ) {
        auto ubOut = resource.ubBuf.template GetBufferByByte<ElementOutput>(ub_offset);
        ub_offset += compute_length * sizeof(ElementOutput);

        auto ubOutCompute = resource.ubBuf.template GetBufferByByte<ElementCompute>(ub_offset);
        ub_offset += compute_length * sizeof(ElementCompute);

        AscendC::LocalTensor<ElementCompute> ubConverted[NumInputs];
        for (int i = 0; i < NumInputs; ++i) {
            ubConverted[i] = resource.ubBuf.template GetBufferByByte<ElementCompute>(ub_offset);
            ub_offset += compute_length * sizeof(ElementCompute);
        }

        return Callbacks(ubOut, ubOutCompute, ubConverted, compute_length);
    }
};

} // namespace Catlass::Epilogue::Fusion

#endif
