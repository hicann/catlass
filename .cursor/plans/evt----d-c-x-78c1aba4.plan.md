<!-- 68bc8e72-08e4-48cd-9434-6965a740e168 21464dd3-d0f0-4f1f-b18e-f7f6e96174bf -->
# EVT 最小原型实现计划（明确设计版）

## 目标

实现 EVT 功能完成 `D = C + X`，其中 C 是 GEMM workspace 累加器结果，X 是辅助输入，采用小步编译验证策略。

## 核心设计决策（已明确）

1. **tla 扩展**: 添加 `for_each` 函数
2. **visit 返回**: 返回 `const&`，UB 由 Callbacks 管理生命周期
3. **visit 签名**: 使用可变参数模板，形式统一
4. **事件同步**: Load 用 MTE2，Store 用 MTE3，Compute 不同步
5. **Accumulator**: VisitorAccLoad 通过 get_callbacks 参数获取 gmSubblockC
6. **UB 分配**: 每次 operator() 调用时分配，tile 间复用
7. **tileOffset**: 全局绝对坐标（blockOffset + subblockOffset + tileOffset）
8. **类型转换**: NumericArrayConverter<Target, Source, RoundMode> 3参数
9. **resource 传递**: BlockEpilogue::operator() 通过参数接收

---

## 实现步骤（13步小步开发）

### 第一步：扩展 tla 库 - for_each

**文件**: `include/tla/int_tuple.hpp`

在 `transform_apply` 之后（约第 68 行）添加：

```cpp
// for_each: 对每个元素执行函数，无返回值
namespace detail {
template <class T, class F, int... I>
CATLASS_HOST_DEVICE constexpr
void for_each_impl(T&& t, F&& f, seq<I...>) {
    (f(get<I>(static_cast<T&&>(t))), ...);
}
} // namespace detail

template <class T, class F>
CATLASS_HOST_DEVICE constexpr
void for_each(T&& t, F&& f) {
    detail::for_each_impl(static_cast<T&&>(t), f, tuple_seq<T>{});
}
```

**验证**: 编译项目，确保 tla 扩展无误。

---

### 第二步：添加 EpilogueWithVisitorCallbacks Dispatch Policy

**文件**: `include/catlass/epilogue/dispatch_policy.hpp`

在文件末尾（第 99 行后）添加：

```cpp
// For AtlasA2, Epilogue with Visitor Callbacks (EVT)
struct EpilogueWithVisitorCallbacks {
    using ArchTag = Arch::AtlasA2;
};
```

**验证**: 编译，确认策略定义正确。

---

### 第三步：实现基础操作符

**新建文件**: `include/catlass/epilogue/fusion/operations.hpp`

```cpp
#ifndef CATLASS_EPILOGUE_FUSION_OPERATIONS_HPP
#define CATLASS_EPILOGUE_FUSION_OPERATIONS_HPP

#include "catlass/catlass.hpp"
#include "aclnnop/aclnn_base.h"

namespace Catlass::Epilogue::Fusion {

// 类型转换操作符（3参数版本）
template <
    typename T,
    typename S,
    AscendC::RoundMode RoundMode = AscendC::RoundMode::ROUND_EVEN
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
```

**验证**: 编译此文件，确认操作符定义正确。

---

### 第四步：实现 VisitorImplBase

**新建文件**: `include/catlass/epilogue/fusion/visitor_impl_base.hpp`

```cpp
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
            [](auto&&... op_params) { return tla::tuple<decltype(op_params)...>(op_params...); }
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
                [](auto&&... built_ops) { return tla::tuple<decltype(built_ops)...>(built_ops...); }
            )
        )
    {}

    tla::tuple<Ops...> ops;
};

} // namespace Catlass::Epilogue::Fusion

#endif
```

**验证**: 编译，确认模板元编程正确。

---

### 第五步：实现 VisitorImpl 和 EmptyCallbacks

**新建文件**: `include/catlass/epilogue/fusion/visitor_impl.hpp`

```cpp
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
        return tla::transform_apply(
            ops,
            [&](auto& op) {
                return op.get_callbacks(resource, ub_offset, compute_length,
                                       blockShapeMNK, blockCoordMNK,
                                       subblockShape, subblockCoord,
                                       gmSubblockC, layoutSubblockC);
            },
            [](auto&&... cbs) {
                auto tuple_cbs = tla::tuple<decltype(cbs)...>(cbs...);
                return Callbacks<decltype(tuple_cbs)>(static_cast<decltype(tuple_cbs)&&>(tuple_cbs));
            }
        );
    }
};

} // namespace Catlass::Epilogue::Fusion

#endif
```

**验证**: 编译，确认回调机制正确。

---

### 第六步：实现 VisitorAuxLoad 和 VisitorAccLoad

**新建文件**: `include/catlass/epilogue/fusion/visitor_aux_load.hpp`

```cpp
#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_AUX_LOAD_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_AUX_LOAD_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"

namespace Catlass::Epilogue::Fusion {

template<class Element, class Layout>
struct VisitorAuxLoad : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    struct Arguments {
        Element* ptr_aux = nullptr;
        Element null_default = Element(0);
        Layout layout = {};
    };

    struct Params {
        Element* ptr_aux;
        Element null_default;
        Layout layout;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(Element* ptr_aux_, Element null_default_, Layout const& layout_)
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
            MatrixCoord const& tileShape,
            uint32_t calCount,
            Args const&... /*unused*/
        ) {
            if (params_ptr->ptr_aux != nullptr) {
                // 从用户提供的 ptr_aux 加载
                AscendC::GlobalTensor<Element> gmAux;
                gmAux.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params_ptr->ptr_aux));
                auto gmOffset = params_ptr->layout.GetOffset(globalTileOffset);
                auto gmTileAux = gmAux[gmOffset];
                AscendC::DataCopy(ubAux, gmTileAux, calCount);
            } else {
                // 从 gmSubblockC 加载（AccLoad 场景）
                auto gmOffset = layoutSubblockC.GetOffset(globalTileOffset);
                auto gmTileC = gmSubblockC[gmOffset];
                AscendC::DataCopy(ubAux, gmTileC, calCount);
            }
            // 同步 MTE2: GM->UB 完成
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            return ubAux;
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
```

**验证**: 编译，确认加载节点正确。

---

### 第七步：实现 VisitorCompute

**新建文件**: `include/catlass/epilogue/fusion/visitor_compute.hpp`

```cpp
#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_COMPUTE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_COMPUTE_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/fusion/operations.hpp"

namespace Catlass::Epilogue::Fusion {

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

        template <typename... ElementInputs>
        CATLASS_DEVICE AscendC::LocalTensor<ElementOutput> const& visit(
            MatrixCoord const&,
            MatrixCoord const&,
            uint32_t calCount,
            AscendC::LocalTensor<ElementInputs> const&... inputs
        ) {
            static_assert(sizeof...(ElementInputs) == NumInputs, "Input count mismatch");

            auto inputs_tuple = tla::tuple<AscendC::LocalTensor<ElementInputs> const&...>(inputs...);

            // 类型转换：将输入转换到 ElementCompute
            [&]<int... Is>(tla::seq<Is...>) {
                ([&] {
                    using ElementInput = typename tla::tuple_element<Is, decltype(inputs_tuple)>::type::Element;
                    auto const& in = tla::get<Is>(inputs_tuple);
                    
                    if constexpr (!std::is_same_v<ElementInput, ElementCompute>) {
                        NumericArrayConverter<ElementCompute, ElementInput, RoundStyle>{}(
                            ubConverted[Is], in, calCount);
                    } else {
                        AscendC::DataCopy(ubConverted[Is], in, calCount);
                    }
                }(), ...);
            }(tla::make_index_sequence<NumInputs>{});

            // 执行计算
            ComputeFn<ElementCompute> compute_fn{};
            [&]<int... Is>(tla::seq<Is...>) {
                compute_fn(ubOutCompute, ubConverted[Is]..., calCount);
            }(tla::make_index_sequence<NumInputs>{});

            // 输出类型转换
            if constexpr (!std::is_same_v<ElementOutput, ElementCompute>) {
                NumericArrayConverter<ElementOutput, ElementCompute, RoundStyle>{}(
                    ubOut, ubOutCompute, calCount);
            } else {
                AscendC::DataCopy(ubOut, ubOutCompute, calCount);
            }

            return ubOut;
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
```

**验证**: 编译，确认计算节点正确。

---

### 第八步：实现 VisitorAuxStore

**新建文件**: `include/catlass/epilogue/fusion/visitor_aux_store.hpp`

```cpp
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
        Element* ptr_aux = nullptr;
        Layout layout = {};
    };

    struct Params {
        Element* ptr_aux;
        Layout layout;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(Element* ptr_aux_, Layout const& layout_)
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
            MatrixCoord const&,
            uint32_t calCount,
            AscendC::LocalTensor<ElementInput> const& input
        ) {
            // 类型转换
            if constexpr (!std::is_same_v<ElementInput, Element>) {
                NumericArrayConverter<Element, ElementInput, RoundStyle>{}(ubAux, input, calCount);
            } else {
                AscendC::DataCopy(ubAux, input, calCount);
            }

            // 同步 V->MTE3: 向量计算完成
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // 写回 GM
            if (params_ptr->ptr_aux != nullptr) {
                AscendC::GlobalTensor<Element> gmAux;
                gmAux.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params_ptr->ptr_aux));
                auto gmOffset = params_ptr->layout.GetOffset(globalTileOffset);
                auto gmTileAux = gmAux[gmOffset];
                AscendC::DataCopy(gmTileAux, ubAux, calCount);
            }
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
```

**验证**: 编译，确认存储节点正确。

---

### 第九步：实现 TreeVisitor

**新建文件**: `include/catlass/epilogue/fusion/tree_visitor.hpp`

```cpp
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
            auto child_outputs = [&]<int... Is>(tla::seq<Is...>) {
                return tla::tuple<decltype(tla::get<Is>(callbacks_tuple).visit(
                    tileOffset, tileShape, calCount, args...))...>(
                    tla::get<Is>(callbacks_tuple).visit(tileOffset, tileShape, calCount, args...)...
                );
            }(tla::make_index_sequence<Rm1>{});
            
            // 将子节点输出传给父节点
            return [&]<int... Is>(tla::seq<Is...>) {
                return tla::get<Rm1>(callbacks_tuple).visit(
                    tileOffset, tileShape, calCount, 
                    tla::get<Is>(child_outputs)...
                );
            }(tla::make_index_sequence<Rm1>{});
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
        auto base_callbacks = VisitorImpl<ChildOps..., NodeOp>::get_callbacks(
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
```

**验证**: 编译，确认树形组合正确。

---

### 第十步：实现 BlockEpilogue EVT 版本

**新建文件**: `include/catlass/epilogue/block/block_epilogue_visitor.hpp`

```cpp
#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_VISITOR_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_VISITOR_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Epilogue::Block {

template <
    class CType_,
    uint32_t COMPUTE_LENGTH_,
    class ElementCompute_,
    class FusionCallbacks_
>
class BlockEpilogue<
    EpilogueWithVisitorCallbacks,
    CType_,
    COMPUTE_LENGTH_,
    ElementCompute_,
    FusionCallbacks_
> {
public:
    using DispatchPolicy = EpilogueWithVisitorCallbacks;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementD = ElementC;
    using LayoutD = LayoutC;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;
    using ElementCompute = ElementCompute_;
    using FusionCallbacks = FusionCallbacks_;

    struct Params {
        typename FusionCallbacks::Params fusion_params;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(typename FusionCallbacks::Params const& fusion_params_)
            : fusion_params(fusion_params_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params)
        : params(params), fusion_callbacks(params.fusion_params)
    {
        // 初始化事件标志
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        // 等待所有事件完成
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void operator()(
        Arch::Resource<ArchTag>& resource,
        GemmCoord const& blockShapeMNK,
        GemmCoord const& blockCoordMNK,
        GemmCoord const& actualBlockShapeMNK,
        AscendC::GlobalTensor<ElementC> const& gmBlockC,
        layout::RowMajor const& layoutBlockC
    )
    {
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;

        // 子块划分
        MatrixCoord subblockShape{
            CeilDiv(actualBlockShape.row(), static_cast<uint32_t>(AscendC::GetSubBlockNum())),
            actualBlockShape.column()
        };
        MatrixCoord subblockCoord{AscendC::GetSubBlockIdx(), 0};
        MatrixCoord actualSubblockShape = MatrixCoord::Min(
            subblockShape, actualBlockShape - subblockCoord * subblockShape);
        MatrixCoord subblockOffset = subblockCoord * subblockShape;

        // 获取 gmSubblockC 和 layoutSubblockC
        auto gmSubblockC = gmBlockC[layoutBlockC.GetOffset(subblockOffset)];
        auto layoutSubblockC = layoutBlockC.GetTileLayout(actualSubblockShape);

        // 分配 UB 空间并获取 callbacks
        uint32_t ub_offset = 0;
        auto callbacks = fusion_callbacks.get_callbacks(
            resource, ub_offset, COMPUTE_LENGTH,
            blockShapeMNK, blockCoordMNK,
            actualSubblockShape, subblockCoord,
            gmSubblockC, layoutSubblockC
        );

        callbacks.begin_epilogue();

        uint32_t rows = actualSubblockShape.row();
        uint32_t cols = actualSubblockShape.column();

        // 遍历所有 tile（tile 间复用 UB）
        for (uint32_t r = 0; r < rows; ) {
            callbacks.begin_row(r);

            for (uint32_t c = 0; c < cols; ) {
                uint32_t remainCols = cols - c;
                uint32_t tileCols = (remainCols < COMPUTE_LENGTH) ? remainCols : COMPUTE_LENGTH;
                uint32_t rowsCap = (tileCols == 0) ? 0 : (COMPUTE_LENGTH / tileCols);
                if (rowsCap == 0) rowsCap = 1;

                uint32_t remainRows = rows - r;
                uint32_t tileRows = (remainRows < rowsCap) ? remainRows : rowsCap;

                MatrixCoord tileShape{tileRows, tileCols};
                MatrixCoord localTileOffset{r, c};
                // 计算全局绝对坐标
                MatrixCoord globalTileOffset = blockOffset + subblockOffset + localTileOffset;
                uint32_t calCount = tileRows * tileCols;

                // 访问当前 tile
                callbacks.visit(globalTileOffset, tileShape, calCount);

                c += tileCols;
            }

            callbacks.end_row(r);

            // 推进行索引
            uint32_t rowsCapAllCols = (cols == 0) ? 0 : (COMPUTE_LENGTH / ((cols < COMPUTE_LENGTH) ? cols : COMPUTE_LENGTH));
            if (rowsCapAllCols == 0) rowsCapAllCols = 1;
            uint32_t advR = ((rows - r) < rowsCapAllCols) ? (rows - r) : rowsCapAllCols;
            r += advR;
        }

        callbacks.end_epilogue();
    }

private:
    Params params;
    FusionCallbacks fusion_callbacks;
};

} // namespace Catlass::Epilogue::Block

#endif
```

在 `include/catlass/epilogue/block/block_epilogue.hpp` 末尾（第 42 行后）添加：

```cpp
#include "catlass/epilogue/block/block_epilogue_visitor.hpp"
```

**验证**: 编译，确认 BlockEpilogue 特化正确。

---

### 第十一步：创建融合总头文件

**新建文件**: `include/catlass/epilogue/fusion/fusion.hpp`

```cpp
#ifndef CATLASS_EPILOGUE_FUSION_FUSION_HPP
#define CATLASS_EPILOGUE_FUSION_FUSION_HPP

#include "catlass/epilogue/fusion/operations.hpp"
#include "catlass/epilogue/fusion/visitor_impl_base.hpp"
#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/fusion/visitor_aux_load.hpp"
#include "catlass/epilogue/fusion/visitor_compute.hpp"
#include "catlass/epilogue/fusion/visitor_aux_store.hpp"
#include "catlass/epilogue/fusion/tree_visitor.hpp"

#endif
```

**验证**: 编译整个项目，确认所有头文件包含正确。

---

### 第十二步：创建 MatmulEvt Kernel

**新建文件**: `include/catlass/gemm/kernel/matmul_evt.hpp`

```cpp
#ifndef CATLASS_GEMM_KERNEL_MATMUL_EVT_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_EVT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class MatmulEvt {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockEpilogue = BlockEpilogue_;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    using BlockScheduler = BlockScheduler_;

    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrWorkspace;
        EpilogueParams epilogueParams;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_,
            GM_ADDR ptrA_, LayoutA const& layoutA_,
            GM_ADDR ptrB_, LayoutB const& layoutB_,
            GM_ADDR ptrWorkspace_, EpilogueParams const& epilogueParams_
        ) : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
            ptrWorkspace(ptrWorkspace_), epilogueParams(epilogueParams_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        size_t elementSize;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        typename BlockEpilogue::FusionCallbacks::Arguments evt_args;
    };

    static bool CanImplement(const Arguments& args)
    {
        return BlockEpilogue::FusionCallbacks::can_implement(args.problemShape, args.evt_args);
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return args.elementSize * args.problemShape.m() * args.problemShape.n() +
               BlockEpilogue::FusionCallbacks::get_workspace_size(args.problemShape, args.evt_args);
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        GemmCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        uint32_t k = problemShape.k();
        LayoutA layoutA{m, k};
        LayoutB layoutB{k, n};

        // 转换 EVT Arguments 到 Params
        typename BlockEpilogue::FusionCallbacks::Params fusion_params = 
            BlockEpilogue::FusionCallbacks::to_underlying_arguments(
                problemShape, args.evt_args, 
                workspace + args.elementSize * m * n  // EVT workspace 在 GEMM workspace 之后
            );
        
        EpilogueParams epilogueParams{fusion_params};
        Params params{problemShape, args.ptrA, layoutA, args.ptrB, layoutB, workspace, epilogueParams};
        return params;
    }

    CATLASS_DEVICE
    MatmulEvt() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const& params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);
        layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);

            blockMmad(
                gmA[gmOffsetA], params.layoutA,
                gmB[gmOffsetB], params.layoutB,
                gmC[gmOffsetC], layoutC,
                actualBlockShape);

            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const& params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockEpilogue blockEpilogue(resource, params.epilogueParams);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);
        layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());

        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        GemmCoord blockShape = L1TileShape::ToCoord();
        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            auto gmBlockC = gmC[layoutC.GetOffset(blockCoord.GetCoordMN() * blockShape.GetCoordMN())];
            auto layoutBlockC = layoutC.GetTileLayout(actualBlockShape.GetCoordMN());
            
            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
            blockEpilogue(resource, blockShape, blockCoord, actualBlockShape, gmBlockC, layoutBlockC);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif
```

**验证**: 编译，确认新 Kernel 正确。

---

### 第十三步：修改示例代码

**文件**: `examples/32_matmul_add_evt/matmul_add_evt.cpp`

修改点：

1. **包含头文件**（约第 19 行后）：
```cpp
#include "catlass/epilogue/fusion/fusion.hpp"
#include "catlass/gemm/kernel/matmul_evt.hpp"
```

2. **定义 EVT**（约第 110 行后，BlockMmad 定义之后）：
```cpp
// 定义 EVT: D = C + X
using ElementCompute = half;
constexpr uint32_t computeLength = 16384;

using EVT = Epilogue::Fusion::TreeVisitor<
    Epilogue::Fusion::VisitorAuxStore<half, AscendC::RoundMode::ROUND_EVEN, LayoutC>,
    Epilogue::Fusion::TreeVisitor<
        Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Plus, half, ElementCompute, AscendC::RoundMode::ROUND_EVEN, 2>,
        Epilogue::Fusion::VisitorAccLoad<half, LayoutC>,  // 加载 C (workspace)
        Epilogue::Fusion::VisitorAuxLoad<half, LayoutC>   // 加载 X
    >
>;
```

3. **修改 BlockEpilogue 定义**（替换原有定义）：
```cpp
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    Epilogue::EpilogueWithVisitorCallbacks,
    CType,
    computeLength,
    ElementCompute,
    EVT
>;
```

4. **准备 EVT Arguments**（约第 120 行，两个分支之前）：
```cpp
// 准备 EVT Arguments
typename EVT::Arguments evt_args{
    {reinterpret_cast<half*>(deviceD), layoutD},  // Store: 输出到 D
    {
        {},  // Compute: Plus 无参数
        {nullptr, half(0), layoutD},  // AccLoad: C (workspace)，nullptr 表示使用 workspace
        {reinterpret_cast<half*>(deviceD), half(0), layoutD}  // AuxLoad: X (存储在 deviceD)
    }
};
```

5. **使用新 Kernel**（替换两个分支的 MatmulKernel 和 Arguments）：
```cpp
// 第一个分支（m > n）
using MatmulKernel = Gemm::Kernel::MatmulEvt<BlockMmad, BlockEpilogue, BlockScheduler>;
typename MatmulKernel::Arguments arguments{options.problemShape, sizeof(half), deviceA, deviceB, evt_args};

// 第二个分支（m <= n）同样修改
```


**验证**: 编译并运行示例，验证结果正确性。

---

## 验证清单

每步完成后验证：

- 代码编译通过
- 无 linter 错误
- 逻辑正确

最终验证：

- `32_matmul_add_evt` 编译成功
- 运行结果与 golden 一致
- 架构清晰，易于扩展

## 关键注意事项

1. **UB 空间**: 每个 Op 分配独立空间，检查 `ub_offset` 不超过 `ArchTag::UB_SIZE`
2. **事件同步**: Load 用 MTE2，Store 用 MTE3，Compute 不同步
3. **类型一致性**: ElementC/ElementCompute/ElementD 兼容
4. **全局坐标**: tileOffset = blockOffset + subblockOffset + localTileOffset
5. **AccLoad**: ptr_aux 为 nullptr 时从 gmSubblockC 读取
6. **编译期优化**: `if constexpr` 跳过不必要的类型转换

### To-dos

- [ ] 扩展 tla 库：添加 for_each 和多参数 transform_apply
- [ ] 添加 EpilogueWithVisitorCallbacks dispatch policy
- [ ] 实现 VisitorImplBase - 参数管理和 workspace 处理
- [ ] 实现 VisitorImpl - Callbacks 聚合和生命周期
- [ ] 实现 TreeVisitor - 树形组合语义
- [ ] 实现基础操作：NumericArrayConverter, Plus 等
- [ ] 实现 VisitorCompute - N 元计算节点
- [ ] 实现 VisitorAuxLoad - GM 到 UB 加载
- [ ] 实现 VisitorAuxStore - UB 到 GM 存储
- [ ] 实现 BlockEpilogue EVT 版本 - 单缓冲遍历
- [ ] 更新 32_matmul_add_evt 示例使用 EVT
- [ ] 创建 fusion.hpp 总头文件
- [ ] 编译测试并验证结果正确性