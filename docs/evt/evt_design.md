# Epilogue Visitor Tree设计文档

## 1. 节点定义和语义说明

### 1.1 VisitorImplBase

```C++
namespace Catlass::Epilogue::Fusion {

template <class... Ops>
struct VisitorImplBase {
  // 每个 Op 的存储聚合：SharedStorage → ubStorage
  using ubStorage = tla::tuple<typename Ops::ubStorage...>;

  // Host 侧融合参数
  using Arguments = tla::tuple<typename Ops::Arguments...>;

  // Device 侧融合参数（Kernel 入口 API）
  using Params = tla::tuple<typename Ops::Params...>;

  template <class ProblemShape>
  CATLASS_HOST_DEVICE static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    uint8_t* op_workspace = reinterpret_cast<uint8_t*>(workspace);
    return tla::transform_apply(
      tla::tuple<Ops...>{}, args,
      [&] (auto&& op_tag, auto const& op_args) {
        using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
        auto ret = Op::to_underlying_arguments(problem_shape, op_args, op_workspace);
        if (op_workspace != nullptr) {
          size_t sz = Op::get_workspace_size(problem_shape, op_args);
          op_workspace += RoundUp(sz, static_cast<size_t>(16));
        }
        return ret;
      },
      [] (auto&&... op_params) { return tla::tuple<decltype(op_params)...>(op_params...); }
    );
  }

  template <class ProblemShape>
  CATLASS_HOST_DEVICE static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return tla::transform_apply(
      tla::tuple<Ops...>{}, args,
      [&] (auto&& op_tag, auto const& op_args) {
        using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
        return Op::can_implement(problem_shape, op_args);
      },
      [] (auto&&... ok) { return (true && ... && ok); }
    );
  }

  template <class ProblemShape>
  CATLASS_HOST_DEVICE static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return tla::transform_apply(
      tla::tuple<Ops...>{}, args,
      [&] (auto&& op_tag, auto const& op_args) {
        using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
        size_t sz = Op::get_workspace_size(problem_shape, op_args);
        return RoundUp(sz, static_cast<size_t>(16));
      },
      [] (auto&&... rounded_sizes) { return (size_t{0} + ... + rounded_sizes); }
    );
  }

  template <class ProblemShape>
  CATLASS_HOST_DEVICE static Catlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    Catlass::Status status = Catlass::Status::kSuccess;
    uint8_t* op_workspace = reinterpret_cast<uint8_t*>(workspace);

    (void) tla::transform_apply(
      tla::tuple<Ops...>{}, args,
      [&] (auto&& op_tag, auto const& op_args) {
        using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
        if (status != Catlass::Status::kSuccess) {
          return status;
        }
        status = Op::initialize_workspace(problem_shape, op_args, op_workspace);
        if (op_workspace != nullptr) {
          size_t sz = Op::get_workspace_size(problem_shape, op_args);
          op_workspace += RoundUp(sz, static_cast<size_t>(16));
        }
        return status;
      },
      [&] (auto const&... /*ignored*/) { return status; }
    );

    return status;
  }

  CATLASS_HOST_DEVICE
  VisitorImplBase() {}

  // 使用 Params 与 ubStorage 构造每个 Op 实例
  CATLASS_HOST_DEVICE
  VisitorImplBase(Params const& params, ubStorage const& storage)
  : ops(
      tla::transform_apply(
        tla::tuple<Ops...>{}, params, storage,
        [] (auto&& op_tag, auto const& op_params, auto const& op_storage) {
          using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
          return Op(op_params, op_storage);
        },
        [] (auto&&... built_ops) { return tla::tuple<decltype(built_ops)...>(built_ops...); }
      )
    )
  {}

  // 每个 Op 的持久化实例（可存放描述符、标量、计数器等）
  tla::tuple<Ops...> ops;
};

} // namespace Catlass::Epilogue::Fusion
```
提供节点、Arguments、Params存储能力。

能力
- **融合编排**: 用 `tla::tuple` 聚合 `Ops...`，以 `transform_apply` 串接 `Load/Compute/Broadcast/Reduction/Store` 的可组合 Visitor 流水。
- **参数桥接**: `to_underlying_arguments` 将每个 `Op` 的 `Arguments`→`Params`，并线性切分对齐 `workspace`（16B 对齐）。
- **可实现性判断**: `can_implement` 汇总各 `Op` 约束，启动前早失败。
- **工作区治理**: `get_workspace_size`/`initialize_workspace` 统一度量、对齐与初始化，错误即停。
- **UB 资源聚合**: 以 `ubStorage` 聚合各 `Op::ubStorage`，构造时与 `Params` 一一配对，便于 UB 受限下资源管理。

动机
- **解耦编排与算子实现**: 统一处理融合流程、参数转换与工作区分配，减少重复代码与维护成本。
- **适配 NPU UB 受限**: 通过统一对齐与滚动分配避免碎片与越界，充分复用 UB。
- **稳定上层 API**: 在 GEMM/MLA/GEMV 等内核中复用 EVT，保持一致的 `Arguments/Params/workspace` 启动路径。
- **可靠性与性能**: 启动前集中裁剪无效组合，运行期轻量分派、常量折叠。
- **易扩展**: 新增节点仅需提供 `Arguments/Params/ubStorage` 与标准静态接口，即可无侵入纳入融合树。

### 1.2 VisitorImpl

```cpp
// 新增：通用 VisitorImpl 基类（依赖 tla::tuple / transform_apply / for_each）
namespace Catlass::Epilogue::Fusion {

template <class... Ops>
struct VisitorImpl : VisitorImplBase<Ops...> {
  using VisitorImplBase<Ops...>::VisitorImplBase;
  using VisitorImplBase<Ops...>::ops;

  template <class CallbacksTuple>
  struct Callbacks {
    CallbacksTuple callbacks_tuple;

    CUTLASS_DEVICE void begin_epilogue() {
      for_each(callbacks_tuple, [] (auto& cb) { cb.begin_epilogue(); });
    }

    CUTLASS_DEVICE void begin_step(int step_idx) {
      for_each(callbacks_tuple, [&] (auto& cb) { cb.begin_step(step_idx); });
    }

    CUTLASS_DEVICE void begin_row(int row_idx) {
      for_each(callbacks_tuple, [&] (auto& cb) { cb.begin_row(row_idx); });
    }

    // 由具体 Op 的 Callbacks 特化 visit（N 元算子自定义），默认删除
    template <typename... ElementInputs>
    CUTLASS_DEVICE auto
    visit(MatrixCoord const& tileOffset,
          MatrixCoord const& tileShape,
          Layout const& layoutTileInUb,
          int32_t calCount,
          AscendC::LocalTensor<ElementInputs> const&... inputs) = delete;

    CUTLASS_DEVICE void end_row(int row_idx) {
      for_each(callbacks_tuple, [&] (auto& cb) { cb.end_row(row_idx); });
    }

    CUTLASS_DEVICE void end_step(int step_idx) {
      for_each(callbacks_tuple, [&] (auto& cb) { cb.end_step(step_idx); });
    }

    CUTLASS_DEVICE void end_epilogue() {
      for_each(callbacks_tuple, [] (auto& cb) { cb.end_epilogue(); });
    }
  };

  // 汇聚所有 Op 的回调为同一个 Callbacks 对象
  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(
      Arch::Resource<ArchTag> &resource,
      uint32_t& UB_offset,
      GemmCoord const& blockShapeMNK,
      GemmCoord const& blockCoordMNK,
      Layout const& layoutBlockC,
      MatrixCoord const& subblockShape,
      MatrixCoord const& subblockCoord,
      AscendC::GlobalTensor<ElementCompute> const& gmSubblockC,
      Layout const& layoutSubblockC,
      uint32_t eventId) {
    return tla::transform_apply(
      ops,
      [&] (auto& op) {
        return op.get_callbacks(resource, UB_offset, blockShapeMNK, blockCoordMNK, 
                               layoutBlockC, subblockShape, subblockCoord, 
                               gmSubblockC, layoutSubblockC, eventId);
      },
      [] (auto&&... cbs) {
        auto tuple_cbs = tla::make_tuple(cbs...);               // 期望由 tla 提供
        return Callbacks<decltype(tuple_cbs)>{tuple_cbs};
      }
    );
  }
};

// 可选：空回调别名
using EmptyCallbacks = VisitorImpl<>::Callbacks<tla::tuple<>>;

} // namespace Catlass::Epilogue::Fusion
```

能力
- **统一回调编排**: 聚合多 `Op` 的回调为单一 `Callbacks`，一次性驱动 `begin_* / end_* / visit` 生命周期。
- **通用 N 元算子**: `visit` 留给各 `Op` 回调特化，实现任意元数输入的逐片处理。

动机
- **解耦编排与实现**: 骨架负责“回调聚合 + 生命周期分发”，各 `Op` 仅关注本地语义与 UB 资源。
- **编译期融合**: 以 `tla` 编译期元运算完成拓扑与参数装配，降低运行期开销。
- **语义对齐 CUTLASS**: 保持 `Callbacks` 生命周期与 `get_callbacks` 工厂接口，便于迁移/替换。
- **适配 UB 受限**: 通过 `VisitorImplBase` 存放 `Ops...` 持久实例，配合上层 UB 规划实现零运行期分配。


### 1.3 TreeVisitor

```cpp
namespace Catlass::Epilogue::Fusion {

template <class NodeOp, class... ChildOps>
struct TreeVisitor : VisitorImpl<ChildOps..., NodeOp> {

  using VisitorImpl<ChildOps..., NodeOp>::VisitorImpl;

  template<class CallbacksImpl>
  struct Callbacks : CallbacksImpl {
    CUTLASS_DEVICE
    Callbacks(CallbacksImpl&& impl)
      : CallbacksImpl(tla::forward<CallbacksImpl>(impl)) {}

    using CallbacksImpl::callbacks_tuple;

    template <typename... ElementInputs>
    CUTLASS_DEVICE auto
    visit(MatrixCoord const& tileOffset,
          MatrixCoord const& tileShape,
          Layout const& layoutTileInUb,
          int32_t calCount,
          AscendC::LocalTensor<ElementInputs> const&... inputs) {
      constexpr int Rm1 = sizeof...(ChildOps);
      return tla::detail::tapply(callbacks_tuple,
        [&] (auto& child_callbacks) {
          return child_callbacks.visit(tileOffset, tileShape, layoutTileInUb, calCount, inputs...);
        },
        [&] (auto&&... child_outputs) {
          return tla::get<Rm1>(callbacks_tuple).visit(tileOffset, tileShape, layoutTileInUb, calCount, child_outputs...);
        },
        tla::make_seq<Rm1>{}
      );
    }
  };

  // Callbacks factory
  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    Arch::Resource<ArchTag> &resource,
    uint32_t& UB_offset,
    GemmCoord const& blockShapeMNK,
    GemmCoord const& blockCoordMNK,
    Layout const& layoutBlockC,
    MatrixCoord const& subblockShape,
    MatrixCoord const& subblockCoord,
    AscendC::GlobalTensor<ElementCompute> const& gmSubblockC,
    Layout const& layoutSubblockC,
    uint32_t eventId
  ) {
    return Callbacks<
    decltype(VisitorImpl<ChildOps..., NodeOp>::
      get_callbacks(
        resource, UB_offset, blockShapeMNK, blockCoordMNK,
        layoutBlockC, subblockShape, subblockCoord,
        gmSubblockC, layoutSubblockC, eventId
      ))>(
      VisitorImpl<ChildOps..., NodeOp>::
      get_callbacks(
        resource, UB_offset, blockShapeMNK, blockCoordMNK,
        layoutBlockC, subblockShape, subblockCoord,
        gmSubblockC, layoutSubblockC, eventId
      )
    );
  }
};

template <class NodeOp, class... ChildOps>
using EVT = TreeVisitor<NodeOp, ChildOps...>;

} // namespace Catlass::Epilogue::Fusion
```

能力
- **树形融合编排**: 支持一个父节点操作符(NodeOp)与多个子节点操作符(ChildOps)的树形组合，实现复杂的数据流图
- **层次化回调**: 子节点先执行，结果作为父节点的输入，支持任意深度的操作符嵌套
- **类型安全**: 编译期验证操作符间的输入输出类型匹配

设计动机
- **保持CUTLASS语义**: 维持TreeVisitor的树形组合语义，便于从CUTLASS迁移

### 1.4 TopologicalVisitor

```cpp
namespace Catlass::Epilogue::Fusion {

template<
  class ElementCompute,
  class EdgeTuple,
  class... Ops
>
struct TopologicalVisitor : VisitorImpl<Ops...> {
  static_assert(tla::is_static_v<EdgeTuple>);
  static_assert(tla::rank(EdgeTuple{}) == sizeof...(Ops));
  static_assert(sizeof...(Ops) > 1);

  using VisitorImpl<Ops...>::VisitorImpl;

  template<class CallbacksImpl>
  struct Callbacks : CallbacksImpl {
    // UB缓冲区引用
    AscendC::LocalTensor<ElementCompute> ubCompute[sizeof...(Ops) - 1];
    
    CUTLASS_DEVICE
    Callbacks(CallbacksImpl&& impl, AscendC::LocalTensor<ElementCompute> ubCompute_[])
      : CallbacksImpl(tla::forward<CallbacksImpl>(impl)) {
      for (int i = 0; i < sizeof...(Ops) - 1; ++i) {
        ubCompute[i] = ubCompute_[i];
      }
    }

    using CallbacksImpl::callbacks_tuple;

    template <typename... ElementInputs>
    CUTLASS_DEVICE auto
    visit(MatrixCoord const& tileOffset,
          MatrixCoord const& tileShape,
          Layout const& layoutTileInUb,
          int32_t calCount,
          AscendC::LocalTensor<ElementInputs> const&... inputs) {
      constexpr int Rm1 = sizeof...(Ops) - 1;
      
      return tla::detail::tapply(EdgeTuple{}, callbacks_tuple, ubCompute,
        // Visit the first R-1 ops in topological order
        [&] (auto&& edge_seq, auto& callbacks, auto& ub_compute) {
          // 执行当前操作
          auto output = callbacks.visit(tileOffset, tileShape, layoutTileInUb, calCount, inputs...);
          
          // 类型转换到ElementCompute
          using ElementOutput = typename tla::remove_cvref_t<decltype(output)>::Element;
          if constexpr (!std::is_same_v<ElementOutput, ElementCompute>) {
            using ConvertOutput = NumericArrayConverter<ElementCompute, ElementOutput, 0, RoundStyle>;
            ConvertOutput{}(ub_compute, output, calCount);
            return ub_compute;
          } else {
            AscendC::DataCopy(ub_compute, output, calCount);
            return ub_compute;
          }
        },
        // Visit the last op
        [&] (auto const&... compute_inputs) {
          return tla::get<Rm1>(callbacks_tuple).visit(tileOffset, tileShape, layoutTileInUb, calCount, compute_inputs...);
        },
        // Transform to visit R-1 ops, apply to visit last op
        tla::make_seq<Rm1>{}
      );
    }
  };

  // Callbacks factory
  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    Arch::Resource<ArchTag> &resource,
    uint32_t& UB_offset,
    GemmCoord const& blockShapeMNK,
    GemmCoord const& blockCoordMNK,
    Layout const& layoutBlockC,
    MatrixCoord const& subblockShape,
    MatrixCoord const& subblockCoord,
    AscendC::GlobalTensor<ElementCompute> const& gmSubblockC,
    Layout const& layoutSubblockC,
    uint32_t eventId
  ) {
    // 申请临时compute UB空间
    AscendC::LocalTensor<ElementCompute> ubCompute[sizeof...(Ops) - 1];
    for (int i = 0; i < sizeof...(Ops) - 1; ++i) {
      ubCompute[i] = resource.ubBuf.template GetBufferByByte<ElementCompute>(UB_offset);
      UB_offset += COMPUTE_LENGTH * sizeof(ElementCompute);
    }
    
    auto base_callbacks = VisitorImpl<Ops...>::
      get_callbacks(
        resource, UB_offset, blockShapeMNK, blockCoordMNK,
        layoutBlockC, subblockShape, subblockCoord,
        gmSubblockC, layoutSubblockC, eventId
      );
    
    return Callbacks<decltype(base_callbacks)>(
      tla::move(base_callbacks), ubCompute
    );
  }
};


template<
  class ElementCompute,
  class EdgeTuple,
  class... Ops
>
using TopologicalVisitor = TopologicalVisitor<ElementCompute, EdgeTuple, Ops...>;

} // namespace Catlass::Epilogue::Fusion
```

能力
- **拓扑排序执行**: 根据EdgeTuple定义的依赖关系，按拓扑顺序执行多个操作符
- **类型转换支持**: 内置ElementCompute类型转换，支持不同精度间的自动转换
- **复杂数据流图**: 支持任意复杂的DAG(有向无环图)操作符组合
- **编译期优化**: 通过EdgeTuple在编译期确定执行顺序，避免运行期依赖解析

设计动机
- **可扩展性**: 支持任意复杂的操作符图，满足NPU上各种融合需求

### 1.5 VisitorCompute

```cpp
template <
    typename T,
    typename S,
    int N,
    AscendC::RoundMode Round = AscendC::RoundMode::ROUND_EVEN
>
struct NumericArrayConverter {
    using result_type = AscendC::LocalTensor<T>;
    using source_type = AscendC::LocalTensor<S>;
    static constexpr AscendC::RoundMode round_style = Round;
    
    CATLASS_DEVICE
    static void convert(result_type& dst, source_type const& src, uint32_t compute_length) {
        // 使用AscendC::Cast进行向量级类型转换
        AscendC::Cast(dst, src, Round, compute_length);
    }
    
    CATLASS_DEVICE
    void operator()(result_type& dst, source_type const& src, uint32_t compute_length) const {
        convert(dst, src, compute_length);
    }
};
```

```cpp
// 对标CUTLASS的plus - 支持可变元
template <typename T>
struct Plus {
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        int32_t compute_length
    ) const {
        // 二元加法：dst = src0 + src1
        AscendC::Add(dst, src0, src1, compute_length);
    }
    
    // 支持N元输入的变参版本
    template <typename... Ts>
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        Ts const&... rest,
        int32_t compute_length
    ) const {
        // 先处理前两个输入
        (*this)(dst, src0, src1, compute_length);
        // 依次累加剩余输入：dst = ((src0+src1)+rest0)+...
        (AscendC::Add(dst, dst, rest, compute_length), ...);
    }
};
```
```cpp

namespace Catlass::Epilogue::Fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// N-nary Elementwise Compute Operation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  template <class> class ComputeFn,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  int NumInputs,
  int ComputeLength,
  class = void
>
struct VisitorCompute : VisitorImpl<> {

  using VisitorImpl<>::VisitorImpl;

  struct Callbacks : EmptyCallbacks {
    // UB缓冲区引用
    AscendC::LocalTensor<ElementOutput> ubOut;
    AscendC::LocalTensor<ElementCompute> ubOutCompute;
    AscendC::LocalTensor<ElementCompute> ubScratch;
    AscendC::LocalTensor<ElementCompute> ubConverted[NumInputs];
    
    // 存储get_callbacks的入参
    GemmCoord blockShapeMNK;
    GemmCoord blockCoordMNK;
    Layout layoutBlockC;
    MatrixCoord subblockShape;
    MatrixCoord subblockCoord;
    AscendC::GlobalTensor<ElementCompute> gmSubblockC;
    Layout layoutSubblockC;
    uint32_t eventId;
    
    // 构造函数
    CATLASS_DEVICE
    Callbacks(
        AscendC::LocalTensor<ElementOutput> ubOut_,
        AscendC::LocalTensor<ElementCompute> ubOutCompute_,
        AscendC::LocalTensor<ElementCompute> ubConverted_[],
        GemmCoord const& blockShapeMNK_,
        GemmCoord const& blockCoordMNK_,
        Layout const& layoutBlockC_,
        MatrixCoord const& subblockShape_,
        MatrixCoord const& subblockCoord_,
        AscendC::GlobalTensor<ElementCompute> const& gmSubblockC_,
        Layout const& layoutSubblockC_,
        uint32_t eventId_
    ) : ubOut(ubOut_), ubOutCompute(ubOutCompute_),
        blockShapeMNK(blockShapeMNK_), blockCoordMNK(blockCoordMNK_), 
        layoutBlockC(layoutBlockC_), subblockShape(subblockShape_), 
        subblockCoord(subblockCoord_), gmSubblockC(gmSubblockC_), 
        layoutSubblockC(layoutSubblockC_), eventId(eventId_) {
        for (int i = 0; i < NumInputs; ++i) {
            ubConverted[i] = ubConverted_[i];
        }
    }
    
    template <typename... ElementInputs>
    CATLASS_DEVICE AscendC::LocalTensor<ElementOutput> const& visit(
        MatrixCoord const& tileOffset,
        MatrixCoord const& tileShape,
        Layout const& layoutTileInUb,
        int32_t calCount,
        AscendC::LocalTensor<ElementInputs> const&... inputs
    ) {
        static_assert(sizeof...(ElementInputs) == NumInputs, "Input count mismatch");
        
        // 现在可以使用存储的成员变量
        // 例如：根据blockShapeMNK、blockCoordMNK等信息进行相应的处理
        
        // 将输入打包成tuple进行处理
        auto inputs_tuple = tla::make_tuple(inputs...);
        
        // 使用索引序列来访问对应的UB缓冲区，优化类型转换
        auto cvt_tuple = tla::transform_apply(
            tla::make_index_sequence<NumInputs>{},
            inputs_tuple,
            [&] (auto idx, auto const& in) -> AscendC::LocalTensor<ElementCompute> const& {
                using ElementInput = typename tla::remove_cvref_t<decltype(in)>::Element;
                
                // 编译期判断：如果类型相同，直接返回输入，不执行转换
                if constexpr (std::is_same_v<ElementInput, ElementCompute>) {
                    return in;  // 直接返回，零拷贝
                } else {
                    // 类型不同时才使用预分配的转换缓冲区
                    auto& tmp = ubConverted[idx];
                    using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, 0, RoundStyle>;
                    ConvertInput{}(tmp, in, calCount);
                    return tmp;
                }
            }
        );

        // 执行计算
        using ComputeOutput = ComputeFn<ElementCompute>;
        using ConvertOutput = NumericArrayConverter<ElementOutput, ElementCompute, 0, RoundStyle>;
        ComputeOutput compute_output{};
        ConvertOutput convert_output{};

        tla::apply(cvt_tuple, [&](auto const&... cvt_inputs){
            compute_output(ubOutCompute, cvt_inputs..., calCount);

            // 输出类型转换也进行优化
            if constexpr (!std::is_same_v<ElementOutput, ElementCompute>) {
                convert_output(ubOut, ubOutCompute, calCount);
            } else {
                AscendC::DataCopy(ubOut, ubOutCompute, calCount);
            }
        });
        return ubOut;
    }
  };

  template <class ProblemShape>
  CATLASS_DEVICE auto
  get_callbacks(
    Arch::Resource<ArchTag> &resource,
    uint32_t& UB_offset,
    GemmCoord const& blockShapeMNK,
    GemmCoord const& blockCoordMNK,
    Layout const& layoutBlockC,
    MatrixCoord const& subblockShape,
    MatrixCoord const& subblockCoord,
    AscendC::GlobalTensor<ElementCompute> const& gmSubblockC,
    Layout const& layoutSubblockC,
    uint32_t eventId
  ) {
    // 申请基础UB缓冲区
    auto ubOut = resource.ubBuf.template GetBufferByByte<ElementOutput>(UB_offset);
    UB_offset += COMPUTE_LENGTH * sizeof(ElementOutput);
    
    auto ubOutCompute = resource.ubBuf.template GetBufferByByte<ElementCompute>(UB_offset);
    UB_offset += COMPUTE_LENGTH * sizeof(ElementCompute);
    
    // 预分配所有可能的转换缓冲区
    AscendC::LocalTensor<ElementCompute> ubConverted[NumInputs];
    for (int i = 0; i < NumInputs; ++i) {
        ubConverted[i] = resource.ubBuf.template GetBufferByByte<ElementCompute>(UB_offset);
        UB_offset += COMPUTE_LENGTH * sizeof(ElementCompute);
    }
    
    return Callbacks(ubOut, ubOutCompute, ubConverted,
                    blockShapeMNK, blockCoordMNK, layoutBlockC,
                    subblockShape, subblockCoord, gmSubblockC,
                    layoutSubblockC, eventId);
  }
};

} // namespace Catlass::Epilogue::Fusion
```

能力描述
- **N 元元素级计算**: 支持多输入张量的元素级融合计算
- **类型转换**: 自动处理输入/输出类型转换

### 1.6 VisitorAuxLoad

```cpp
namespace Catlass::Epilogue::Fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Load Operations for NPU
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ThreadMap,
  class Element,
  class Layout
>
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
  
  using ubStorage = tla::tuple<>;  // 无UB存储需求

  template <class ProblemShape>
  CATLASS_HOST_DEVICE static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return Params(args.ptr_aux, args.null_default, args.layout);
  }

  template <class ProblemShape>
  CATLASS_HOST_DEVICE static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  CATLASS_HOST_DEVICE static Catlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return Catlass::Status::kSuccess;
  }

  template <class ProblemShape>
  CATLASS_HOST_DEVICE static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return args.ptr_aux != nullptr;
  }


  struct Callbacks : EmptyCallbacks {
    // UB缓冲区引用
    AscendC::LocalTensor<Element> ubAux;
    
    // 存储get_callbacks的入参
    GemmCoord blockShapeMNK;
    GemmCoord blockCoordMNK;
    Layout layoutBlockC;
    MatrixCoord subblockShape;
    MatrixCoord subblockCoord;
    AscendC::GlobalTensor<Element> gmSubblockAux;
    Layout layoutSubblockAux;
    uint32_t eventId;
    Params const* params_ptr;
    
    // 构造函数
    CATLASS_DEVICE
    Callbacks(
        AscendC::LocalTensor<Element> ubAux_,
        GemmCoord const& blockShapeMNK_,
        GemmCoord const& blockCoordMNK_,
        Layout const& layoutBlockC_,
        MatrixCoord const& subblockShape_,
        MatrixCoord const& subblockCoord_,
        AscendC::GlobalTensor<Element> const& gmSubblockAux_,
        Layout const& layoutSubblockAux_,
        uint32_t eventId_,
        Params const* params_ptr_
    ) : ubAux(ubAux_), blockShapeMNK(blockShapeMNK_), blockCoordMNK(blockCoordMNK_), 
        layoutBlockC(layoutBlockC_), subblockShape(subblockShape_), 
        subblockCoord(subblockCoord_), gmSubblockAux(gmSubblockAux_), 
        layoutSubblockAux(layoutSubblockAux_), eventId(eventId_), params_ptr(params_ptr_) {}
    
    CATLASS_DEVICE AscendC::LocalTensor<Element> const& visit(
        MatrixCoord const& tileOffset,
        MatrixCoord const& tileShape,
        Layout const& layoutTileInUb,
        int32_t calCount
    ) {
        // 从全局内存加载数据到UB
        
        
        if (params_ptr->ptr_aux != nullptr) {
            // 构建GM tensor
            AscendC::GlobalTensor<Element> gmAux;
            gmAux.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params_ptr->ptr_aux));
            
            // 计算GM偏移
            auto gmOffset = params_ptr->layout.GetOffset(tileOffset);
            auto gmTileAux = gmAux[gmOffset];
            
            // 从GM加载到UB
            AscendC::Load(ubAux, gmTileAux, calCount);
        } else {
            // 使用默认值填充
            AscendC::Duplicate(ubAux, params_ptr->null_default, calCount);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId_);
        return ubAux;
    }
  };

  template <class ProblemShape>
  CATLASS_DEVICE auto
  get_callbacks(
    Arch::Resource<ArchTag> &resource,
    uint32_t& UB_offset,
    GemmCoord const& blockShapeMNK,
    GemmCoord const& blockCoordMNK,
    Layout const& layoutBlockC,
    MatrixCoord const& subblockShape,
    MatrixCoord const& subblockCoord,
    AscendC::GlobalTensor<Element> const& gmSubblockAux,
    Layout const& layoutSubblockAux,
    uint32_t eventId
  ) {
    // 申请UB缓冲区
    ubAux = resource.ubBuf.template GetBufferByByte<Element>(UB_offset);
    UB_offset += COMPUTE_LENGTH * sizeof(Element);
    
    return Callbacks(ubAux, blockShapeMNK, blockCoordMNK, layoutBlockC,
                    subblockShape, subblockCoord, gmSubblockAux,
                    layoutSubblockAux, eventId, params_ptr);
  }
};

} // namespace Catlass::Epilogue::Fusion


// AccLoad：累加源 C 的便捷别名（完全等价于 VisitorAuxLoad）
using VisitorAccLoad = VisitorAuxLoad;
```
能力描述

- **NPU向量化加载**: 支持从全局内存到UB的数据加载，适配NPU的存储层次结构

### 1.7 VisitorAuxStore

```cpp
namespace Catlass::Epilogue::Fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Store Operations for NPU
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ThreadMap,
  class Element,
  FloatRoundStyle RoundStyle,
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
  using ubStorage = tla::tuple<>;  // 无UB存储需求

  template <class ProblemShape>
  CATLASS_HOST_DEVICE static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return Params(args.ptr_aux, args.null_default, args.layout);
  }

  template <class ProblemShape>
  CATLASS_HOST_DEVICE static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  CATLASS_HOST_DEVICE static Catlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return Catlass::Status::kSuccess;
  }

  template <class ProblemShape>
  CATLASS_HOST_DEVICE static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return args.ptr_aux != nullptr;
  }


  struct Callbacks : EmptyCallbacks {
    // UB缓冲区引用
    AscendC::LocalTensor<Element> ubAux;
    
    // 存储get_callbacks的入参
    GemmCoord blockShapeMNK;
    GemmCoord blockCoordMNK;
    Layout layoutBlockC;
    MatrixCoord subblockShape;
    MatrixCoord subblockCoord;
    AscendC::GlobalTensor<Element> gmSubblockAux;
    Layout layoutSubblockAux;
    uint32_t eventId;
    Params const* params_ptr;
    
    // 构造函数
    CATLASS_DEVICE
    Callbacks(
        AscendC::LocalTensor<Element> ubAux_,
        GemmCoord const& blockShapeMNK_,
        GemmCoord const& blockCoordMNK_,
        Layout const& layoutBlockC_,
        MatrixCoord const& subblockShape_,
        MatrixCoord const& subblockCoord_,
        AscendC::GlobalTensor<Element> const& gmSubblockAux_,
        Layout const& layoutSubblockAux_,
        uint32_t eventId_,
        Params const* params_ptr_
    ) : ubAux(ubAux_), blockShapeMNK(blockShapeMNK_), blockCoordMNK(blockCoordMNK_), 
        layoutBlockC(layoutBlockC_), subblockShape(subblockShape_), 
        subblockCoord(subblockCoord_), gmSubblockAux(gmSubblockAux_), 
        layoutSubblockAux(layoutSubblockAux_), eventId(eventId_), params_ptr(params_ptr_) {}
    
    template <typename ElementInput>
    CATLASS_DEVICE void visit(
        MatrixCoord const& tileOffset,
        MatrixCoord const& tileShape,
        Layout const& layoutTileInUb,
        int32_t calCount,
        AscendC::LocalTensor<ElementInput> const& input
    ) {
        // 类型转换（如果需要）
        if constexpr (!std::is_same_v<ElementInput, Element>) {
            using ConvertInput = NumericArrayConverter<Element, ElementInput, 0, RoundStyle>;
            ConvertInput{}(ubAux, input, calCount);
        } else {
            AscendC::DataCopy(ubAux, input, calCount);
        }
        
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId_);
        
        // 从UB存储数据到全局内存
        if (params_ptr->ptr_aux != nullptr) {
            // 构建GM tensor
            AscendC::GlobalTensor<Element> gmAux;
            gmAux.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params_ptr->ptr_aux));
            
            // 计算GM偏移
            auto gmOffset = params_ptr->layout.GetOffset(tileOffset);
            auto gmTileAux = gmAux[gmOffset];
            
            // 从UB存储到GM
            AscendC::Store(gmTileAux, ubAux, calCount);
        }
    }
  };

  template <class ProblemShape>
  CATLASS_DEVICE auto
  get_callbacks(
    Arch::Resource<ArchTag> &resource,
    uint32_t& UB_offset,
    GemmCoord const& blockShapeMNK,
    GemmCoord const& blockCoordMNK,
    Layout const& layoutBlockC,
    MatrixCoord const& subblockShape,
    MatrixCoord const& subblockCoord,
    AscendC::GlobalTensor<Element> const& gmSubblockAux,
    Layout const& layoutSubblockAux,
    uint32_t eventId
  ) {
    // 申请UB缓冲区
    auto ubAux = resource.ubBuf.template GetBufferByByte<Element>(UB_offset);
    UB_offset += COMPUTE_LENGTH * sizeof(Element);
    
    return Callbacks(ubAux, blockShapeMNK, blockCoordMNK, layoutBlockC,
                    subblockShape, subblockCoord, gmSubblockAux,
                    layoutSubblockAux, eventId, params_ptr);
  }
};

} // namespace Catlass::Epilogue::Fusion
```

能力描述

- **NPU向量化存储**: 支持从UB到全局内存的数据存储，适配NPU的存储层次结构


## 2. 示例

```cpp
// 用户定义 EVT：D = C + RowBroadcast(X)
using EVT = TreeVisitor<
    VisitorAuxStore<ElementD, LayoutD, COMPUTE_LENGTH>,      // 根：写回 D
    TreeVisitor<VisitorCompute<Plus, ElementD, ElementCompute, 2, COMPUTE_LENGTH>,                   // 父：计算 D = C + X
                VisitorAccLoad<ElementC, LayoutC, COMPUTE_LENGTH>,       // 子1：提供矩阵C
                VisitorAuxLoad<ElementX, LayoutX, COMPUTE_LENGTH>>   // 子2：提供矩阵X 
>;

std::vector<fp16_t> hostA(lenA);
std::vector<fp16_t> hostB(lenB);
std::vector<fp16_t> hostX(lenX);

// 分配设备内存
uint8_t *deviceA, *deviceB, *deviceC, *deviceD;
ACL_CHECK(aclrtMalloc(&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
ACL_CHECK(aclrtMalloc(&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
ACL_CHECK(aclrtMalloc(&deviceC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
ACL_CHECK(aclrtMalloc(&deviceD, sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

// 定义布局
using LayoutC = layout::RowMajor;
using LayoutD = layout::RowMajor;
using LayoutX = layout::RowMajor;
LayoutC layoutC{m, n};
LayoutD layoutD{m, n};
LayoutX layoutX{m, n};

// 准备EVT的Arguments
typename EVT::Arguments evt_args{
    // VisitorAuxStore的Arguments
    {reinterpret_cast<ElementD*>(deviceD), layoutD},
    // VisitorCompute的Arguments (通常为空)
    {
      {},
      // VisitorAccLoad的Arguments
      {reinterpret_cast<ElementC*>(deviceC), ElementC(0), layoutC},
      // VisitorRowBroadcast的Arguments
      {reinterpret_cast<ElementX*>(deviceX), ElementX(0), layoutX}
    },
};

using MyBlockEpilogue = BlockEpilogue<
  EpilogueWithVisitorCallbacks,
  CType,
  COMPUTE_LENGTH,
  ElementCompute,
  EVT
>;

using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using MatmulKernel = Gemm::Kernel::MatmulEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;

typename MatmulKernel::Arguments arguments{options.problemShape, sizeof(half), deviceA, deviceB, deviceD, evt_args};

//matmul_epilogue
BlockEpilogue blockEpilogue(resource, params.evt_args);
}
```

## 3. 运行

以BlockEpilogue的形式传入matmul_epilogue。

```C++
namespace Catlass::Epilogue::Block {

template <
    class CType_,
    uint32_t COMPUTE_LENGTH_,
    ElementCompute,
    class FusionCallbacks_
>
class BlockEpilogue <
    EpilogueWithVisitorCallbacks,
    CType_,
    ElementCompute,
    COMPUTE_LENGTH_,
    FusionCallbacks_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueWithVisitorCallbacks;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    using LayoutComputeInUb = layout::RowMajor;

    // Epilogue params definition
    struct Params {
        CATLASS_HOST_DEVICE
        Params() {}
    };

    CATLASS_DEVICE
    BlockEpilogue(const Params &params_callbacks, Arch::Resource<ArchTag> &resource) : fusion_callbacks(params_callbacks)
    {
        // 预热两个事件通道，构建双缓冲
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        // 确保两个事件均已完成
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

    CATLASS_DEVICE
    void operator() (
        Arch::Resource<ArchTag> &resource,
        GemmCoord const &blockShapeMNK,
        GemmCoord const &blockCoordMNK,
        GemmCoord const &actualBlockShapeMNK,
        AscendC::GlobalTensor<ElementCompute> const &gmBlockC,
        LayoutC const &layoutBlockC
    )
    {   
        // 分块处理
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        
        // 子块划分（适配NPU subblock模型）
        MatrixCoord subblockShape{
            CeilDiv(actualBlockShape.row(), static_cast<uint32_t>(AscendC::GetSubBlockNum())),
            actualBlockShape.column()
        };
        MatrixCoord subblockCoord{ AscendC::GetSubBlockIdx(), 0 };
        MatrixCoord actualSubblockShape = MatrixCoord::Min(
            subblockShape, actualBlockShape - subblockCoord * subblockShape);
        MatrixCoord subblockOffset = subblockCoord * subblockShape;
        
        // 将当前 subblock 沿行方向一分为二，构造 A/B 两套回调及不同事件，实现双缓冲
        uint32_t rows = actualSubblockShape.row();
        uint32_t cols = actualSubblockShape.column();
        uint32_t rowsA = rows / 2;
        uint32_t rowsB = rows - rowsA;

        MatrixCoord halfShapeA{rowsA, cols};
        MatrixCoord halfShapeB{rowsB, cols};

        // GM 半区偏移
        auto gmSubblockA = gmBlockC[layoutBlockC.GetOffset(subblockOffset + MatrixCoord{0, 0})];
        auto gmSubblockB = gmBlockC[layoutBlockC.GetOffset(subblockOffset + MatrixCoord{rowsA, 0})];
        auto layoutSubblockA = layoutBlockC.GetTileLayout(halfShapeA);
        auto layoutSubblockB = layoutBlockC.GetTileLayout(halfShapeB);

        // 独立 UB 空间分配：先为 A，再为 B
        uint32_t UB_offset = 0;
        auto callbacksA = fusion_callbacks.get_callbacks(
            resource,
            UB_offset,
            blockShapeMNK,
            blockCoordMNK,
            layoutBlockC,
            halfShapeA,
            subblockCoord,
            gmSubblockA,
            layoutSubblockA,
            EVENT_ID0
        );
        auto callbacksB = fusion_callbacks.get_callbacks(
            resource,
            UB_offset,
            blockShapeMNK,
            blockCoordMNK,
            layoutBlockC,
            halfShapeB,
            subblockCoord,
            gmSubblockB,
            layoutSubblockB,
            EVENT_ID1
        );

        // 开始 epilogue 生命周期（双实例）
        callbacksA.begin_epilogue();
        callbacksB.begin_epilogue();

        // 行指针分别在两个半区推进；列循环内交替/贴近发射 visit，减少 scalar 延迟
        for (uint32_t rA = 0, rB = 0; rA < rowsA || rB < rowsB; ) {
            if (rA < rowsA) callbacksA.begin_row(rA);
            if (rB < rowsB) callbacksB.begin_row(rowsA + rB);

            for (uint32_t cStart = 0; cStart < cols;) {
                uint32_t remainCols = cols - cStart;
                uint32_t tileCols = (remainCols < COMPUTE_LENGTH) ? remainCols : COMPUTE_LENGTH;
                uint32_t rowsCap = (tileCols == 0) ? 0 : (COMPUTE_LENGTH / tileCols);
                if (rowsCap == 0) rowsCap = 1;

                uint32_t rowsRemainA = (rA < rowsA) ? (rowsA - rA) : 0;
                uint32_t rowsRemainB = (rB < rowsB) ? (rowsB - rB) : 0;
                uint32_t rowsThisA = (rowsRemainA < rowsCap) ? rowsRemainA : rowsCap;
                uint32_t rowsThisB = (rowsRemainB < rowsCap) ? rowsRemainB : rowsCap;

                // A 半区 tile
                if (rowsThisA) {
                    MatrixCoord tileShapeA{rowsThisA, tileCols};
                    MatrixCoord tileOffsetA{rA, cStart};
                    auto layoutTileInUbA = layout::RowMajor::template MakeLayoutInUb<ElementCompute>(tileShapeA);
                    callbacksA.visit(tileOffsetA, tileShapeA, layoutTileInUbA, rowsThisA * tileCols);
                }

                // B 半区 tile，紧随 A 发射，贴近 visit
                if (rowsThisB) {
                    MatrixCoord tileShapeB{rowsThisB, tileCols};
                    MatrixCoord tileOffsetB{rowsA + rB, cStart};
                    auto layoutTileInUbB = layout::RowMajor::template MakeLayoutInUb<ElementCompute>(tileShapeB);
                    callbacksB.visit(tileOffsetB, tileShapeB, layoutTileInUbB, rowsThisB * tileCols);
                }

                cStart += tileCols;
            }

            if (rA < rowsA) callbacksA.end_row(rA);
            if (rB < rowsB) callbacksB.end_row(rowsA + rB);

            // 行推进：按所有列合并的行上限推进两个半区
            uint32_t rowsCapAllCols = (cols == 0) ? 0 : (COMPUTE_LENGTH / ((cols < COMPUTE_LENGTH) ? cols : COMPUTE_LENGTH));
            if (rowsCapAllCols == 0) rowsCapAllCols = 1;
            if (rA < rowsA) {
                uint32_t advA = ((rowsA - rA) < rowsCapAllCols) ? (rowsA - rA) : rowsCapAllCols;
                rA += advA;
            }
            if (rB < rowsB) {
                uint32_t advB = ((rowsB - rB) < rowsCapAllCols) ? (rowsB - rB) : rowsCapAllCols;
                rB += advB;
            }
        }

        callbacksA.end_epilogue();
        callbacksB.end_epilogue();
    }

private:
    Params params;
    FusionCallbacks fusion_callbacks;

};

}  // namespace Catlass::Epilogue::Block

```