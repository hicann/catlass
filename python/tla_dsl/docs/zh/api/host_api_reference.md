---
nav_order: 15
---

<!--
手工维护的中文 Host API 参考。
英文源：docs/en/api/host_api_reference.md
（由 python/tla_dsl/tools/generate_host_api_reference.py + 源码英文 docstring 生成）。
英文稿变更后请同步翻译更新本文件。
不要用术语表 / glossary 自动生成本文件。
-->

# TLA DSL Host API 参考

本文档介绍 **TLA DSL 的 Host 侧 API**（通常以 `import catlass.tla as tla` 导入）。
内容覆盖：`@tla.kernel` 装饰器、Host 侧 `@dataclass` 打包、`tla.compile` /
`KernelLauncher` 启动、Host tensor。
环境变量见 [环境变量](../kernel_development/core_concepts/env_vars.md)。Kernel 侧接口见 [Kernel API 参考](kernel_api_reference.md)。

接口说明与调用示例来自各 API 源码 docstring；这些接口均在 Python Host 脚本中、
`@tla.kernel` 函数体**外**调用。

DLPack 接入教程见 [Host Tensor 接入](../kernel_development/core_concepts/tensor_binding.md)。
动态 layout 编程见 [静态与动态 Layout](../kernel_development/core_concepts/layout.md)。

---

## 目录

- [1. 装饰器](#1-装饰器)
- [2. 编译与启动](#2-编译与启动)
  - [2.1 编译](#21-编译)
  - [2.2 启动](#22-启动)
  - [2.3 查看 IR](#23-查看-ir)
- [3. Host Tensor](#3-host-tensor)
  - [3.1 创建与绑定](#31-创建与绑定)
  - [3.2 动态 Layout](#32-动态-layout)

---

## 1. 装饰器

Host 侧 `@tla.kernel` 入口，以及 Host 侧 `@dataclass` 打包。
被装饰的 kernel 函数体在 Host 端不执行。

### `kernel`

**源码：** [`catlass.dsl.kernel`](../../../catlass/dsl.py#L305)

功能说明：

将 Python 函数标注为 TLA Kernel 入口。函数体在 Host 端不执行。
返回 `TlaJitFunction`。调用该对象会返回 `KernelLauncher`（此时不启动）；再调用
launcher 或 `.launch(...)` 才会编译并执行。

函数原型：

```python
tla.kernel(fn: Callable[..., Any] | None = None, *, auto_sync: str | None = None) -> TlaJitFunction | Callable[[Callable[..., Any]], TlaJitFunction]
```

参数说明：

- *`fn`*（`Callable[..., Any] | None`）：被装饰的函数。用 `@tla.kernel` 或
  `@tla.kernel(auto_sync=...)`；只有无法使用装饰器语法时才手写 `tla.kernel(fn)`。
- *`auto_sync`*（`str | None`）：可选。`"v0"` 表示由框架自动插入局部 mutex；
  默认 `None`（同步仍由用户显式控制）。

约束说明：

- 被装饰的函数不能用 Python 的 `async def` 定义。
- `auto_sync` 只能是 `"v0"` 或 `None`。
- Kernel 参数类型：

  | 类别 | 类型 |
  | --- | --- |
  | Tensor | `tla.Tensor` |
  | Python 标量 | `bool` / `int` / `float` |
  | `tla` 标量 | `Bool`、`Int8/16/32/64`、`UInt8/16/32/64`、`Float16/32`、`BFloat16` |
  | 编译期常量 | `tla.Constexpr[...]` |
  | 结构体 | 字段类型属于上表的 `@dataclass` 实例 |

调用示例：

```python
@tla.kernel
def vadd(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(src, dst)

@tla.kernel(auto_sync="v0")
def vadd_auto(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(src, dst)

vadd(tx, ty, options="--npu-arch 3510")(block_num=1)
```

---

### `dataclass`

**源码：** [`dataclasses.dataclass`](../../../catlass/execution_lowering.py#L759)

功能说明：

用 Python 标准库 `@dataclass` 在 Host 侧打包 kernel 入参。可在 Host 上创建实例后传给
`tla.compile` / 启动；也可在 kernel 内构造字段实例。

函数原型：

```python
dataclasses.dataclass(cls: type, *, frozen: bool = False, kw_only: bool = False) -> type
```

参数说明：

- *`frozen`*（`bool`）：为 `True` 时实例不可变。默认 `False`。
- *`kw_only`*（`bool`）：为 `True` 时字段必须按关键字传入。默认 `False`。

约束说明：

- 作为 kernel 入参时，只支持设置 `frozen` / `kw_only`；其它 stdlib 选项
  （如 `slots=True`、`init=False`）会在编译期报错。
- 支持的字段类型：

  | 类别 | 类型 | 约束 |
  | --- | --- | --- |
  | Tensor | `tla.Tensor` | 不支持动态 GM；请改用静态 tensor 字段或顶层 tensor 入参 |
  | Python 标量 | `bool` / `int` / `float` | — |
  | `tla` 标量 | `Bool`、`Int8/16/32/64`、`UInt8/16/32/64`、`Float16/32`、`BFloat16` | — |
  | 编译期常量 | `tla.Constexpr[...]` | 不进入 kernel ABI / IR，且在 kernel 内只读 |

调用示例：

```python
from dataclasses import dataclass
import catlass.tla as tla

@dataclass(frozen=True, kw_only=True)
class TilingData:
    TILE_M: tla.Constexpr[int]
    tiling_int: int
    out: tla.Tensor

@tla.kernel
def struct_arg_kernel(tiling: TilingData) -> None:
    # TILE_M 为编译期常量；tiling_int 为运行时标量。
    ...

tiling = TilingData(TILE_M=128, tiling_int=64, out=tout)
artifact = tla.compile(struct_arg_kernel, tiling, options="--npu-arch 3510")
artifact(tiling, block_num=1)
```

---

## 2. 编译与启动

将装饰后的 kernel 编译为设备二进制并启动。同一份二进制需要多次启动时，用
`tla.compile` 再 `artifact(...)` / `.launch(...)`；单次或少量启动可直接调用
`@tla.kernel` 得到 `KernelLauncher` 再启动。缓存 / 架构 / IR dump 等非函数参数见
[环境变量](../kernel_development/core_concepts/env_vars.md)。

### 2.1 编译

生成设备二进制。日常入口是 `tla.compile`；
`TlaJitFunction.compile` 是装饰后函数上的底层辅助接口。

#### `compile`

**源码：** [`catlass.base_dsl.compiler.CompileCallable.__call__`](../../../catlass/base_dsl/compiler.py#L46)

功能说明：

编译 `@tla.kernel` 函数，返回包装了 `TlaKernelArtifact` 的可调用执行器。
这是公开的 `tla.compile` 入口。调用返回的执行器即可启动
（`artifact(*tensors, block_num=...)`）。适合先编译一次，再对同一份二进制多次启动。

函数原型：

```python
tla.compile(func: Any, *args: Any, **kwargs: Any) -> TlaJitExecutor
```

参数说明：

- *`func`*（`TlaJitFunction`）：被 `@tla.kernel` 装饰的函数。必填。
- *`args`*（`Any`）：作为编译类型样本的 Host tensor / 标量 / `@dataclass` 实例
  （如 `from_dlpack` 或 `make_fake_tensor` 的返回值）。
- *`kwargs`*：Host 编译参数。用 `options="--npu-arch 3510"` 指定芯片名。
  缓存 / IR dump / 强制重编译由 `CATLASS_DSL_*` 环境变量控制。

约束说明：

- `func` 必须是 `@tla.kernel` 得到的 `TlaJitFunction`。
- `args` 只作编译期类型样本，不必绑定 NPU 缓冲（`make_fake_tensor` 合法）。
- 用 `options="--npu-arch 3510"` 指定芯片名；不支持的取值在编译时报错。
- `block_num` / `stream` 等启动参数写在返回执行器上
  （`artifact(...)` / `TlaJitExecutor.launch`），而不是 `tla.compile`。

调用示例：

```python
artifact = tla.compile(vadd, tx, ty, options="--npu-arch 3510")
artifact(tx, ty, block_num=1)
artifact(tx, ty, block_num=1)  # 同一份二进制再次启动
```

---

#### `TlaJitFunction.compile`

**源码：** [`catlass.dsl.TlaJitFunction.compile`](../../../catlass/dsl.py#L195)

功能说明：

编译当前 `@tla.kernel` 函数并返回 `TlaKernelArtifact`。
日常 Host 入口是 `tla.compile(fn, *args, options=...)`；只有已持有
`TlaJitFunction` 且需要原始 `TlaKernelArtifact` 时才调用 `.compile()`。

函数原型：

```python
TlaJitFunction.compile(*, type_args: Sequence[Any] | None = None, **kwargs: Any) -> TlaKernelArtifact
```

参数说明：

- *`type_args`*（`Sequence[Any] | None`）：作为编译类型样本的 Host tensor / 标量。
  可选，默认 `None`（不做张量特化）。
- *`kwargs`*：Host 编译参数。用 `options="--npu-arch 3510"` 指定芯片名。
  缓存 / IR dump 由 `CATLASS_DSL_*` 环境变量控制。

约束说明：

- `type_args` 只作编译期类型样本，不必绑定 NPU 缓冲（`make_fake_tensor` 合法）。
- 用 `options="--npu-arch 3510"` 指定芯片名；不支持的取值在编译时报错。

调用示例：

```python
artifact = my_kernel.compile(
    type_args=[tx, ty],
    options="--npu-arch 3510",
)
```

---

### 2.2 启动

在 NPU 上运行已编译的 kernel。经 `tla.compile` 后，调用执行器或
`TlaJitExecutor.launch`；经调用 `@tla.kernel` 后，使用 `KernelLauncher.launch`
（或直接调用 launcher）。

#### `TlaJitExecutor.launch`

**源码：** [`catlass.base_dsl.jit_executor.TlaJitExecutor.launch`](../../../catlass/base_dsl/jit_executor.py#L99)

功能说明：

在 NPU 上启动经 `tla.compile` 得到的已编译 kernel，传入运行时入参与启动参数
（如 `block_num`、`stream`）。

函数原型：

```python
TlaJitExecutor.launch(*launch_args: Any, *, block_num: int | None = None, args: Sequence[Any] | None = None, **kwargs: Any) -> TlaExecutionResult
```

参数说明：

- *`launch_args`*（`Any`）：位置形式的运行时 kernel 入参，与 `@tla.kernel`
  签名对应（已绑定的 Host tensor、标量或 `@dataclass` 实例）。与 `args=` 互斥。
- *`block_num`*（`int | None`）：启动的 block 数。可选，默认 `1`；传入时须为 `int`。
- *`args`*（`Sequence[Any] | None`）：显式运行时实参序列。可选，默认 `None`。
  不能与非空的 `*launch_args` 同时使用。
- *`stream`*（`Any`，经 `**kwargs`）：可选 ACL stream 句柄（常为 `int`）。
  省略时优先用 `torch.npu.current_stream`；否则须显式传 `stream=`，
  或依赖 `CATLASS_DSL_NPU_DEVICE`。

约束说明：

- `artifact(...)` 与 `.launch(...)` 共用同一套运行时规则。
- `*launch_args` 与 `args=` 不能同时非空（`TlaUnsupportedAbiError`）。
- tensor 启动实参须为已绑定的 NPU 缓冲（`from_dlpack`）；
  `make_fake_tensor` 仅用于编译样本。
- `block_num` 须为 `int`（默认 `1`）。

调用示例：

```python
artifact = tla.compile(vadd, tx, ty, options="--npu-arch 3510")
artifact(tx, ty, block_num=1)
# 等价于：
artifact.launch(tx, ty, block_num=1)
# 或显式传 args：
artifact.launch(args=(tx, ty), block_num=1)
```

---

#### `KernelLauncher.launch`

**源码：** [`catlass.catlass_dsl.tla.KernelLauncher.launch`](../../../catlass/catlass_dsl/tla.py#L75)

功能说明：

在 NPU 上启动 `@tla.kernel`。先调用被装饰的 kernel 得到本对象
（`launcher = my_kernel(*tensors, options=...)`，此时不启动），再调用
`launcher.launch(...)` 或直接调用 launcher（`launcher(block_num=...)`，等价于
`.launch(args=..., **kwargs)`）。尚无缓存产物，或 runtime 选项与上次编译不一致时，
本方法会先编译再启动；runtime 选项不变且已有缓存产物时，不重新编译，直接启动。
需要固定一份编译产物并多次启动、且编译与启动分离时，使用 `tla.compile` 返回的
`TlaJitExecutor`。

函数原型：

```python
KernelLauncher.launch(*, block_num: int | None = None, type_args: Sequence[Any] | None = None, args: Sequence[Any] | None = None, **kwargs: Any) -> TlaExecutionResult
```

参数说明：

- *`block_num`*（`int | None`）：启动的 block 数。可选，默认 `1`（也可来自
  构造 launcher 时保存的 kwargs）。传入时须为 `int`。
- *`type_args`*（`Sequence[Any] | None`）：编译期类型样本。可选；省略时由
  `args` / 第一次调用时的 launch args 推断。
- *`args`*（`Sequence[Any] | None`）：显式运行时实参序列。可选。不能与
  launcher 上已由第一次调用（`my_kernel(*tensors)`）保存的 launch args 同时使用。
- *`stream`*（`Any`，经 `**kwargs`）：可选 ACL stream 句柄（常为 `int`）。
  省略时优先用 `torch.npu.current_stream`；否则须显式传 `stream=`，
  或依赖 `CATLASS_DSL_NPU_DEVICE`。
- *`options`*（`str`，经 `**kwargs`）：芯片名，如 `options="--npu-arch 3510"`。
  可在第一次调用或本次传入；本调用 kwargs 覆盖构造 launcher 时保存的值。

约束说明：

- 调用 `@tla.kernel` 返回 `KernelLauncher`，不会启动；本方法（或调用 launcher）
  在尚无缓存产物或 runtime 选项变化时编译并启动，runtime 选项不变且已有缓存产物时
  不重新编译、直接启动。
- 第一次调用已传入 tensor（`my_kernel(tx, ty)`）时，第二次调用 / `.launch`
  只应传 `block_num` 等启动 kwargs。
- 对同一 `KernelLauncher` 重复 `.launch` 时，runtime 选项不变则复用已编译产物，
  不重新编译。需要显式分离编译与启动时，使用 `tla.compile`。
- launcher 已持有 launch args 时，不能再传 `args=`（`TlaUnsupportedAbiError`）。
- tensor 启动实参须为已绑定的 NPU 缓冲（`from_dlpack`）；
  `make_fake_tensor` 仅用于编译 / 类型样本。
- `block_num` 须为 `int`（默认 `1`）。

调用示例：

```python
@tla.kernel
def vadd(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(src, dst)

vadd(tx, ty, options="--npu-arch 3510")(block_num=1)
# 或：
launcher = vadd(tx, ty, options="--npu-arch 3510")
launcher.launch(block_num=1)
# 第一次未传 tensor 时，可在 .launch 里传 args：
vadd(options="--npu-arch 3510").launch(args=(tx, ty), block_num=1)
```

---

### 2.3 查看 IR

导出前端 TLA IR，不生成设备二进制，也不启动。

#### `TlaJitFunction.dump_mlir`

**源码：** [`catlass.dsl.TlaJitFunction.dump_mlir`](../../../catlass/dsl.py#L253)

功能说明：

返回该 kernel 的 TLA IR（`tlair`）MLIR 文本。不编译设备二进制，也不 launch。

函数原型：

```python
TlaJitFunction.dump_mlir(*, type_args: Sequence[Any] | None = None) -> str
```

参数说明：

- *`type_args`*（`Sequence[Any] | None`）：类型样本，用法同 `.compile()`。
  可选，默认 `None`。

约束说明：

- `type_args` 规则与 `.compile()` 相同。
- 返回的是前端 TLA IR（`tlair`），不是 `TlaKernelArtifact.lowered_llvm` 中的
  HIVM/LLVM 形式。

调用示例：

```python
text = my_kernel.dump_mlir(type_args=[fa, fb])
print(text[:500])
```

---

## 3. Host Tensor

构造 Host 侧 `tla.Tensor`，并可将静态 layout 尺寸标为动态，使同一份编译产物可在不同 shape 下运行。详见 [静态与动态 Layout](../kernel_development/core_concepts/layout.md)。

### 3.1 创建与绑定

用 `from_dlpack` 绑定真实 NPU 缓冲，或用 `make_fake_tensor` 造仅含元数据的类型样本。

#### `from_dlpack`

**源码：** [`catlass.tla.runtime.from_dlpack`](../../../catlass/tla/runtime.py#L625)

功能说明：

将 DLPack NPU tensor **零拷贝**绑定为 TLA Host tensor。返回对象与 `tensor_dlpack` 共享同一块设备缓冲。

函数原型：

```python
tla.from_dlpack(tensor_dlpack: object, *, layout_tag: Any, origin_shape: Any | None = None, assumed_align: int | None = None, stream: int | None = -1, element_type: type | None = None) -> _Tensor
```

参数说明：

- *`tensor_dlpack`*（`object`）：实现了 `__dlpack__()` 的对象。须为 Ascend/NPU
  缓冲（如 `torch_npu`）。CPU / NumPy 不可用。必填。
- *`layout_tag`*（`tla.arch.*`）：布局标签，如 `tla.arch.RowMajor`、
  `tla.arch.ColumnMajor`、`tla.arch.zN`。必填。
- *`origin_shape`*（`tuple | int | None`）：逻辑 origin，Python int 树。
  可选；省略时由 DLPack 物理 shape 与 `layout_tag` 推导。不是 Kernel 的
  `tla.make_shape`。
- *`assumed_align`*（`int | None`）：预留参数，当前无实际效果。
- *`stream`*（`int | None`）：传给 `__dlpack__(stream=...)`。默认 `-1`（不做流同步）。
  `None` 表示省略 `stream` 参数。
- *`element_type`*（`type | None`）：可选。覆盖从 DLPack 推导出的元素类型；
  默认 `None` 表示沿用 DLPack。当 DLPack 无法表达真实类型时使用（例如 fp8），
  传入 `tla.Float8E4M3FN` / `Float8E5M2`。须与导出缓冲的每元素位宽一致。

约束说明：

- 所有权遵循 DLPack consumer 约定：capsule 会被消费，返回的 Host tensor
  销毁时调用 deleter；同时会保留对 `tensor_dlpack` 的引用，因此
  `from_dlpack(x.contiguous().to(device), ...)` 这类临时源是安全的。
- capsule 仅能消费一次；再次传入已消费的 capsule 会抛 `RuntimeTensorError`。
  需要再次绑定时请重新调用 `from_dlpack`。
- 二维 `RowMajor` 须先 `tensor.contiguous()`；二维 `ColumnMajor` 须先
  `tensor.permute(1, 0).contiguous()`。物理布局不符时抛 `RuntimeTensorError`。
  显式传入 `origin_shape` 则跳过该检查。
- 默认得到静态 layout。跨 shape 复用编译产物时再调用
  `mark_layout_dynamic` / `mark_compact_shape_dynamic`。
- 若指定 `element_type`，其每元素位宽须与导出的 DLPack 缓冲一致。

调用示例：

```python
tx = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
ty = from_dlpack(
    y.permute(1, 0).contiguous(),
    layout_tag=tla.arch.ColumnMajor,
)
```

---

#### `make_fake_tensor`

**源码：** [`catlass.tla.runtime.make_fake_tensor`](../../../catlass/tla/runtime.py#L830)

功能说明：

构造仅含元数据、不绑定设备缓冲的 Host tensor（`data_ptr == 0`）。
用于无需 NPU 时给 `tla.compile` 提供类型样本。真实缓冲请用 `from_dlpack`。

函数原型：

```python
tla.make_fake_tensor(dtype: Any, shape: Any, stride: Any, *, layout_tag: Any | None = None, addrspace: Any = AddressSpace.gm, origin_shape: Iterable[Any] | None = None, coord: Iterable[Any] | None = None, assumed_align: int | None = None) -> _Tensor
```

参数说明：

- *`dtype`*：元素类型，如 `tla.Float16` / `tla.Float32`。必填。
- *`shape`*（`int | tuple`）：逻辑 shape 树（zN 等物理布局用嵌套 tuple）。必填。
- *`stride`*（`int | tuple`）：stride 树，结构须与 `shape` 一致。必填。
- *`layout_tag`*：`tla.arch` 标签。可选，默认 `tla.arch.RowMajor`。
- *`addrspace`*：地址空间。可选，默认 `AddressSpace.gm`。
- *`origin_shape`*（`int | tuple | None`）：逻辑 origin。可选，默认等于 `shape`。
- *`coord`*（`int | tuple | None`）：坐标树。可选；省略时由 layout 推导（通常为零）。
- *`assumed_align`*（`int | None`）：预留参数，当前无实际效果。

约束说明：

- `shape` / `stride` / `origin_shape` / `coord` 须为 Python int 树，
  不能是 Kernel 侧 `tla.make_shape` / `tla.make_stride` / `tla.make_coord`。
- 始终未绑定，不能直接 launch；真实缓冲须改用 `from_dlpack`。
- 显式传入的 `shape` / `stride` 按原样使用（不做 layout remap）。

调用示例：

```python
fa = make_fake_tensor(tla.Float16, (128, 64), (64, 1))
fzn = make_fake_tensor(
    tla.Float16,
    ((16, 2), (16, 4)),
    ((16, 256), (1, 512)),
    layout_tag=tla.arch.zN,
    origin_shape=(32, 64),
)
```

---

### 3.2 动态 Layout

将静态 layout 尺寸标为动态。详见 [静态与动态 Layout](../kernel_development/core_concepts/layout.md)。

#### `Tensor.mark_layout_dynamic`

**源码：** [`catlass.tla.runtime._Tensor.mark_layout_dynamic`](../../../catlass/tla/runtime.py#L274)

功能说明：

将所有 shape 维标为动态，使一份 artifact 可接受不同 extents。
stride 除 leading 维（保持 `1`）外均变为动态；广播 stride `0` 保留。
对应的 `origin_shape` 叶节点也变为动态，编译类型不再依赖具体 DLPack 尺寸。

函数原型：

```python
tensor.mark_layout_dynamic(leading_dim: int | None = None) -> '_Tensor'
```

参数说明：

- *`leading_dim`*（`int | None`）：stride 为 `1` 的 leading 维索引。
  可选，默认 `None`（由 `layout_tag` 或紧凑 stride 顺序推断）。

约束说明：

- 原地修改并返回 `self`（可链式调用）。
- 各 `coord` 叶节点必须为 `0`；切片子视图会失败。
- `leading_dim` 对应维的 stride 必须为 `1`。
- NZFamily 布局下，每组两个物理 shape 叶节点对应一个逻辑 `origin_shape` 轴。

调用示例：

```python
ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
ta = ta.mark_layout_dynamic()
artifact = tla.compile(my_kernel, ta, options="--npu-arch 3510")
```

---

#### `Tensor.mark_compact_shape_dynamic`

**源码：** [`catlass.tla.runtime._Tensor.mark_compact_shape_dynamic`](../../../catlass/tla/runtime.py#L346)

功能说明：

将指定的一个紧凑 shape 维（`mode`）标为动态。以该维为因子的 major 维 stride
也会变为动态。对应的 `origin_shape` 叶节点同步标记，编译类型不再依赖具体尺寸。

函数原型：

```python
tensor.mark_compact_shape_dynamic(mode: int, stride_order: tuple[int, ...] | None = None) -> '_Tensor'
```

参数说明：

- *`mode`*（`int`）：要标记为动态的扁平 shape 叶节点索引（从 0 开始）。必填。
- *`stride_order`*（`tuple[int, ...] | None`）：紧凑 stride 顺序（外层 → 内层）。
  可选；省略时由当前 stride 推断。

约束说明：

- 原地修改并返回 `self`。
- 各 `coord` 叶节点必须为 `0`。
- `stride_order` 须为 `range(rank)` 的一个排列。
- NZFamily 布局下，物理维 0/1 对应逻辑 M，物理维 2/3 对应逻辑 N。

调用示例：

```python
ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
ta = ta.mark_compact_shape_dynamic(mode=0)
```

---
