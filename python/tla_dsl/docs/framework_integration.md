# Torch Tensor 接入

为便于与 `torch` / `torch_npu` 互通，TLA DSL 通过 [DLPack](https://github.com/dmlc/dlpack) 协议，将设备侧 tensor 转换为 Host 侧 `tla.Tensor`。本文说明接入约定、可用 API 与常见用法，并介绍如何绕过 DLPack、直接绑定设备缓冲区。

---

## 1. 使用 `from_dlpack` 显式转换

运行时提供将兼容 DLPack 的张量转为 `tla.Tensor` 的接口。下面从 torch 开始，经 `from_dlpack` 绑定，一直到 `compile` / launch：

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

@tla.kernel
def foo(src: tla.Tensor, dst: tla.Tensor) -> None:
    n = src.origin_shape[0]
    # ... 在 vector/cube 区域内完成计算 ...

torch.npu.set_device(0)

x = torch.rand(1024, dtype=torch.float32, device="npu") * 10.0 - 5.0
y = torch.zeros_like(x)

tx = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
ty = from_dlpack(y.contiguous(), layout_tag=tla.arch.RowMajor)
# 需要跨 shape 复用编译产物时再 mark_*_dynamic
tx = tx.mark_compact_shape_dynamic(0)
ty = ty.mark_compact_shape_dynamic(0)

artifact = tla.compile(foo, tx, ty, options="--npu-arch 3510")
artifact(tx, ty, block_num=1)
torch.npu.synchronize()
```

其中 `x` 须实现 `__dlpack__`（通常为 NPU 上的 `torch` 张量，如经 `torch_npu`）。转换**零拷贝**：返回的 `tla.Tensor` 与源张量共享同一块设备内存；源张量生命周期须覆盖后续 `tla.compile` / launch，否则指针失效。

默认得到**静态 layout**（具体 shape / stride / origin 写进编译类型）。要得到动态或混合静态/动态 layout，须在转换后调用 `mark_layout_dynamic` / `mark_compact_shape_dynamic`（见 [动态 Layout](dsl_dynamic_layout.md)）。

完整签名如下：

```python
def from_dlpack(
    tensor_dlpack: object,
    *,
    layout_tag: Any,
    origin_shape: Any | None = None,
    assumed_align: int | None = None,
    stream: int | None = -1,
) -> Tensor:
```

| 参数 | 说明 |
|------|------|
| `tensor_dlpack` | 实现 `__dlpack__` 的对象；须为 **Ascend/NPU 设备**上的缓冲区，CPU / NumPy 不可用 |
| `layout_tag` | **必填**。`tla.arch` 布局哨兵，如 `RowMajor` / `ColumnMajor` |
| `origin_shape` | 可选。逻辑 origin，须为 `tla.make_shape(...)` 的结果。省略时由物理 shape/stride 与 `layout_tag` 推导 |
| `assumed_align` | **预留参数**。当前 `from_dlpack` 路径统一按元素自然对齐（如 `f32`→4）处理，传入值不影响实际 lowering / 访存行为 |
| `stream` | 传给框架的 `__dlpack__(stream=...)`，只影响「把 torch 张量交给 `from_dlpack` 时」框架侧要不要做流同步；默认 `-1` 表示不做同步 |

### 1.1 布局与物理存储

未传 `origin_shape` 时，二维稠密缓冲须满足样例约定：

| `layout_tag` | Host 侧准备 |
|--------------|-------------|
| `RowMajor` | `tensor.contiguous()`（物理行主序） |
| `ColumnMajor` | `tensor.permute(1, 0).contiguous()`（逻辑列主，物理仍为行主序紧凑缓冲） |

物理 shape/stride 与 `layout_tag` 不一致时抛出 `RuntimeTensorError`。

若显式传入 `origin_shape`，则直接使用该逻辑 origin，并跳过上述物理 stride 校验；shape / stride 元数据由逻辑 origin + `layout_tag` 经 layout remap 得到。

### 1.2 代码示例

下面演示如何用 `from_dlpack` 将 PyTorch 张量转为 `tla.Tensor`，并查看转换结果：

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

torch.npu.set_device(0)
x = torch.rand(30, 20, dtype=torch.float32, device="npu") * 10.0 - 5.0
y = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)

print(y.shape)         # 逻辑 shape
print(y.stride)        # stride 树
print(y.origin_shape)  # 逻辑 origin
print(y.layout_tag)    # 如 'row_major'
print(y.dtype)         # 如 'f32'
print(y.addrspace)     # 如 'gm'
print(y.data_ptr)      # 设备地址
print(y)               # !tla.tensor 类型串（编译元数据）
```

转换后可通过下列属性查看张量信息：

- `tensor.shape`：逻辑 shape  
- `tensor.stride`：stride  
- `tensor.origin_shape`：逻辑 origin  
- `tensor.layout_tag`：布局标签  
- `tensor.dtype`：元素类型  
- `tensor.addrspace`：地址空间  
- `tensor.data_ptr`：设备指针  

---

## 2. 绕过 DLPack 协议

仍可使用 **torch / torch_npu** 上的设备张量；绕过的只是 DLPack capsule 解析与自动 layout 推导，改为用手写元数据绑定。公开入口是 `make_fake_tensor`。

适用场景：

1. 避免 DLPack 解析开销；  
2. 需要完全自管 shape / stride / `layout_tag` / `origin_shape`；  
3. 没有真实设备缓冲，只需给 `tla.compile` 提供类型样本。

`data_ptr` 为 `None` / `0`（默认）表示未绑定的 compile 期占位；传入非 0 设备地址则视为已绑定缓冲。源 `torch` 张量须在后续 compile / launch 期间保持存活。

手写元数据并绑定已有 NPU 缓冲：

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

torch.npu.set_device(0)
# 数据仍在 torch 侧；须在 NPU 上
a = torch.rand(64, 128, dtype=torch.float32, device="npu") * 10.0 - 5.0
rows, cols = a.shape

ta = make_fake_tensor(
    tla.make_shape(rows, cols),
    tla.Float32,
    origin_shape=tla.make_shape(rows, cols),
    coord=tla.make_coord(0, 0),
    stride=tla.make_stride(cols, 1),  # 行主紧凑，须与 a 的物理布局一致
    layout_tag=tla.arch.RowMajor,
    data_ptr=int(a.contiguous().data_ptr()),
)

# 需要动态 GM 时可继续：
# ta = ta.mark_layout_dynamic()

artifact = tla.compile(kernel, ta, ...)
artifact(ta, ..., block_num=...)
```

与 `from_dlpack` 路径的差别：不调用 `from_dlpack`，dtype / shape / stride / `layout_tag` 全部由 Host 手写；指针来自 `torch.Tensor.data_ptr()`。须自行保证这些元数据与物理缓冲一致；若再调用 `mark_*_dynamic`，该 tensor 的各维 `coord` 也必须为 0。

仅需类型样本、不绑真实缓冲时，省略 `data_ptr`（或传 `0` / `None`）即可：

```python
import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

rows, cols = 64, 128
fake = make_fake_tensor(
    tla.make_shape(rows, cols),
    tla.Float32,
    origin_shape=tla.make_shape(rows, cols),
    coord=tla.make_coord(0, 0),
    stride=tla.make_stride(cols, 1),
    layout_tag=tla.arch.RowMajor,
)
artifact = tla.compile(kernel, fake, options="--npu-arch 3510")
```

有 NPU 上的框架张量时，优先走第 1 节 `from_dlpack`；需要自管元数据时走本节 `make_fake_tensor`。
