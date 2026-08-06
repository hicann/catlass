# 基础 MMAD 端到端示例

本目录下提供的系列样例演示 **TLA DSL** 下基础矩阵乘的计算，包括 GM→L1→L0 拷贝、`tla.mmad` 与写回 GM 的完整链路。

| 文件 | 概述 |
|------|------|
| [**`basic_matmul.py`**](basic_matmul.py) | 基础矩阵乘示例，支持动态场景，含双缓冲/unit_flag优化，核函数为 `basic_mmad_kernel`，是其他变体的基准版本。 |
| [**`basic_matmul_atomic_add.py`**](basic_matmul_atomic_add.py) | 启用原子加的矩阵乘示例，每个 tile 块计算后立刻搬移，在 GM 上进行累加，DTYPE_C类型固定为 `tla.Float32`。 |
| [**`basic_matmul_mutex.py`**](basic_matmul_mutex.py) | 使用 `Mutex` 原语的矩阵乘示例，`Mutex` 原语同步不再区分源和目的指令，是互斥锁模型。 |
| [**`basic_matmul_mutex_with.py`**](basic_matmul_mutex_with.py) | 同样是利用 `Mutex` 原语的矩阵乘示例，使用 `with tla.mutex_guard(...)` 表达，替换 `mutex.lock/unlock` 的做法。 |
| [**`basic_matmul_auto_sync.py`**](basic_matmul_auto_sync.py) | 使用 `@tla.kernel(auto_sync="v0")` 自动插入核内 mutex 的独立端到端用例，覆盖 MMAD/FIX unit flag 协议且无需手写本地同步。 |
| [**`basic_mmad_ptr.py`**](basic_mmad_ptr.py) | 最小可运行矩阵乘示例，演示显式传入指针构造 `tla.Tensor` 的做法，矩阵尺寸为静态。 |

---

## 1. 基本介绍

基础矩阵乘算子实现形如$(m, k)$和$(k, n)$的两矩阵的乘法运算，输出形如$(m, n)$，计算公式为：
$$
\begin{aligned}
C &= A \times B \\
C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
\end{aligned}
$$


### 1.1 问题规模与分块

- 逻辑 GEMM：`m × n × k`（默认为 `256`, `512`, `1024`，可通过 `--m` / `--n` / `--k` 指定单一尺寸）。
- 左矩阵排布默认为 `tla.arch.RowMajor`，右矩阵排布默认为 `tla.arch.RowMajor` （可分别通过 `--layout-a` 和 `--layout-b` 传递）,输出矩阵固定为 `tla.arch.RowMajor`。
- L1 分块：`l1_tm × l1_tn × l1_tk` （默认为 `256`, `256`, `128`）；L0 分块：`l0_tm × l0_tn × l0_tk` （默认为 `256`, `256`, `32`）。
- 以L1分块对输出尺寸进行切分，划分为二维网格，按 `block_idx` （当前核数） / `block_dim` （总核数） 遍历该 **二维 block 网格**。

### 1.2 支持元素类型

- **`DTYPE_A` / `DTYPE_B`**：分别为左、右矩阵的数据类型（默认为 **`tla.Float16`**），可通过外部 `--dtype-a` / `--dtype-b` 传递，要求二者相同以满足mmad计算要求。
- **`DTYPE_C`**：输出矩阵乘的数据类型（默认为 **`tla.Float32`**），可通过外部 `--dtype-c` 传递。
- 以下是支持的数据组合类型：

| dtype-a | dtype-b | dtype-c |
|---------|---------|------------------|
| f16 | f16 | f32 |
| f16 | f16 | f16 |
| bf16 | bf16 | f32 |
| bf16 | bf16 | bf16 |
| f32 | f32 | f32 |

### 1.3 执行流程

样例由 Host 侧数据准备、Kernel 编译与启动、Device 侧计算、Host
侧校验四部分组成。

#### 1.3.1 Host 侧准备

1. 调用 `tla.initialize(device=<id>)` 初始化全局运行时，整个进程内**只能同时有一次有效初始化**，任务执行后需执行 `tla.finalize` 。
2. 初始化指定 NPU，再调用`torch.npu.set_device(<id>)`，运行时和 PyTorch 使用同一设备。
3. 确定待测试的矩阵尺寸、布局和数据类型，然后在 NPU 设备上构造输入矩阵 A 和 矩阵 B，再转为 `tla.Tensor`， 推荐两种做法：
   - 调用`from_dlpack(...)` 基于源张量对象（如 `torch.Tensor` ）转为 `tla.Tensor`，要求原张量对象实现 [`__dlpack__`](https://dmlc.github.io/dlpack/latest/) 属性，且在运行期间不被垃圾回收；

   - 显式构造 `tla.Tensor`， 可参考 [`basic_mmad_ptr.py`](basic_mmad_ptr.py)， `data_ptr` 使用设备地址（非0视为已绑定）。

#### 1.3.2 Kernel 编译与启动

1. 调用 `tla.compile(func, *sample_args, type_args = None, **compile_kwargs)` 执行编译操作，参数说明：

   - `func`：`@tla.kernel` 或 `@tla.jit` 装饰的核函数，是一个 `TlaJitFunction` 对象。
   - `sample_args`：类型样本，通常与launch执行时的参数一致。
   - `type_args`： 显式覆盖推断结果，如设置 `arch_scope` 为 `aiv.c310` 或 `aic.c310` 可覆盖目标架构类型。
   - `compile_kwargs`：编译参数，常用参数见下表：

   | 参数 | 默认值 | 说明 |
   |------|--------|------|
   | `backend` | `"ascend"` | 后端名 |
   | `arch_scope` | `target_arch`和`core_type`推导 | 指定核类型与架构，合法值为 `"aic.c310"`, `"aiv.c310"` |
   | `target_arch` | 平台默认（如 `"c310"`） | 目标芯片架构 |
   | `core_type` | `"aiv"` | 核函数类别，合法值 `"aic"`或`"aiv"` |
   | `kernel_mode` | `core_type` | 与`core_type`相同，支持 `"mix"` |
   | `cache` | `True` | 是否命中磁盘/内存 (也可通过设置 `TLA_DSL_CACHE` 环境变量修改) |
   | `cache_dir` | `"artifacts/runtime-cache"` | 编译产物缓存路径 （也可通过设置 `TLA_DSL_CACHE_DIR` 环境变量修改）  |
   | `force_recompile` | `False` | 是否强制重新编译（忽略cache，也可通过设置 `TLA_DSL_FORCE_RECOMPILE` 环境变量修改） |
   | `hivmc` | 自动解析 | `hivmc-a5` 的可执行文件路径 | 
   | `hivmc_args` | `()` | 追加给 `hivmc` 的参数组（如 `("--verbose",)` |

    `tla.compile(...)` 完成后，返回一个 `kernel` 对象，预期会生成 `manifest.json`， `kernel.o` 以及 `lowered.mlir` 三个文件，其中 `manifest.json` 为编译元信息，如果下一次缓存命中则直接复用该产物 `kernel.o`。
2. 编译产物包含 `artifact.kernel_binary_path`（`.o` 路径）和 `artifact.cache_key`（缓存键），Host 可据此检查缓存复用情况和定位产物。
3. 通过 `artifact(*args, **launch_kwargs)` 在设备上启动 Kernel，其中位置参数 `args` 是与kernel 签名一致的host侧对象（如 `tla.Tensor`, 标量值等），运行时参数 `launch_kwargs` 通常包括：
   | 参数 | 默认值 | 说明 |
   |------|--------|------|
   | `block_dim` | `1` | 启动的核数 |
   | `device`    | 优先从 `current_device_id()` 获取当前设备ID(`int`) | 使用的设备ID |
   | `stream`    | 优先从 `current_stream()` 获取当前异步流(`int`) | `rtStream_t`编号 |

#### 1.3.3 Device 侧计算

1. Kernel 从 `mem_a`、`mem_b` 的 `origin_shape` 读取运行时参数  `m`, `n`, `k`，按l1_tile 尺寸计算二维 block 网格。每个AI Core 从自身 `block_idx` 开始，以核数 `block_dim` 为步长处理一个或多个 Tile 分块。
2. 为 A、B 分别在 L1 和 L0 分配两份缓冲区，通过两级 ping-pong
   双缓冲隐藏搬运开销。
3. 启动数据搬运，MTE2 将当前 K tile 从 GM 搬到 L1，MTE1 再将 L1 子 tile 搬到
   L0A/L0B。各流水线通过 `l1*_copy_start/end`、
   `l0*_copy_start` 和 `l0_copy_end` flag 协调缓冲区的读取与复用。
4. 调用 `tla.mmad` 完成 L0C 上的矩阵乘动作，完成矩阵乘累加计算（注意[basic_matmul_atomic_add](basic_mmad_kernels_atomic_add.py) 中的累加动作通过GM上的原子加实现），可开启 `ENABLE_UNIT_FLAG` 以提升并行度；
5. 完成分块的矩阵乘后，通过 FIXPIPE 流水写回到 GM 。 Kernel 退出前等待所有同步标志位恢复，保证流水正常。

#### 1.3.4 Host 侧结果校验

1. 调用 `torch.matmul(...)` 计算golden值（参考结果）。
2. 调用 `torch.npu.synchronize()` 确保流同步完成，使用 `torch.isclose(...)` 逐元素比较 NPU 计算结果与 golden 的误差。
3. 调用 `tla.finalize()` 清空全局状态。

---

## 2. 运行样例

 要运行本路径下的样例，请参考[环境配置](../../../README.md#2.从零开始：环境与完整指令)一节的有关内容。

### 2.1 命令行参数

```text
basic_matmul.py [-h] [--device DEVICE] [--m M] [--n N] [--k K]
                [--layout-a {row,col}] [--layout-b {row,col}]
                [--dtype-a {f16,bf16,f32}]
                [--dtype-b {f16,bf16,f32}]
                [--dtype-c {f16,bf16,f32}]
                [--block-dim BLOCK]
                [--sentinel SENTINEL]
                [--cache-dir CACHE_DIR] 
                [--force-recompile]
                [--no-cache]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | TLA 和 PyTorch 使用的 NPU 设备号。 |
| `--m` / `--n` / `--k` | 未指定 | 自定义一组运行时矩阵尺寸。 |
| `--layout-a` / `--layout-b` | `"row"` / `"row"` | A、B 的 GM 布局，可选 `"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--dtype-a` / `--dtype-b` / `--dtype-c` | `"f16"` / `"f16"` / `"f32"` | 左、右矩阵A，B和结果矩阵 C 的数据类型，可选范围包括 `"f16"`, `"bf16"` 和 `"f32"` 。 |
| `--block-dim` | `-1`（哨兵值，后续替换为`tla.get_aicore_num(device)`） | 表示使用 NPU 所启用的 AI Core 核数 |
| `--sentinel` | `-7.0` | Kernel 启动前写入 C 的哨兵值，用于暴露未写回的元素。 |
| `--cache-dir` | `artifacts/runtime-cache` | 编译缓存目录 |
| `--force-recompile` | `False` | 忽略已有缓存并强制重新编译。 |
| `--no-cache` | `False` | 禁用编译缓存。 |


### 2.2 调用示例

在 `python/tla_dsl` 目录下执行：

```bash
cd python/tla_dsl

# 查看帮助
python examples/end_to_end/basic_mmad/basic_matmul.py --help

# 基础测试 (默认 m=256, n=512, k=1024)
python examples/end_to_end/basic_mmad/basic_matmul.py

# 指定重新编译 (不启用缓存)
python examples/end_to_end/basic_mmad/basic_matmul.py --force-recompile

# 指定使用的 NPU ID 以及核数
python examples/end_to_end/basic_mmad/basic_matmul.py --block-dim 1 --device 1

# 指定单一尺寸、布局和 dtype
python examples/end_to_end/basic_mmad/basic_matmul.py \
  --m 256 --n 512 --k 128 \
  --layout-a row --layout-b col \
  --dtype-a f16 --dtype-b f16 --dtype-c f32
```

执行上述测试后，预期输出：
```plain
passed=True cache_key=<CACHE_KEY>
kernel.o=<CACHE_DIR>/<CACHE_KEY>/kernel.o
```

其中 `passed`结果为`True`或`False` 表明 NPU 计算结果与golden参考值精度校验是否通过；`cache_key` 是编译缓存的哈希值，`kernel.o` 后续的路径即为编译后的二进制路径。

---

## 3. 特性 Kernel 介绍

以下介绍本目录下的一些特性Kernel。

## 3.1 basic_matmul_atomic_add

**文件**：`basic_matmul_atomic_add.py`

使用 **原子加（Atomic Add）** 将每个 K tile 的计算部分和直接累加到 GM 输出 C。每个 K 分块计算完毕后立即通过 FIX 流水写回 GM。其中，**TLA DSL** 所采用的原子加特性关键代码如下：
```python
import catlass as tla 

# ...

# is_first_l1_tile 判断是否为首块累加的K Tile ( `k_l1 == 0` )
if not is_first_l1_tile:
   # 非首个 Tile 块启用原子加
   tla.copy(
      gm_c_by_core,
      l0_c,
      tla.params.CopyL0C2DstParams(
            unit_flag=0b11,
            atomic_mode=tla.params.AtomicMode.ADD
      )
   )
else:
   # 第一个 Tile 块搬出直接覆写 GM C
   tla.copy(
      gm_c_by_core,
      l0_c,
      tla.params.CopyL0C2DstParams(unit_flag=0b11)
   )
```

启用原子加的操作在`tla.copy(...)`中通过向 `CopyL0C2DstParams` 参数中传入 `atomic_mode` 实现，目前支持的模式包括 `AtomicMode.ADD` 和 `AtomicMode.NONE` （默认行为，无原子操作）。


## 3.2 basic_matmul_mutex

**文件**：`basic_matmul_mutex.py`

Mutex 原语是一种互斥锁，是面向数据层面的依赖，较 set/wait 原语可以更简洁的描述同步关系。

在 **TLA DSL** 中，使用显式 **Mutex 锁/解锁**（`mutex.lock(pipe)` / `mutex.unlock(pipe)`）操作，当某条流水线申请到对某个`mutex`资源的 `lock` 操作后，该锁供其所有，阻塞其他申请该 `mutex` 资源的流水，直至其释放（ `unlock` 操作）。

以 GM 向 L1A 的数据搬运操作为例，示例如下：
```python
import catlass as tla

# ...

# mutex 资源申请
mutex_l1a = tla.mutex(resource="l1a", id=0)

# ...

# 启动 MTE2 源流水的数据搬运 (GM->L1A)
mutex_l1a.lock(pipe=tla.arch.MTE2)
tla.copy(l1_a, gm_a_by_l1)
mutex_l1a.unlock(pipe=tla.arch.MTE2)

# ...

# mutext_l1a 释放后（L1A数据准备完成）
# 可以启动 MTE1 源流水的数据搬运 (L1A->L0A)
mutex_l1a.lock(pipe=tla.arch.MTE1)
mutex_l0a.lock(pipe=tla.arch.MTE1)
tla.copy(l0_a, l1_a_by_l0)
mutex_l0a.unlock(pipe=tla.arch.MTE1)
mutex_l1a.unlock(pipe=tla.arch.MTE1)
```

---

## 3.3 basic_matmul_mutex_with

**文件**：`basic_matmul_mutex_with.py`

使用 `with tla.mutex_guard(...)` **上下文管理器**实现 Mutex 同步，RAII 风格，可以省去 `mutex.lock(pipe)` 和 `mutex.unlock(pipe)` 的手动管理。

仍以GM 至 L1A 的数据搬运过程为例，使用 `with` 上下文管理的写法是：
```python
import catlass as tla

# ...
with tla.mutex_guard(mutex_l1a):
   tla.copy(l1_a, gm_a_by_l1)
```

这一操作与手动 `lock` / `unlock` 操作等价，由 mutex 串行化。

---

## 3.4 basic_matmul_auto_sync

**文件**：`basic_matmul_auto_sync.py`

在装饰器 `@tla.kernel` 传入 `auto_sync` 参数（目前支持 `auto_sync="v0"` ），可自动处理多级流水线间的同步，覆盖 MMAD/FIX unit flag 协议且无需手写本地同步，示例如下：

```python
@tla.kernel(auto_sync="v0")
def kernel_func(...):
   # ...
```

---

## 3.5 basic_mmad_ptr

**文件**：`basic_mmad_ptr.py`

片上数据直采的矩阵乘示例，展示 **TLA DSL** 通过手写 `tla.Tensor` 绕过DLPack的传参做法，样例仅支持静态场景，以及 Device 侧的处理，默认问题规模为 64×64×64。

在Host侧，根据原指针显式创建 `tla.Tensor`表示。例如：
```python
import catlass as dsl

# ...

# 创建 torch.Tensor
_torch_tensor = torch.rand(M_DIM, K_DIM, dtype=torch.float32, device="cpu") * 10.0 - 5.0
contiguous = _torch_tensor.contiguous()
with runtime_mod._eager_capture():
   # 通过显式 tla.Tensor创建
   _tensor =  tla.Tensor(
         tla.make_shape(row, col),
         tla.Float32,
         origin_shape=tla.make_shape(row, col),
         coord=tla.make_coord(0, 0),
         stride=tla.make_stride(col, 1),
         data_ptr=int(contiguous.data_ptr()),  # 传入源数据指针
   )
```

在 Device 侧，通过 `tla.utils.LocalmemAllocator()` 分配片上空间，以此直接取得各数据指针，例如在分配 L1A 和 L1B 大小时：
```python
import catlass as dsl

# ...
allocator = tla.utils.LocalmemAllocator()

l1a_ptr = allocator.allocate(L1_STAGE_BYTES, 512, tla.AddressSpace.l1)
l1a_ptr = tla.recast_ptr(l1a_ptr, dtype=tla.Float32) # 按照 A 的类型
l1b_ptr = allocator.allocate(L1_STAGE_BYTES, 512, tla.AddressSpace.l1)
l1b_ptr = tla.recast_ptr(l1b_ptr, dtype=tla.Float32)
```
