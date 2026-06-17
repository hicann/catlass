# 基础 MMAD 端到端示例

本目录演示 **TLA DSL** 下 GM→L1→L0 拷贝、`tla.mmad` 与写回 GM 的完整链路。

| 文件 | 作用 |
|------|------|
| **`basic_matmul.py`** | 主入口：可配置 GM **布局**与 **元素类型**，多 block、K 维分块、L1/L0 双缓冲；**`--run`** 用 **torch + torch_npu** 上板并校验；**`--build-only`** 仅编译；**`--dump-tlair`** 仅导出 TLA MLIR。 |
| **`basic_mmad_kernels.py`** | 设备内核（`@tla.kernel` 的 `basic_mmad_kernel`）及问题规模 `m/n/k`、分块常量。 |
| **`basic_mmad_kernels_mutex.py`** | 使用显式 `mutex.lock/unlock` 的同步版本，可通过 `--use-mutex` 选择。 |
| **`basic_mmad_kernels_mutex_with.py`** | 使用 `with tla.mutex_guard(...)` 的同步版本，可通过 `--use-mutex-with` 选择。 |

矩阵尺寸、dtype 与 CLI 默认值以源码与 **`--help`** 为准。

---

## 问题规模与分块（`basic_mmad_kernels.py`）

- 逻辑 GEMM：`m × n × k` = **`333 × 444 × 555`**（改 `m` / `n` / `k` 或 `l1_*` / `l0_*` 时须与 host 侧形状一致）。
- L1 分块：`l1_tm × l1_tn × l1_tk` = **256×256×256**；L0 分块：`l0_tm × l0_tn × l0_tk` = **256×256×64**。
- 按 `block_idx` / `block_dim` 遍历 **二维 block 网格**，对每个输出 tile 沿 K 做 **L1 外环 + L0 内环**，L1/L0 **双缓冲**，`tla.flag` 在 MTE1/MTE2/CUBE/FIX 间同步。

---

## Host 与 GM 张量（`basic_matmul.py`）

**`--run` 路径**（需要 **PyTorch**、**torch_npu**、Ascend 运行时）：

1. `tla.initialize(device=<id>)` 设置 ACL 设备；`torch.npu.set_device(<id>)` 对齐 Torch 当前 NPU。
2. 在 NPU 上构造 **`torch_tensor_*`**（`device="npu"`），golden 用 Torch 矩阵乘。
3. **`catlass.runtime.from_dlpack`** + **`_create_tla_tensor`** 将每个 `torch_tensor_*` 包成 **`tla_tensor_*`**（`tla.Tensor`），供 `tla.compile` / launch。
4. 列主 GM：`torch_tensor` 经 **`_device_buffer_for_layout`**（`permute(1,0).contiguous()`）与 `layout_tag` 对齐。

**`--build-only` / `--dump-tlair`**：仅用 **`_compile_only_type_args`** 构造**元数据** `tla.Tensor`（`_eager_capture`），**不**依赖 torch_npu 或真实设备 buffer。

**`layout_tag`（GM）**

- `A`（M×K）：`--layout-a row` → `tla.arch.RowMajor`；`col` → `ColumnMajor`。
- `B`（K×N）：`--layout-b` 同上。
- **`C` 固定 `tla.arch.RowMajor`**。

Lowering 会为不同组合选择 **GM row_major→L1 zN**、**GM column_major→L1 nZ** 等路径（以实际 IR 为准）。

---

## 元素类型与 L0C

- **`DTYPE_A` / `DTYPE_B`**：`tla.compile` 前由 `_apply_kernel_dtypes` 写入 `basic_mmad_kernels`，须相同（`tla.mmad` 要求）。
- **`DTYPE_C`**：kernels 内恒 **`tla.Float32`**（L0C / `make_tensor_like`）；Cube 在 L0C 上 **fp32 累加**。
- **`DTYPE_GM_C`**：GM 上 C 的元素类型；`tla.copy(gm_c, l0_c)` 对应 `copy_cc_to_gm_row_major_float` / `_half` / `_bf16` 等。

| dtype-a | dtype-b | dtype-c（GM C） |
|---------|---------|------------------|
| f16 | f16 | f32 |
| f16 | f16 | f16 |
| bf16 | bf16 | f32 |
| bf16 | bf16 | bf16 |
| f32 | f32 | f32 |

---

## 内核结构概要（`basic_mmad_kernel`）

1. L1/L0 缓冲 + `recast_ptr`（当前 `DTYPE_A` / `DTYPE_B` / `DTYPE_C`）。
2. 按 block `tile_view` GM 大块。
3. K 维：`copy` GM→L1→L0（双缓冲），`tla.mmad(..., init_c=...)` 首片 `True` 其后 `False`。
4. 块尾 L0C→GM；`fix_done` 等与 FIX 管线同步。
5. 仅 **`tla.cube()`**；`arch_scope` 由 `_runtime_kwargs` 固定为 **`aic.c310`**。

---

## 其他文件

| 文件 | 说明 |
|------|------|
| `dump_ir.py` | 依赖同目录下的 **`basic_mmad`** 小内核模块（32³ f32）导出 MLIR 制品；若仓库未提供该模块，请改用 **`basic_matmul.py --dump-tlair`**。 |
| `dump_mlir.sh` | 在 `python/tla_dsl` 根目录调用 `dump_ir.py` 与 **`TlaCompile`**，刷新 `artifacts/` 下 MLIR 与 trace。 |

---

## 环境前提

- `cd python/tla_dsl && pip install -e .`（及 MLIR Python 绑定等）。
- **`--run`**：`torch`、`torch_npu`，`ASCEND_HOME_PATH` / CANN 已配置。
- 编译 toolchain 常需：  
  `export TLA_DSL_HIVM_TEMPLATE_BC=$PWD/mlir/build/bc/meta_op.aic.c310.bc`  
  （路径以本机 `mlir/build` 为准。）

---

## 运行指令

在 **`python/tla_dsl`** 下执行：

```bash
cd python/tla_dsl

# 仅打印 TLA MLIR（单一 layout + dtype 三元组；勿与 --all-layouts / --all-mmad-dtypes 同用）
python examples/end_to_end/basic_mmad/basic_matmul.py --dump-tlair

# 只编译 kernel.o（无需 torch_npu / --device）
python examples/end_to_end/basic_mmad/basic_matmul.py --build-only

# 上板并校验（默认即 --run；--device 默认 2，仅 run 路径使用）
python examples/end_to_end/basic_mmad/basic_matmul.py --device 0

python examples/end_to_end/basic_mmad/basic_matmul.py --run --device 0 \
  --layout-a row --layout-b col \
  --dtype-a f16 --dtype-b f16 --dtype-c f32

python examples/end_to_end/basic_mmad/basic_matmul.py --run --all-layouts --device 0
python examples/end_to_end/basic_mmad/basic_matmul.py --run --all-mmad-dtypes --device 0
```

常用参数：`--block`、`--sentinel`、`--atol`、`--cache-dir`、`--force-recompile`、`--no-cache`。

```bash
python examples/end_to_end/basic_mmad/basic_matmul.py --help
```

---

## 成功运行时的终端输出

**`--run`** 会打印 `compile_ok=True`、`host=torch_npu`、`launch_ok=True`、`kernel.o path=...`，以及 `C unchanged?` / `C equals expected matmul?` / `first mismatch=...` 等（与 `m×n×k`、block、`--sentinel`、dtype 有关；golden 为 **Torch** 在 NPU 上的 matmul）。

**`--build-only`** 仅打印 `compile_ok=True` 与 `kernel.o path=...`。

若失败，请检查 **`TLA_DSL_HIVM_TEMPLATE_BC`**、`ASCEND_HOME_PATH`（run 路径）、**`torch_npu`** 及 **`--device`**。
