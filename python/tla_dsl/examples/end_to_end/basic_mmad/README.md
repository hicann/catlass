# 基础 MMAD 端到端示例

本目录提供两条入口，覆盖 **TLA DSL** 下 GM→L1→L0 的拷贝、`tla.mmad` 与写回 GM 的完整链路：

| 脚本 | 作用 |
|------|------|
| **`basic_matmul.py`** | 主示例：可配置 GM **布局**与 **元素类型**，多 block、K 维分块、L1/L0 双缓冲；默认会 **编译 + 上板 + 与 NumPy 对比**。内核体在 **`basic_mmad_kernels.py`**（真实 `.py` 源，便于 lowering 变换）。 |
| **`basic_mmad.py`** | 固定 **32×32×32、f32** 的小内核，逻辑简单；**`dump_ir.py` / `dump_mlir.sh`** 用它导出 **TLA MLIR**（`tla` 方言）与制品路径，便于对照 IR。 |

下文中的矩阵尺寸、dtype 组合与命令行参数均以仓库内脚本为准；修改 **`basic_mmad_kernels.py`**、**`basic_matmul.py`** 或 **`basic_mmad.py`** 中的默认值后，请以 **`--help`** 与源码为准。

---

## 问题规模与分块（`basic_mmad_kernels.py`）

- 逻辑 GEMM：`m × n × k` = **`333 × 444 × 555`**（可在该文件中改 `m` / `n` / `k` 与 `l1_*` / `l0_*` 分块，**需与 `basic_matmul.py` 侧 host 张量一致**）。
- L1 分块：`l1_tm × l1_tn × l1_tk` = **256×256×256**；L0 分块：`l0_tm × l0_tn × l0_tk` = **256×256×64**。
- 按 `block_idx` / `block_dim` 遍历 **二维 block 网格**（`grid_m × grid_n`），对每个输出 tile 沿 K 做 **L1 外环 + L0 内环**，L1/L0 各自 **双缓冲**（`l1a0/l1a1`、`l0a0/l0a1` 等），并用 `tla.flag` 在 MTE1/MTE2/CUBE/FIX 之间同步。

---

## GM 张量与 `layout_tag`（`basic_matmul.py`）

Host 侧通过 `tla.Tensor` 构造与内核一致的视图（节选逻辑）：

- **形状**：`A` 为 `(m, k)`，`B` 为 `(k, n)`，`C` 为 `(m, n)`；`origin_shape` 与逻辑形状相同。
- **`layout_tag`**（GM 侧）  
  - `A`：`--layout-a row` → `tla.arch.RowMajor`；`col` → `tla.arch.ColumnMajor`。  
  - `B`：`--layout-b row` / `col` 同上。  
  - **`C` 固定为 `tla.arch.RowMajor`**（输出在 GM 上为行主序存储）。
- **列主 GM 填数**：`basic_matmul._fill_gm_from_dense` 会把行主序的 dense golden 转成设备上列主视图期望的线性序（见源码注释）。

**说明**：**A、B 在 GM 上可分别选行主或列主**；**C 在 GM 上固定为行主**。Lowering 会为不同组合选择 **GM row_major→L1 zN**、**GM column_major→L1 nZ** 等拷贝路径（以编译器实际生成的 IR 为准）。

---

## 元素类型与 L0C（`basic_mmad_kernels.py` + `basic_matmul.py`）

- **`DTYPE_A` / `DTYPE_B`**：在 `tla.compile` 前由 `basic_matmul._apply_kernel_dtypes` 写入 `basic_mmad_kernels`，须 **二者相同**（`tla.mmad` 要求 lhs/rhs 元素类型一致）。
- **`DTYPE_C`**：在 kernels 中 **恒为 `tla.Float32`**，用于 **L0C** 与 `make_tensor_like(..., dst_dtype=DTYPE_C)`；Cube MMAD 在 L0C 上按 **fp32 累加**。
- **`DTYPE_GM_C`**：GM 上 **C 矩阵的元素类型**（`f32`，或缩窄的 `f16` / `bf16`）；`tla.copy(gm_c, l0_c)` 会对应降低为 `copy_cc_to_gm_row_major_float` / `_half` / `_bf16` 等。

**允许的 `(dtype-a, dtype-b, dtype-c)` 组合**（与 `_validate_mmad_dtype_triple` 一致）：

| dtype-a | dtype-b | dtype-c（GM C） |
|---------|---------|------------------|
| f16 | f16 | f32 |
| f16 | f16 | f16 |
| bf16 | bf16 | f32 |
| bf16 | bf16 | bf16 |
| f32 | f32 | f32 |

- 使用 **bf16** 做 host 张量时，若 NumPy 无 `bfloat16` dtype，需安装 **`ml_dtypes`**（`pip install ml_dtypes`），否则会按源码提示退出。

---

## 内核结构概要（`basic_mmad_kernels.basic_mmad_kernel`）

1. 在 L1/L0 分配缓冲，`recast_ptr` 为当前 `DTYPE_A` / `DTYPE_B` / `DTYPE_C`。  
2. 按 block 取 GM 上大块 `tile_view(mem_*, ...)`。  
3. 对 K 维分块：`copy` GM→L1（双缓冲），再 L1→L0（双缓冲），`tla.mmad(..., init_c=...)` 在首个 K 片为 `True`，其后为 `False`。  
4. 块尾 `copy` L0C→GM 子块；`fix_done` 等与 **FIX** 管线的 flag 用于收尾同步。  
5. 全程 **仅 `tla.cube()` 语义**（无 vector 区）；`arch_scope` 由 **`basic_matmul._runtime_kwargs`** 固定为 **`aic.c310`**。

---

## 其他文件

| 文件 | 说明 |
|------|------|
| `dump_ir.py` | 对 **`basic_mmad`**（32³）做 **`tla` 方言** 前端降低，并调用 bridge 写出 `artifacts/` 下 **`0_basic_mmad.tlair.mlir`**（TLA MLIR）、**`1_basic_mmad.lowered.mlir`**（经类型桥接后的 MLIR）与 **`3_basic_mmad.intermediate_trace.txt`**（pass trace）。 |
| `dump_mlir.sh` | 在 **`python/tla_dsl`** 根执行 `dump_ir.py`，再调用 **`TlaCompile`**（含 `--emit=tlair` 等与工具链约定的参数）处理 **`tla` 方言 MLIR**，并刷新降低后的 MLIR 与 trace。 |

---

## 环境前提

- 已安装 **`python/tla_dsl`** 可编辑包：`cd python/tla_dsl && pip install -e .`（并具备 MLIR Python 绑定等依赖）。  
- 上板运行需 Ascend 运行时；编译常需：  
  `export TLA_DSL_HIVM_TEMPLATE_BC=$PWD/mlir/build/bc/meta_op.aic.c310.bc`  
  （路径以你本机 `mlir/build` 为准；在 Catlass 容器/CI 中通常已配好。）

---

## 运行指令

以下均在 **Catlass 仓库根目录下的 `python/tla_dsl`** 执行（使 `examples/end_to_end/...` 与 `catlass` 导入一致）。

```bash
cd python/tla_dsl

# 仅打印 TLA MLIR（`--dump-tlair`；需单一 layout + 单一 dtype 三元组，勿与 --all-layouts / --all-mmad-dtypes 同用）
python examples/end_to_end/basic_mmad/basic_matmul.py --dump-tlair

# 只编译生成 kernel.o，不上板
python examples/end_to_end/basic_mmad/basic_matmul.py --build-only

# 上板并校验（`--device` 默认为 2，可按本机改；终端打印对比结果）
python examples/end_to_end/basic_mmad/basic_matmul.py --run --device 0

# 指定 GM 布局与类型（示例：B 为列主，A/B f16、C f32）
python examples/end_to_end/basic_mmad/basic_matmul.py --run \
  --layout-a row --layout-b col \
  --dtype-a f16 --dtype-b f16 --dtype-c f32

# 扫全部 (layout-a, layout-b) 组合
python examples/end_to_end/basic_mmad/basic_matmul.py --run --all-layouts

# 扫全部允许的 MMAD dtype 三元组（可与 --all-layouts 组合）
python examples/end_to_end/basic_mmad/basic_matmul.py --run --all-mmad-dtypes
```

常用可选参数：`--block`、`--sentinel`、`--atol`、`--cache-dir`、`--force-recompile`、`--no-cache`。完整说明见：

```bash
python examples/end_to_end/basic_mmad/basic_matmul.py --help
```

---

## 成功运行时的终端信息（示例）

`basic_matmul.py` 在 `--run` 下会打印 `compile_ok=True`、`launch_ok=True`、`kernel.o path=...`，以及 `C unchanged?` / `C equals expected matmul?` / `first mismatch=...` 等。这些字段的**具体数值**取决于当前 **`m×n×k`**、block 划分、**`--sentinel`** 与 dtype 组合，用于与 NumPy golden 对照即可。

若运行 **`basic_mmad.py`**（**32×32×32**、f32 小例子），逻辑输出矩阵 **C** 为 **32×32**，共 **1024** 个元素；在全部被内核写回时，`C changed count` 等与元素个数相关的统计应与 **1024** 一致。

若编译或驱动缺失，请根据报错检查 **`TLA_DSL_HIVM_TEMPLATE_BC`**、`ASCEND_HOME_PATH` 及 **`--device`**。
