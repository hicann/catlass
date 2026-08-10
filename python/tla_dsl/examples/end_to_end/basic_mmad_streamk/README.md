# StreamK MMAD 端到端示例

本目录演示 **TLA DSL** 下 StreamK MatMul：AIC（Cube）完成 normal / 尾轮 StreamK 分块 MMAD，AIV（Vector）对 workspace 中的 K 切片归约并写回 GM C。

| 文件 | 作用 |
|------|------|
| **`basic_mmad_streamk.py`** | 主入口：可配置 GM **布局**与 **元素类型**，多 block、K 维分块、L1/L0 双缓冲与 StreamK workspace；默认用 **torch + torch_npu** 上板并校验精度（`--run` 为默认模式）。 |
| **`basic_mmad_streamk_kernels.py`** | 设备内核（单一 `streamk_mmad_kernel`：`tla.cube` + `tla.vector` + `cross_flag` 同步）。 |
| **`streamk_config.py`** | 问题规模 / dtype / L1·L0 分块等编译期常量（host 在 compile 前写入）。 |

矩阵尺寸、dtype 与 CLI 默认值以源码与 **`--help`** 为准。

---

## 问题规模与分块（`streamk_config.py` / `basic_mmad_streamk.py`）

- 默认逻辑 GEMM：`m × n × k` = **`256 × 256 × 512`**（可用 `--m/--n/--k` 覆盖）。
- 默认 **`--block`** = 设备 `cube_core_num`（对齐 Catlass `GetCoreNumAic()`）；也可用 `--block N` 覆盖。
- L1：`256×256×128`；L0：`256×256×32`（`L0_TM=L1_TM`，`L0_TN=L1_TN`）。
- **尾轮-only StreamK**：`streamk_blocks = (grid_m * grid_n) % BLOCK_DIM`；`normal_blocks` 直写 GM C；尾轮 K-tile 按 work-centric 均分到各 AIC，写入固定大小 workspace（每核 2 槽）。
- 当 `streamk_blocks == 0` 时退化为纯 normal（例如 `--m 512 --n 512 --k 512 --block 4`）。
- MN 任务坐标使用 Zn swizzle（`SWIZZLE_OFFSET=3`），与 CPP scheduler 一致。
- workspace 归约固定在同一 kernel 的 AIV 段完成（含 f16/bf16 稠密 cast），不存在 kernel 之外的归约路径。

---

## Host 与 GM 张量（`basic_mmad_streamk.py`）

**`--run` 路径**（需要 **PyTorch**、**torch_npu**、Ascend 运行时）：

1. `tla.initialize(device=<id>)` 设置 ACL 设备；`torch.npu.set_device(<id>)` 对齐 Torch 当前 NPU。
2. 在 NPU 上构造 **`torch_tensor_*`**（`device="npu"`）与 **workspace**；golden 用 Torch 矩阵乘。
3. **`catlass.runtime.from_dlpack`** 将每个 `torch_tensor_*` 包成 **`tla.Tensor`**，供 `tla.compile` / launch。
4. 列主 GM：`torch_tensor` 经 **`_device_buffer_for_layout`**（`permute(1,0).contiguous()`）与 `layout_tag` 对齐。

**`layout_tag`（GM）**

- `A`（M×K）：`--layout-a row` → `tla.arch.RowMajor`；`col` → `ColumnMajor`。
- `B`（K×N）：`--layout-b` 同上。
- **`C` 与 workspace 固定 `tla.arch.RowMajor`**。

---

## 元素类型与 L0C

- **`DTYPE_A` / `DTYPE_B`**：`tla.compile` 前由 `_apply_kernel_dtypes` 写入 kernels，须相同（`tla.mmad` 要求）。
- **`DTYPE_C`**：kernels 内恒 **`tla.Float32`**（L0C / workspace 累加）。
- **`DTYPE_GM_C`**：GM 上 C 的元素类型；AIV 在写回前按需稠密 cast（even-cast + `deinterleave`）。

| dtype-a | dtype-b | dtype-c（GM C） |
|---------|---------|------------------|
| f16 | f16 | f32 |
| f16 | f16 | f16 |
| bf16 | bf16 | f32 |
| bf16 | bf16 | bf16 |
| f32 | f32 | f32 |

---

## 内核结构概要（`streamk_mmad_kernel`）

1. **`with tla.cube()`**：L1/L0 双缓冲；normal 全 K 写 GM C；StreamK 段写 workspace（含 Cross-block 第二槽）。`aic_finish`（mode-2）按 C++/tail_splitk 时机 Set：normal-only 提前；mixed 在 W store 后；all-streamk 每 task 后。
2. **`with tla.vector()`**：`cross_core_wait_flag(aic_finish, MTE2)` + mode-0 `aiv_ibarrier`；随后按 StreamK tile ReduceAdd（必要时 densify cast）写回 GM C。归约按 Catlass StreamkMatmul AIV 的方式分摊：只有与产出该 tile 的 AIC 配对的 AIV 参与，tile 的行块在这些 AIV 间连续切分。
3. Host 一次 compile / launch（`arch_scope=aic.c310`），block 数与 AIC 一致；产物为单个 mix kernel（`streamk_mmad_kernel_mix_aic` / `_mix_aiv` 为其 AIC/AIV 两个入口）。

---

## 环境前提

- `cd python/tla_dsl && pip install -e .`（及 MLIR Python 绑定等）。
- **`--run`**：`torch`、`torch_npu`，`ASCEND_HOME_PATH` / CANN 已配置。
- 编译 toolchain 常需正确解析 AIC/AIV template bitcode（勿错误固定单一 `TLA_DSL_HIVM_TEMPLATE_BC`）。

---

## 运行指令

在 **`python/tla_dsl`** 下执行：

```bash
cd python/tla_dsl

# 上板并校验（默认即 --run，精度校验默认开启；--device 默认 2）
python examples/end_to_end/basic_mmad_streamk/basic_mmad_streamk.py --device 0

python examples/end_to_end/basic_mmad_streamk/basic_mmad_streamk.py --run --device 0 \
  --layout-a row --layout-b col \
  --dtype-a f16 --dtype-b f16 --dtype-c f32

python examples/end_to_end/basic_mmad_streamk/basic_mmad_streamk.py --run --all-layouts --device 0
python examples/end_to_end/basic_mmad_streamk/basic_mmad_streamk.py --run --all-mmad-dtypes --device 0
```

常用参数：`--block`、`--sentinel`、`--atol`、`--cache-dir`、`--force-recompile`、`--no-cache`、`--m/--n/--k`、`--no-verify`（跳过 golden 与精度比较，仅 compile/launch）。

与 Catlass `66_ascend950_streamk_matmul` 对齐的 `msprof op` 示例：

```bash
msprof op \
  --kernel-name=streamk_mmad_kernel_mix_aic \
  --launch-count=1 --warm-up=5 --kill=on \
  --output=/tmp/dsl_streamk_perf \
  python examples/end_to_end/basic_mmad_streamk/basic_mmad_streamk.py \
    --run --device 4 --no-verify \
    --layout-a row --layout-b row \
    --dtype-a f32 --dtype-b f32 --dtype-c f32 \
    --m 256 --n 512 --k 1024
```

```bash
python examples/end_to_end/basic_mmad_streamk/basic_mmad_streamk.py --help
```

---

## 成功运行时的终端输出

默认运行会打印 `compile_ok=True`、`host=torch_npu`、`launch_ok=True`、`kernel.o` 路径，以及 `C unchanged?` / `C equals expected matmul?` / `first mismatch=...` 等（与 `m×n×k`、block、`--sentinel`、dtype 有关；golden 为 **Torch** 在 NPU 上的 matmul）。加 **`--no-verify`** 时跳过 golden 与精度比较相关输出。

若失败，请检查 **AIC/AIV bitcode**、`ASCEND_HOME_PATH`、**`torch_npu`** 及 **`--device`**。
