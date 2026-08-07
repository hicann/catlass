# Matmul EVG 端到端示例

本目录演示 **TLA DSL** 下 Ascend950 GEMM + EVG（Epilogue Visitor Graph）尾处理：Cube 路径与 `basic_mmad` 一致（GM→L1→L0、`tla.mmad`、L1/L0 双缓冲）；尾处理经 FixPipe 后由 AIV 完成融合算子。GM 形状为 **dynamic**（`mark_layout_dynamic`）。对应 Catlass C++ 示例 `64_ascend950_matmul_evg_*`。

| 文件 | 场景 | 数据通路 |
| --- | --- | --- |
| **`matmul_add.py`** | D = A×B + X | L0C→GM workspace + AIV |
| **`matmul_add_ub.py`** | D = A×B + X | L0C→UB + AIV（无 GM workspace） |
| **`matmul_bias.py`** | D = A×B + bias(1×N) | L0C→GM + RowBroadcast |
| **`matmul_leaky_relu.py`** | D = LeakyRelu(A×B)，α=0.1 | L0C→GM + AIV |
| **`matmul_sigmoid.py`** | D = Sigmoid(A×B) | L0C→GM + AIV |
| **`matmul_silu.py`** | D = Silu(A×B) | L0C→GM + AIV |
| **`matmul_tanh.py`** | D = Tanh(A×B) | L0C→GM + AIV |

每个文件含 `@tla.kernel` 与 host CLI。矩阵尺寸、dtype 与默认值以源码与 **`--help`** 为准。

---

## 问题规模与分块

- 默认逻辑 GEMM：`m × n × k` = **`256 × 256 × 256`**（`--m` / `--n` / `--k`）。
- L1：`l1_tm × l1_tn × l1_tk` = **256×256×128**；L0：`l0_tm × l0_tn × l0_tk` = **256×256×32**。
- Host 整图一次 launch；多 MN tile 由 kernel 内按 `block_idx` 调度。
- EVG UB 多缓冲深度：模块常量 `EVG_UB_STAGES`（默认 2）。
- `--block-dim`：AIC block 数；默认 `-1` 表示取设备 AIC 核数。

---

## Host 与 GM 张量

需要 **PyTorch**、**torch_npu**、Ascend 运行时：

1. `tla.initialize(device=<id>)`；`torch.npu.set_device(<id>)`。
2. 在 NPU 上构造 A / B / 输出（及 workspace、bias 等辅张量）。
3. **`catlass.runtime.from_dlpack`** → `tla.Tensor`，再 `mark_layout_dynamic()`，供 `tla.compile` / launch。
4. `--layout-a` / `--layout-b`：`row` 或 `col`；输出与 workspace 固定 row-major。

---

## 元素类型

- Cube 在 L0C 上 **fp32 累加**；`dtype-a` 与 `dtype-b` 须相同。
- `dtype-c` 为 GM 输出（及 workspace / 辅输入）元素类型：

| 算子 | dtype-a / dtype-b | dtype-c |
| --- | --- | --- |
| add, bias, leaky_relu, sigmoid, silu | f16 / bf16 / f32 | f16 或 f32 |
| add_ub, tanh | f16 / bf16 / f32 | **仅 f32** |

CLI 默认 `--dtype-a/b/c` 均为 **f32**。

---

## 环境前提

- `cd python/tla_dsl && pip install -e .`（及 MLIR Python 绑定等）。
- `torch`、`torch_npu`，`ASCEND_HOME_PATH` / CANN 已配置。
- Mix kernel 需同时提供 AIC + AIV bitcode，例如：

```bash
export TLA_DSL_HIVM_TEMPLATE_BC=\
"$PWD/csrc/mlir/build/bc/meta_op.aic.c310.bc,$PWD/csrc/mlir/build/bc/meta_op.aiv.c310.bc"
```

---

## 运行指令

在 **`python/tla_dsl`** 下执行：

```bash
cd python/tla_dsl

python examples/end_to_end/basic_mmad_evg/matmul_add.py --device 0 \
  --m 256 --n 256 --k 256 \
  --layout-a row --layout-b row \
  --dtype-a f16 --dtype-b f16 --dtype-c f32

# 其余变体替换文件名即可
python examples/end_to_end/basic_mmad_evg/matmul_add_ub.py --device 0
python examples/end_to_end/basic_mmad_evg/matmul_bias.py --device 0
python examples/end_to_end/basic_mmad_evg/matmul_tanh.py --device 0
```

常用参数：`--block-dim`、`--cache-dir`、`--force-recompile`、`--no-cache`。详见各脚本 `--help`。
