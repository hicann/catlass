# Batched Matmul（example 01 / C++ ex01·ex67）

对每个 batch 计算

$$
C_b = A_b @ B_b,\quad b = 0,\ldots,B-1
$$

各 batch 的 `(M, N, K)` 相同。GM 排布与 C++ 一致：

| 矩阵 | 逻辑形状 | 展平 2D（交给 kernel） | batch stride |
|------|----------|------------------------|--------------|
| A | `(B, M, K)` | `(B*M, K)` | `M*K` |
| B | `(B, K, N)` | `(B*K, N)` | `K*N` |
| C | `(B, M, N)` | `(B*M, N)` | `M*N` |

| 文件 | 说明 |
|------|------|
| `batched_matmul.py` | Host + `@tla.kernel`（同文件）：造数、compile/launch、torch golden |

## 约束与实现要点

- Device 侧在 `batch * grid_m * grid_n` 个工作项上 grid-stride，单次 launch
- **Dynamic GM（schema v4）**：Host 对 A/B/C 调用 `mark_layout_dynamic()`；kernel 工作尺寸取自 `mem_*.origin_shape`（`k=A[1]`，`n=C[1]`，`B=B[0]//k`，`M=A[0]//B`）
- MN：`GemmIdentityBlockSwizzle<3, dir>`；**host 在 compile 前**按 `M>N` 设 Zn/Nz（`const_expr`），与 C++ ex01/ex67 一致（不是 kernel 内运行时 if）
- 默认 `ENABLE_UNIT_FLAG=True`（unit-flag 融合 L0C→GM）；`--all-layouts` 下非 row/row 时 host 会关掉，避免跨 layout 残留状态
- Tile 对齐 C++ ex67：L1 `256×256×128`，L0 `256×256×32`

Launch 使用 `block_dim=`（CLI `--block`）；不要写 `block=`（会被忽略，实际 BlockDim=1）。

## 运行

需已构建 TLA DSL，并设置好 CANN / AscendNPU-IR / `PYTHONPATH`（见仓库 `set_env` / README）。

```bash
cd "${CATLASS_ROOT}/python/tla_dsl/examples/end_to_end/batched_matmul"

# 默认：batch=5, m=256, n=512, k=1024
python batched_matmul.py --run --device 4

# 自定义
python batched_matmul.py --run --device 4 --batch 4 --m 256 --n 256 --k 256 --block 8

# 扫全部 GM layout × mmad dtype（与 basic_mmad 一致）
python batched_matmul.py --run --device 4 --all-layouts --all-dtypes
```

`--all-layouts`：4 种 `(layout-a, layout-b)`。  
`--all-dtypes`：`f16,f16,f32 | f16,f16,f16 | bf16,bf16,f32 | bf16,bf16,bf16 | f32,f32,f32`。

成功时可见：

```text
compile_ok=True ...
launch_ok=True
C equals batched golden? True
first mismatch=None
```

## 默认

| 项 | 值 |
|----|-----|
| Shape | `B=5, M=256, N=512, K=1024` |
| dtype | f16 / f16 / f16（L0C fp32） |
| layout | A/B/C row |
| Tile | L1 `256×256×128`，L0 `256×256×32` |
| Swizzle | OFFSET=3；DIRECTION host 按 `M>N` |
| arch_scope | `aic.c310` |
