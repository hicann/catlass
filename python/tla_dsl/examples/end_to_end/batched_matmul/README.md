# Batched Matmul（TLA DSL）

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

Device 侧在 `batch * grid_m * grid_n` 个工作项上 grid-stride，单次 launch。

**Dynamic GM（schema v4）**：Host 对 A/B/C 调用 `mark_layout_dynamic()`；kernel 工作尺寸取自 `mem_*.origin_shape`（`k=A[1]`，`n=C[1]`，`B=B[0]//k`，`M=A[0]//B`）。`SWIZZLE_DIRECTION` 仍由 Host 按 `M>N` 在 compile 前设置。

## 文件

| 文件 | 说明 |
|------|------|
| `batched_matmul_kernels.py` | `@tla.kernel`：batch×MN 分块 + L1/L0 pingpong + flag |
| `batched_matmul.py` | Host：造数、compile/launch、torch golden |
| `test_batched_matmul_precision.py` | 200 条随机 shape 精度泛化 |

## 运行

需已 `bash build.sh`，并设置好 CANN / AscendNPU-IR / `PYTHONPATH`。

```bash
cd python/tla_dsl/examples/end_to_end/batched_matmul

# 默认：batch=5, m=256, n=512, k=1024
python batched_matmul.py --run --device 4

# 自定义
python batched_matmul.py --run --device 4 --batch 4 --m 256 --n 256 --k 256 --block 8

# 扫全部 GM layout × mmad dtype（与 basic_mmad / run_dsl_test.sh 一致）
python batched_matmul.py --run --device 4 --all-layouts --all-dtypes

# 只编译 / 看 TLA IR
python batched_matmul.py --build-only --device 4
python batched_matmul.py --dump-tlair
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
