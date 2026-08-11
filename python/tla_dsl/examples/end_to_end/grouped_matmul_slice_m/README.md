# Grouped Matmul Slice-M（example 02 / C++ ex60）

| 文件 | 作用 |
|------|------|
| `grouped_matmul_slice_m.py` | Host + `@tla.kernel`（同文件）：造数、compile/launch、torch golden |

## 语义

- A：`(M, K)`，按组高度沿 M 切组
- B：`G` 个 `(K, N)` 紧挨存放，展平为 `(G*K, N)`
- C：`(M, N)`，与 A 同切分
- `group_list`：长度 `G+1` 的 Int32 前缀（含前导 0），`currentM[g] = prefix[g+1]-prefix[g]`
- Device 通过 `tensor[i]` 标量下标读 `group_list`（GM），不能用 Python list 被 `tla.range` 归纳变量索引

$$
C_g = A_g \times B_g,\quad g = 0..G-1
$$

## 约束与实现要点

- 一次 launch 覆盖全部 group
- 组高与组起点须为 `L1_TM`（256）倍数，保证 `tile_view` 可寻址
- 组内 MN：`GemmIdentityBlockSwizzle<3, dir>`；**host 在 compile 前**按 `m/G ≥ n` 设 Zn/Nz（`const_expr`），与 C++ ex60 一致
- 跨组 `start_core_idx` 与 C++ `GroupedMatmulSliceMTla` 一致，做核间负载均衡
- M/N/K/G 运行时来自 dynamic GM `origin_shape`；模块级 `M_DIM`/`N_DIM`/`K_DIM`/`GROUPS` 仅供 host 分配与 compile-time type_args
- L1 双缓冲 + **K 向 soft-pipeline prefetch**（对齐 C++ `BlockMmadPingpongTla`：先装 k=0，循环内预取下一块并与当前 L0 重叠）
- 默认 `ENABLE_UNIT_FLAG=True`（unit-flag 融合 L0C→GM）

Launch 使用 `block_dim=`（CLI `--block`）；不要写 `block=`（会被忽略，实际 BlockDim=1）。

## 运行

需已构建 TLA DSL，并设置好 CANN / AscendNPU-IR / `PYTHONPATH`（见仓库 `set_env` / README）。

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"

python examples/end_to_end/grouped_matmul_slice_m/grouped_matmul_slice_m.py \
  --run --device 4 --m 1024 --n 256 --k 256 --groups 4 --block 8

# 组切分：--group-mode average|random（默认 random）
python examples/end_to_end/grouped_matmul_slice_m/grouped_matmul_slice_m.py \
  --run --device 4 --m 1024 --n 256 --k 256 --groups 4 --group-mode average

# 扫 layout / dtype（与 basic_mmad 一致）
python examples/end_to_end/grouped_matmul_slice_m/grouped_matmul_slice_m.py \
  --run --device 4 --all-layouts --all-dtypes
```

成功时可见 `compile_ok` / `launch_ok` 与 golden 比对结果。

## 默认

| 项 | 值 |
|----|-----|
| Shape | `G=4, M=1024, N=256, K=256` |
| dtype | f16 / f16 / f16（L0C fp32） |
| layout | A/B/C row |
| Tile | L1 `256×256×256`，L0 `256×256×64` |
| Swizzle | OFFSET=3；DIRECTION host 按 `m/G ≥ n` |
| arch_scope | `aic.c310` |
