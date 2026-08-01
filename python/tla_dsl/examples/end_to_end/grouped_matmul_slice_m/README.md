# Grouped Matmul Slice-M（example 02）


| 文件 | 作用 |
|------|------|
| `grouped_matmul_slice_m.py` | Host：构造 Int32 `group_list` 前缀 + **单次 launch** + golden |
| `grouped_matmul_slice_m_kernels.py` | 多 group cube GEMM（device 读 `group_list`） |

## 语义

- A：`(M, K)`，按组高度沿 M 切组
- B：`G` 个 `(K, N)` 紧挨存放
- C：`(M, N)`，与 A 同切分
- `group_list`：长度 `G+1` 的 Int32 前缀（含前导 0），`currentM[g] = prefix[g+1]-prefix[g]`
- Device 通过 `tensor[i]` 标量下标读 `group_list`（GM），不能用 Python list 被 `tla.range` 归纳变量索引

$C_g = A_g \times B_g,\quad g = 0..G-1$

## 约束

- 一次 launch 覆盖全部 group
- 组高与组起点须为 `L1_TM`（256）倍数，保证 `tile_view` 可寻址
- 组内 MN 为 identity 映射（无 Zn/Nz swizzle）
- 跨组 `start_core_idx` 与 C++ `GroupedMatmulSliceMTla` 一致，做核间负载均衡
- M/N/K/G 运行时来自 dynamic GM `origin_shape`；模块级 `M_DIM`/`N_DIM`/`K_DIM`/`GROUPS` 仅供 host 分配与 compile-time type_args

## 运行

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"

python examples/end_to_end/grouped_matmul_slice_m/grouped_matmul_slice_m.py \
  --run --device 4 --m 1024 --n 256 --k 256 --groups 4 --block 8
```

## 精度扫测（200 条随机用例）

```bash
cd examples/end_to_end/grouped_matmul_slice_m
python test_grouped_matmul_slice_m_precision.py --device 4 --num-cases 200 --seed 0
```

输入：`torch.randn` + `clamp([-5,5])`；判定：`torch.isclose(rtol=1/128)`。

## 默认

| 项 | 值 |
|----|-----|
| Shape | `G=4, M=1024, N=256, K=256` |
| dtype | f16 / f16 / f16（L0C fp32） |
| layout | A/B/C row |
| Tile | L1 `256×256×256`，L0 `256×256×64` |
| arch_scope | `aic.c310` |
