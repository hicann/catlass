"""Compile-time knobs for StreamK MMAD (host mutates before ``tla.compile``)."""

from __future__ import annotations

import catlass as tla

# GM / tile element types. Host mutates DTYPE_A/B/GM_C before ``tla.compile``.
# Cube MMAD always accumulates in fp32 on L0C (``DTYPE_C``). ``DTYPE_GM_C`` is the
# GM C element type; AIV casts before the final GM store when narrowed.
DTYPE_A = tla.Float16
DTYPE_B = tla.Float16
DTYPE_C = tla.Float32
DTYPE_GM_C = tla.Float32

ENABLE_UNIT_FLAG = True

# Default problem: one 256x256 MN tile with K split across the StreamK cores.
# ``BLOCK_DIM`` is the compile-time schedule knob; the kernel strides tasks by
# ``tla.arch.block_dim()`` so it tracks the launch block count.
m = 256
n = 256
k = 512
l1_tm = 256
l1_tn = 256
l1_tk = 128
l0_tm = 256
l0_tn = 256
l0_tk = 32
# Fallback when host has not yet resolved device AIC count (e.g. dump-tlair).
BLOCK_DIM = 2
# Zn swizzle offset; must match ``get_block_coord``.
SWIZZLE_OFFSET = 3

AIV_TILE_M = 16
# Vector sub-blocks per AIC: the mix kernel pairs each AIC with two AIVs.
AIV_SUB_BLOCK_NUM = 2
# One Ascend VF register is 256B → 64 f32 lanes. Chunk UB tiles to this size.
AIV_REG_M = 1
AIV_REG_N = 64
AIV_M_CHUNKS = AIV_TILE_M // AIV_REG_M
AIV_N_CHUNKS = l1_tn // AIV_REG_N
# f32→f16/bf16 even-cast packs into even lanes; densify by casting two 64-f32
# chunks then deinterleave into one dense 128-lane narrow vector.
AIV_DENSE_N = AIV_REG_N * 2
AIV_N_DENSE_CHUNKS = AIV_N_CHUNKS // 2
