"""StreamK schedule constants shared by the host and the kernel."""

from __future__ import annotations

import streamk_config as cfg


def ceil_div(numer: int, denom: int) -> int:
    return (numer + denom - 1) // denom


def schedule_constants(
    m_val: int | None = None,
    n_val: int | None = None,
    k_val: int | None = None,
    l1_tm_val: int | None = None,
    l1_tn_val: int | None = None,
    l1_tk_val: int | None = None,
    block_dim: int | None = None,
) -> dict[str, int]:
    # Read module-level knobs at call time (host mutates them before compile).
    m_val = cfg.m if m_val is None else m_val
    n_val = cfg.n if n_val is None else n_val
    k_val = cfg.k if k_val is None else k_val
    l1_tm_val = cfg.l1_tm if l1_tm_val is None else l1_tm_val
    l1_tn_val = cfg.l1_tn if l1_tn_val is None else l1_tn_val
    l1_tk_val = cfg.l1_tk if l1_tk_val is None else l1_tk_val
    block_dim = cfg.BLOCK_DIM if block_dim is None else block_dim
    loops_m = ceil_div(m_val, l1_tm_val)
    loops_n = ceil_div(n_val, l1_tn_val)
    loops_k = ceil_div(k_val, l1_tk_val)
    total_mn = loops_m * loops_n
    streamk_blocks = total_mn % block_dim
    normal_blocks = total_mn - streamk_blocks
    k_tile_num_per_core = (streamk_blocks * loops_k) // block_dim if streamk_blocks else 0
    k_tile_remain = (streamk_blocks * loops_k) % block_dim if streamk_blocks else 0
    core_loops = (total_mn // block_dim) * block_dim + min(
        streamk_blocks * loops_k, block_dim
    )
    streamk_cores = core_loops - normal_blocks
    return {
        "loops_m": loops_m,
        "loops_n": loops_n,
        "loops_k": loops_k,
        "total_mn": total_mn,
        "streamk_blocks": streamk_blocks,
        "normal_blocks": normal_blocks,
        "streamk_cores": streamk_cores,
        "core_loops": core_loops,
        "k_tile_num_per_core": k_tile_num_per_core,
        "k_tile_remain": k_tile_remain,
    }


def workspace_rows(block_dim: int | None = None) -> int:
    """Workspace row count: ``l1_tm`` rows per core slot, two slots per AIC core."""
    if block_dim is None:
        block_dim = cfg.BLOCK_DIM
    return cfg.l1_tm * 2 * block_dim
