import torch
import torch_npu
import numpy as np
import time
from catlass_mla import CATLASSMLA

"""
- Minimum reproducible error script

1) Run with (TRANSFORM, SYNC) = (0, 0) -> There should be no NaN's in kernel output

2) Run with (TRANSFORM, SYNC) = (1, 1) -> It will transform the qk_nope_head_dim to kv_lora_rank
and call torch.npu.synchronize() after einsum. There should be no NaN's in kernel output

3) Run with (TRANSFORM, SYNC) = (1, 0) -> It will transform the qk_nope_head_dim to kv_lora_rank
and will not call torch.npu.synchronize() after einsum. There should be NaN's in kernel output

"""

TRANSFORM = 1
SYNC = 1

print(f"Running with TRANSFORM = {TRANSFORM} SYNC = {SYNC}")

torch.manual_seed(0)

bsz = 1
n_heads = 128
kv_seqlen = 128

n_kv_heads = 1
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
softmax_scale = 1 / np.sqrt(128)

dtype = torch.float16
device = torch.device("npu:0")

kv_seq_lens = np.array([kv_seqlen] * bsz)

W = 1e-2 * torch.randn(qk_nope_head_dim, kv_lora_rank).to(device)

catlass_score_mla_test = CATLASSMLA(bsz, kv_seq_lens, n_heads, n_kv_heads, kv_lora_rank, qk_rope_head_dim, dtype, device)
_, q_rope, k, k_rope, block_tables, device_mem, lse_idxs = catlass_score_mla_test.generate_input_data(generate_on_the_fly = True)

q_min_range = -1.0
q_max_range = 1.0
q = torch.empty((bsz, n_heads, qk_nope_head_dim),
                    dtype=dtype, device=device
                ).uniform_(q_min_range, q_max_range)

if TRANSFORM:
    print("q_nope shape before transformation:\t", q.shape)
    q = torch.einsum('ijp,pq->ijq', q, W)
    print("q_nope shape after transformation:\t", q.shape)
if SYNC:
    torch.npu.synchronize()

catlass_out = catlass_score_mla_test.run(q, q_rope, k, k_rope, 
                                        kv_seq_lens, block_tables, 
                                        device_mem, 
                                        return_lse = False, 
                                        softmax_scale = softmax_scale)

if torch.isnan(catlass_out).any():
    torch.set_printoptions(threshold=torch.inf)
    print(catlass_out)
    raise ValueError("There is at least 1 NaN in output of CATLASS MLA")
else:
    print("[OK] No NaN in output")

