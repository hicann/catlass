import sys
import os
import time
import torch 
import torch_npu
import numpy as np
from typing import Tuple
import argparse
import matplotlib.pyplot as plt

from gen_data import TestPagedMLAttention
from utils import calc_num_ops, calc_mem_size, calc_cube_throughput, calc_hbm_throughput
from torch_catlass_attention import mla, catlass_mla_prepare
from catlass_mla_utils import catlass_select_lse, catlass_kernel_prepare, catlass_score_mla

torch.manual_seed(0)


class CATLASSMLA:
    def __init__(self, bsz, kv_seqlens, n_heads, n_kv_heads, kv_lora_rank, qk_rope_head_dim, dtype, device):
        self.bsz = bsz
        self.kv_seqlens = np.array(kv_seqlens)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.dtype = dtype
        self.device = device
        self.max_kv_seqlen = max(self.kv_seqlens)

        self.actseqlen = [1 for b in range(bsz)]
        self.actseqlenkv = kv_seqlens

        assert len(kv_seqlens) == self.bsz

        self.scale = 1/np.sqrt(128.0)

    def generate_input_data(self, verbose = False, generate_on_the_fly = True):
        kv_heads = 1
        q_seqlen = 1
        block_size = 128
        batch = self.bsz
        kv_seqlen = self.max_kv_seqlen
        embedding_size_nope = self.kv_lora_rank
        embedding_size_rope = self.qk_rope_head_dim
        q_seqlen_list = [q_seqlen] * batch
        kv_seqlen_list = self.kv_seqlens
        num_head = self.n_heads

        num_blocks = (batch * kv_seqlen) // block_size

        testObj = TestPagedMLAttention()
        testObj.check_attr(batch, q_seqlen, kv_seqlen, num_blocks, block_size)

        mask_type = 0 # Default
        gen_data_dtype = np.float16

        if verbose:
            print(f"Input generation starts for (bsz, n_heads, kv_seq_len) = {batch, num_head, kv_seqlen}")

        gen_data_params = testObj.GenDataParams(
            q_seqlen_list , kv_seqlen_list, num_head,
            kv_heads, embedding_size_nope, embedding_size_rope,
            num_blocks, block_size, mask_type, gen_data_dtype)

        (query_nope, query_rope, kv_nope_cache, kv_rope_cache,
        block_tables, ref_output) = testObj.calc_data(gen_data_params, self.device, save_file=False, calc_ref = False)

        torch.npu.set_device(self.device)
        
        q_nope_pt = torch.from_numpy(query_nope).to(self.device)
        q_rope_pt = torch.from_numpy(query_rope).to(self.device)
        k_nope_pt = torch.from_numpy(kv_nope_cache).to(self.device)
        k_rope_pt = torch.from_numpy(kv_rope_cache).to(self.device)
        ref_output_pt = torch.from_numpy(ref_output).to(self.device)
        block_tables_pt = torch.from_numpy(block_tables).to(self.device)
        block_tables_pt = block_tables_pt.reshape(-1).to(torch.int32)
        dtype_str = "float16"

        if verbose:
            print(f"Inputs are generated (bsz, n_heads, kv_seq_len) = {batch, num_head, kv_seqlen}")
        
        device_mem, lse_idxs = catlass_kernel_prepare(batch, num_head,
                                                      embedding_size_nope, embedding_size_rope,
                                                      num_blocks, block_size, np.array(kv_seqlen_list), self.device, dtype_str)
            
        return q_nope_pt, q_rope_pt, k_nope_pt, k_rope_pt, block_tables_pt, device_mem, lse_idxs

    def run(self, q, q_rope, k, k_rope, kv_seq_lens, block_tables, device_mem, softmax_scale = 0.08838834764831843, return_lse = True, lse_idxs = None, dtype_str = "float16"):
        return catlass_score_mla(q, q_rope, k, k_rope, kv_seq_lens, block_tables, device_mem, softmax_scale, return_lse, dtype_str, lse_idxs)

    def perf(self):
        warm_up = 5
        n_repeat = 25

        kv_seqlen = self.max_kv_seqlen

        q, q_rope, k, k_rope, block_tables, device_mem, lse_idxs = self.generate_input_data(verbose = False, generate_on_the_fly = True)
        kv_seq_lens = self.kv_seqlens

        for i in range(warm_up):
            self.run(q, q_rope, k, k_rope, kv_seq_lens, block_tables, device_mem, return_lse = False, lse_idxs = lse_idxs)
            torch.npu.synchronize() # <-- Without this, benchmark runs may fail

        start = [torch.npu.Event(enable_timing=True) for i in range(n_repeat)]
        end = [torch.npu.Event(enable_timing=True) for i in range(n_repeat)]

        cache = torch.empty(int(256e6), dtype=torch.int, device=self.device)
        torch.npu.synchronize()

        for i in range(n_repeat):
            cache.zero_()

            start[i].record()
            self.run(q, q_rope, k, k_rope, kv_seq_lens, block_tables, device_mem, return_lse = False, lse_idxs = lse_idxs)
            end[i].record()
            torch.npu.synchronize() # <-- Without this, benchmark runs may fail

        torch.npu.synchronize()
        elapsed = [start[i].elapsed_time(end[i]) for i in range(n_repeat)]
        m_elapsed = np.median(elapsed)

        return m_elapsed 

    def ref_mla_absorb(self, q_nope, q_pe, kv_cache, pe_cache, softmax_scale=None, return_lse = True):
        
        device = self.device

        q_nope_ref = q_nope.reshape(self.bsz, self.n_heads, self.kv_lora_rank).unsqueeze(1).to(device)
        q_rope_ref = q_pe.reshape(self.bsz, self.n_heads, self.qk_rope_head_dim).unsqueeze(1).to(device)
        kv_cache_ref = kv_cache.reshape(self.bsz, self.max_kv_seqlen, self.kv_lora_rank).to(device)
        pe_cache_ref = pe_cache.reshape(self.bsz, self.max_kv_seqlen, self.qk_rope_head_dim).to(device)

        kv_lens_torch = torch.from_numpy(self.kv_seqlens).to(device)
        pos = torch.arange(self.max_kv_seqlen, device=device).view(1, 1, 1, self.max_kv_seqlen)
        kv_mask = pos >= kv_lens_torch.view(self.bsz, 1, 1, 1)

        scores = torch.einsum("bshc,btc->bsht", q_nope_ref, kv_cache_ref)
        scores += torch.einsum("bshr,btr->bsht", q_rope_ref, pe_cache_ref)
        if softmax_scale is not None:
            scores = scores * softmax_scale
        if kv_mask is not None:
            scores = scores.masked_fill(kv_mask, float("-inf"))
        max_scores, _ = torch.max(scores, dim=-1, keepdim=True)
        lse = torch.log(torch.sum(torch.exp(scores - max_scores), dim=-1, keepdim=True)) + max_scores
        probs = torch.exp(scores - lse)
        x = torch.einsum("bsht,btc->bshc", probs, kv_cache_ref)

        if return_lse:
            return x, lse.squeeze(-1)
        else:
            return x

def benchmark(bsz_array, max_seqlen_array, n_heads_array):
    n_kv_heads = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 64

    dtype = torch.bfloat16
    device = "npu:0"

    print(f"\n=== CATLASS MLA Benchmark Starts ===\n")

    for n_heads in n_heads_array:
        for bsz in bsz_array:
            for seqlen in max_seqlen_array:
                kv_seqlens = [seqlen] * bsz
                catlass = CATLASSMLA(bsz, kv_seqlens, n_heads, n_kv_heads, kv_lora_rank, qk_rope_head_dim, dtype, device)
                m_elapsed = catlass.perf()
                print("n_heads: {: <5}\t bsz: {: <5}\t seqlen: {: <5}\t elapsed: {:.2f} ms\t".format(n_heads, bsz, seqlen, m_elapsed))
            print("-" * 20)


def functional_test(bsz, max_kv_seqlen, n_heads):
    n_kv_heads = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    softmax_scale = 1 / np.sqrt(128)

    dtype = torch.float16
    device = torch.device("npu:0")

    min_kv = 128
    max_kv = max_kv_seqlen

    choices = np.arange(min_kv, max_kv + 1, 128, dtype=np.int32)
    kv_seq_lens = np.random.choice(choices, size=bsz, replace=True).astype(np.int32)

    if max_kv not in kv_seq_lens:
        rand_idx = np.random.randint(0, bsz)
        kv_seq_lens[rand_idx] = max_kv

    catlass_score_mla_test = CATLASSMLA(bsz, kv_seq_lens, n_heads, n_kv_heads, kv_lora_rank, qk_rope_head_dim, dtype, device)
    q, q_rope, k, k_rope, block_tables, device_mem, lse_idxs = catlass_score_mla_test.generate_input_data(generate_on_the_fly = True)

    ref_out, ref_lse = catlass_score_mla_test.ref_mla_absorb(q, q_rope, k, k_rope, softmax_scale=softmax_scale)
    
    torch.npu.synchronize()
    catlass_out, catlass_lse = catlass_score_mla_test.run(q, q_rope, k, k_rope, 
                                                          kv_seq_lens, block_tables, 
                                                          device_mem, 
                                                          return_lse = True, lse_idxs = lse_idxs, 
                                                          softmax_scale = softmax_scale)

    out_diff = torch.abs(catlass_out.cpu().flatten() - ref_out.cpu().flatten()).max().item()
    lse_diff = torch.abs(catlass_lse.cpu().flatten() - ref_lse.cpu().flatten()).max().item()

    assert torch.allclose(catlass_out.cpu().flatten().to(torch.float32), ref_out.cpu().flatten().to(torch.float32), rtol=0, atol=0.01)
    assert torch.allclose(catlass_lse.cpu().flatten().to(torch.float32), ref_lse.cpu().flatten().to(torch.float32), rtol=0, atol=0.05)

    print("Test OK")

    print(f"Out diff: {out_diff}")
    print(f"LSE diff: {lse_diff}")

    print("(min, max) LSE:", catlass_lse.min().item(), catlass_lse.max().item())

    os.makedirs("scatters", exist_ok=True)

    l_host = catlass_lse.cpu().flatten()
    lse_gt = ref_lse.cpu().flatten()

    plt.scatter(l_host, lse_gt, s=5, alpha=0.7)
    plt.xlabel('CATLASS LSE')
    plt.ylabel('Ground Truth LSE')
    plt.title(f'CATLASS vs GT (bsz={bsz}, n_heads={n_heads}, maxseqlen={max_kv_seqlen})\n(Result Diff: {out_diff})')
    plt.grid()
    plt.savefig(f'scatters/scatter_plot_catlass_vs_gt_lse_bsz{bsz}_nheads{n_heads}_maxseqlen{max_kv_seqlen}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--bench", action="store_true", help="Run benchmarks")
    parser.add_argument("--bsz", type=int, nargs='+', help="Batch sizes (space separated list)")
    parser.add_argument("--seqlen", type=int, nargs='+', help="Sequence lengths (space separated list)")
    parser.add_argument("--n_heads", type=int, default=128, help="Number of attention heads")

    args = parser.parse_args()

    bsz = args.bsz 
    max_seqlen = args.seqlen
    n_heads = args.n_heads

    if args.test == True:
        n = n_heads
        b = bsz[0]
        m = max_seqlen[0]

        print(f"== Functional test for (n_heads, bsz, maxseqlen) = {(n, b, m)}")
        functional_test(b, m, n)
        print("-" * 20, "\n")

    if args.bench == True:
        benchmark(bsz, max_seqlen, n_heads)
