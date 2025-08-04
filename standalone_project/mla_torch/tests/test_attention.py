import numpy as np
import torch
import torch_npu
from ml_dtypes import bfloat16

from gen_data import TestPagedMLAttention
from torch_catlass_attention import mla


def main(
    device="npu:0",
    batch=1,
    q_seqlen=1,
    kv_seqlen=128,
    num_head=16,
    num_blocks=16,
    block_size=128,
    embedding_size_nope=512,
    embedding_size_rope=64,
    mask_type=0,
    dtype_str="float16",
    verbose=False
    ):

    if dtype_str == "float16":
        dtype = np.float16
    elif dtype_str == "bf16":
        dtype =  bfloat16
    else:
        raise NotImplementedError

    # get input tensor and reference result
    kv_heads = 1
    q_seqlen_list = [q_seqlen] * batch
    kv_seqlen_list = [kv_seqlen] * batch

    testObj = TestPagedMLAttention()
    testObj.check_attr(batch, q_seqlen, kv_seqlen, num_blocks, block_size)

    gen_data_params = testObj.GenDataParams(
        q_seqlen_list, kv_seqlen_list, num_head,
        kv_heads, embedding_size_nope, embedding_size_rope,
        num_blocks, block_size, mask_type, dtype)
    print("gen_data_params: ", gen_data_params)

    (query_nope, query_rope, kv_nope_cache, kv_rope_cache,
     block_tables, ref_output) = testObj.calc_data(gen_data_params, save_file=False)

    # convert to torch tensor on device
    torch.npu.set_device(device)

    q_nope_pt = torch.from_numpy(query_nope).to(device)
    q_rope_pt = torch.from_numpy(query_rope).to(device)
    k_nope_pt = torch.from_numpy(kv_nope_cache).to(device)
    k_rope_pt = torch.from_numpy(kv_rope_cache).to(device)
    ref_output_pt = torch.from_numpy(ref_output).to(device)
    block_tables_pt = torch.from_numpy(block_tables).to(device)

    if verbose:
        print("q_nope_pt.shape", q_nope_pt.shape)
        print("q_rope_pt.shape", q_rope_pt.shape)
        print("k_nope_pt.shape", k_nope_pt.shape)
        print("k_rope_pt.shape", k_rope_pt.shape)
        print("block_tables_pt.shape", block_tables_pt.shape)
        print("block_tables_pt", block_tables_pt)
        print("ref_output_pt.shape", ref_output_pt.shape)

    # launch custom kernel
    torch.npu.synchronize()
    result = mla(
        q_nope_pt, q_rope_pt, k_nope_pt, k_rope_pt, block_tables_pt, dtype_str
        )
    torch.npu.synchronize()

    absdiff = torch.abs(result - ref_output_pt)
    max_diff = float(absdiff.max().cpu())
    mean_diff =  float(absdiff.mean().cpu())

    if verbose:
        print("ref_output_pt", ref_output_pt)
        print("result", result)
        print("result absdiff:", absdiff)

    # TODO: convert to pytest after fixing all cases
    print("absdiff.max(), absdiff.mean()", max_diff, mean_diff)
    print("max_diff < 1e-3: ", max_diff < 1e-3)
    print("mean_diff < 1e-4: ", mean_diff < 1e-4)

    print("\n =========== \n")

if __name__ == "__main__":
    # correct cases
    main()
    main(num_head=128)

    # TODO: debug those incorrect cases
    # main(batch=2, verbose=True)
    # main(kv_seqlen=256, verbose=True)
    # main(batch=4, num_blocks=64, kv_seqlen=1024)

    # main(dtype_str="bf16")
