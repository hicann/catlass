import os
import torch
import torch_npu
import numpy as np
import sys


torch.npu.set_device(1)
np.random.seed(3)
torch.manual_seed(3)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype = torch.int64)
    for i in range(len(seqlens_list) + 1):
        cu[i] = sum(seqlens_list[:i])
    return cu


def gen_data(N1, N2, D, list_seq):
    g = N1 / N2
    scale = 1 / (D ** 0.5)
    pre_tocken = 65536
    next_tocken = 0

    seqlens_list_q = np.array(list_seq)
    seqlens_list_k = np.array(list_seq)
    B = len(list_seq)

    keep_prob = 1.0
    cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
    cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
    S1 = seqlens_list_q.sum()
    S2 = seqlens_list_k.sum()

    print("S1: ", S1)
    print("S2: ", S2)

    pttype = torch.float16
    limit = 2
    q = limit * (torch.rand([S1, N1, D]) - 0.5).to(pttype)
    k = limit * (torch.rand([S2, N2, D]) - 0.5).to(pttype)
    v = limit * (torch.rand([S2, N2, D]) - 0.5).to(pttype)
    dout = limit * (torch.rand([S1, N1, D]) - 0.5).to(pttype)

    cu_seq_len_list = cu_seqlens_q[1:].cpu().numpy().tolist()
    cu_seq_kvlen_list = cu_seqlens_k[1:].cpu().numpy().tolist()
    
    q = q.npu()
    k = k.npu()
    v = v.npu()
    dout = dout.npu()
    queryRight = q.clone().npu()
    # Mark: modify keyRight as workspace float output
    keyRight = k.clone().float().npu()
    keyRight = torch.rand([16, 128, 128]).float().npu()
    queryRight = torch.rand([16, 128, 256]).to(torch.float16).npu()

    print("q.shape ", q.shape)
    print("k.shape ", k.shape)
    print("v.shape ", v.shape)
    print("dout.shape ", dout.shape)
    print("queryRight.shape ", queryRight.shape)
    print("queryRight.dtype", queryRight.dtype)
    print("keyRight.shape", keyRight.shape)
    print("keyRight.dtype", keyRight.dtype)

    print("cu_seq_len_list is ", cu_seq_len_list)
    print("cu_seq_kvlen_list is ", cu_seq_kvlen_list)

    atten_mask_npu = (torch.triu(torch.ones([2048, 2048]), diagonal=1)).to(torch.bool).npu()

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    torch.npu.synchronize()
    npu_rst = torch_npu.npu_fusion_attention(
            q, k, v, N1,
            pse=None,
            padding_mask=None,
            atten_mask=atten_mask_npu,
            scale=scale,
            keep_prob=keep_prob,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seq_len_list),
            actual_seq_kvlen=tuple(cu_seq_kvlen_list),
            pre_tockens=pre_tocken,
            next_tockens=next_tocken,
            inner_precise=0,
            sparse_mode=2,
            prefix=None)
    out_npu = npu_rst[0]
    x_max_npu = npu_rst[1]
    x_sum_npu = npu_rst[2]
    torch.npu.synchronize()
    out_npu.backward(dout)
    dq_golden_npu = q.grad
    dk_golden_npu = k.grad
    dv_golden_npu = v.grad
    torch.npu.synchronize()

    print("soft_max_max shape ", x_max_npu.shape, x_max_npu.dtype)
    print("soft_max_sum shape ", x_sum_npu.shape, x_sum_npu.dtype)
    print("attention in shape ", out_npu.shape, out_npu.dtype)

    print("dq_golden shape ", dq_golden_npu.shape, dq_golden_npu.dtype)
    print("dk_golden shape ", dk_golden_npu.shape, dk_golden_npu.dtype)
    print("dv_golden shape ", dv_golden_npu.shape, dv_golden_npu.dtype)

    q.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "q.bin"))
    k.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "k.bin"))
    v.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "v.bin"))
    dout.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "dout.bin"))
    queryRight.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "queryRight.bin"))
    keyRight.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "keyRight.bin"))
    atten_mask_npu.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "atten_mask.bin"))
    x_max_npu.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "row_max.bin"))
    x_sum_npu.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "row_sum.bin"))
    out_npu.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "out.bin"))
    np.array(cu_seq_len_list).tofile(os.path.join(WORKSPACE, "data", "cu_seq_qlen.bin"))
    np.array(cu_seq_kvlen_list).tofile(os.path.join(WORKSPACE, "data", "cu_seq_kvlen.bin"))

    dq_golden_npu.cpu().to(torch.float).numpy().tofile(os.path.join(WORKSPACE, "data", "dq_golden.bin"))
    dk_golden_npu.cpu().to(torch.float).numpy().tofile(os.path.join(WORKSPACE, "data", "dk_golden.bin"))
    dv_golden_npu.cpu().to(torch.float).numpy().tofile(os.path.join(WORKSPACE, "data", "dv_golden.bin"))


if __name__ == '__main__':
    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)

    N1 = int(sys.argv[1])
    N2 = int(sys.argv[2])
    D = int(sys.argv[3])
    list_seq = list(map(int, sys.argv[4:]))

    gen_data(N1, N2, D, list_seq)