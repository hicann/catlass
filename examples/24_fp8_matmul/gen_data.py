import numpy as np
import torch
import os
import argparse

def gen_data_fp8(row, col):
    data = torch.randn((row, col),dtype=torch.float32)
    data_e4m3 = data.to(torch.float8_e4m3fn)
    return data_e4m3

def gen_data(M, N, K, transA, transB):
    os.makedirs('./input', exist_ok=True)
    os.makedirs('./output', exist_ok=True)

    a_fp8 = gen_data_fp8(M, K)
    b_fp8 = gen_data_fp8(K, N)

    if(transA == 1):
        a_fp8 = a_fp8.t()
    if(transB == 1):
        b_fp8 = b_fp8.t()
    print("------------------------------a_fp8--------------------------------")
    print(a_fp8)
    print("------------------------------b_fp8--------------------------------")
    print(b_fp8)
    a_np = torch.tensor(a_fp8.flatten().untyped_storage(), dtype = torch.int8).numpy()
    b_np = torch.tensor(b_fp8.flatten().untyped_storage(), dtype = torch.int8).numpy()
    a_np.tofile('./input/a_8.bin')
    b_np.tofile('./input/b_8.bin')

    a_fp16 = a_fp8.to(torch.float16)
    b_fp16 = b_fp8.to(torch.float16)
    if(transA == 1):
        a_fp16 = a_fp8.t().to(torch.float16)
    if(transB == 1):
        b_fp16 = b_fp16.t().to(torch.float16)
    a_fp16_np = a_fp16.numpy()
    b_fp16_np = b_fp16.numpy()
    a_fp16_np.tofile('./input/a_16.bin')
    b_fp16_np.tofile('./input/b_16.bin')
    print("------------------------------c_fp32--------------------------------")
    c_fp32 = a_fp16.to(torch.float32) @ b_fp16.to(torch.float32)    
    c_np = c_fp32.numpy()
    print(c_np)
    print("------------------------生成fp8和fp16数据--------------------------")
    c_np.tofile('./output/expected_data.bin')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('M', type=int)
    parser.add_argument('N', type=int)    
    parser.add_argument('K', type=int)
    parser.add_argument('transA', type=int)
    parser.add_argument('transB', type=int)
    args = parser.parse_args()
    gen_data(args.M, args.N, args.K, args.transA, args.transB)
