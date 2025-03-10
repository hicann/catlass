import torch
import torch_npu
import argparse
import numpy as np
import ast
torch.npu.set_device(6)
# torch.ops.load_library("/data00/JinRuiqi/15/test/libxpu_ops.so")

def run_profiling(problemCount, mList, nList, kList):
    for _ in range(5):
        mat_a = torch.randn(4096, 4096).to(dtype=torch.bfloat16).npu()
        mat_b = torch.randn(4096, 4096).to(dtype=torch.bfloat16).npu()
        c = torch.matmul(mat_a, mat_b).cpu()
    torch.npu.synchronize()

    # for _ in range(20):
    #     mat_a = torch.randn(M, K).to(dtype=torch.int8).npu()
    #     mat_b = torch.randn(K, N).to(dtype=torch.int8).npu()
    #     scale = torch.randn(N).to(dtype=torch.bfloat16).npu()
    #     bias = torch.randn(N).to(dtype=torch.bfloat16).npu()
    #     pertoken_scale = torch.randn(M).to(dtype=torch.float).npu()
    #     expect = torch_npu.npu_quant_matmul(mat_a, mat_b, scale, offset=None, bias=bias, pertoken_scale=pertoken_scale, output_dtype=torch.bfloat16).cpu()
    # torch.npu.synchronize()

    for _ in range(5):
        x = []
        for i in range(problemCount):
            x_tensor = torch.randn(mList[i], kList[i], device='npu', dtype=torch.int8)
            x.append(x_tensor)
        
        weight = []
        for i in range(problemCount):
            weight_tensor = torch.randn(kList[i], nList[i], device='npu', dtype=torch.int8)
            weight.append(weight_tensor)

        bias = []
        for i in range(problemCount):
            bias_tensor = torch.randn(nList[i], device='npu', dtype=torch.int32)
            bias.append(bias_tensor)

        scale = []
        for i in range(problemCount):
            scale_tensor = torch.randn(nList[i], device='npu', dtype=torch.int64)
            scale.append(scale_tensor)
            
        group_list = None
        split_item = 0
        npu_out = torch_npu.npu_grouped_matmul(x, weight, bias=bias, scale=scale, group_list=group_list, split_item=split_item)
    torch.npu.synchronize()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('problemCount', action='store', type=int)
    parser.add_argument('mList', action='store', type=str)
    parser.add_argument('nList', action='store', type=str)
    parser.add_argument('kList', action='store', type=str)
    args = parser.parse_args()

    problemCount = args.problemCount
    mStr = args.mList
    mArray = list(ast.literal_eval(mStr))
    mList = np.array(mArray, dtype=np.uint32)
    print(mList)
    nStr = args.nList
    nArray = list(ast.literal_eval(nStr))
    nList = np.array(nArray, dtype=np.uint32)
    print(nList)
    kStr = args.kList
    kArray = list(ast.literal_eval(kStr))
    kList = np.array(kArray, dtype=np.uint32)
    print(kList)

    run_profiling(problemCount, mList, nList, kList)

    