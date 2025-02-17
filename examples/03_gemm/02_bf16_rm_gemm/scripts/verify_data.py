import numpy as np
import argparse
import sys
from ml_dtypes import bfloat16

NP_DATA_TYPE = bfloat16

def compareOutputData(M, N, K):
    print("--- Test Begin ---\n")
    with open("./data/output.txt", "w") as f:
        print("---- BF16 -- RowMajor -- Gemm ---- \n", file=f)
        print("  M : {} , N : {} , K : {} \n".format(M, N, K), file=f)
        
        np.set_printoptions(threshold=np.inf, precision=3, suppress=True)
        
        h_exp = np.fromfile("./data/output/exp_res.bin", dtype=NP_DATA_TYPE)
        h_res = np.fromfile("./data/output/our_res.bin", dtype=NP_DATA_TYPE)
        
        # 先转换为 float32 避免 bfloat16 计算精度问题
        h_exp = h_exp.astype(np.float32)
        h_res = h_res.astype(np.float32)
        
        allCnt = M * N
        compare_res = np.isclose(h_exp, h_res, atol=1e-3, rtol=1e-3, equal_nan=False)
        
        false_count = np.sum(~compare_res)  # 误差超过阈值的个数
        # errors = np.abs(h_exp - h_res)
        # mean_error = np.mean(errors)
        # max_error = np.max(errors)
        # std_error = np.std(errors)
        accuracy = ((allCnt - false_count) / allCnt) * 100
        
        print("===========================================================", file=f)
        print("errorCnt:", false_count, file=f)
        print("Total Count:", allCnt, file=f)
        print("Accuracy: {:.2f}%".format(accuracy), file=f)
        # print("Mean Error: {:.6f}".format(mean_error), file=f)
        # print("Max Error: {:.6f}".format(max_error), file=f)
        # print("Std Dev Error: {:.6f}".format(std_error), file=f)
        print("===========================================================", file=f)
    
    print("--- Test End ---\n")

if __name__ == "__main__":
    M, N, K = 32, 32, 32
    if len(sys.argv) > 1:
        M = int(sys.argv[1])
    if len(sys.argv) > 2:
        N = int(sys.argv[2])
    if len(sys.argv) > 3:
        K = int(sys.argv[3])
    compareOutputData(M, N, K)
