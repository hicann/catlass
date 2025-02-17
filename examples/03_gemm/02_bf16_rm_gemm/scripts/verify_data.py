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
        
        allCnt = M * N
        false_count = 0
        errors = []
        # 古法比较
        for i in range(allCnt):
            diff = abs(h_exp[i] - h_res[i])
            tolerance = 1e-3 + 1e-3 * abs(h_exp[i])
            if diff > tolerance:
                false_count += 1
            errors.append(diff)
        
        errors = np.array(errors)
        accuracy = ((allCnt - false_count) / allCnt) * 100
        
        print("===========================================================", file=f)
        print("errorCnt:", false_count, file=f)
        print("Total Count:", allCnt, file=f)
        print("Accuracy: {:.2f}%".format(accuracy), file=f)
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
