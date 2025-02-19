import numpy as np
import argparse
import sys
NP_DATA_TYPE = np.int32

def compareOutputData(M, N, K):
    print("--- Test Begin ---\n")
    with open("./data/output.txt","a") as f:
        print("---- INT8 -- RowMajor -- EpilogueGemm ---- \n", file = f)
        print("  M : {} , N : {} , K : {} \n".format(M , N, K), file = f)
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        h_exp = np.fromfile("./data/output/exp_res.bin",dtype=NP_DATA_TYPE)
        h_res = np.fromfile("./data/output/our_res.bin",dtype=NP_DATA_TYPE)
        h_exp.reshape((M * N,1))  # 都是一列数据
        h_res.reshape((M * N,1))
        # atol 是绝对容忍度
        # rtol 是相对容忍度
        compare_res = np.isclose(h_exp,h_res,atol=1e-3,rtol=1e-3,equal_nan=False)
        allCnt = M * N
        print("===========================================================", file=f)
        false_count = (~compare_res).sum().item()
        print("errorCnt:", false_count, file=f)
        print("Total Count:", allCnt, file=f)
        print("Accuracy: {:.2f}%".format(((allCnt - false_count) / allCnt) * 100), file=f)
        print("===========================================================", file=f)
        # print("\n", file=f)
        # print(h_res.reshape((M,N)), file=f)
        # print("\n", file=f)
        # print(h_exp.reshape((M,N)), file=f)
    print("--- Test End ---\n")

if __name__ == "__main__":
    M = 32
    N = 32
    K = 32
    if len(sys.argv) > 1:
        M = int(sys.argv[1])
    if len(sys.argv) > 2:
        N = int(sys.argv[2])
    if len(sys.argv) > 3:
        K = int(sys.argv[3])
    compareOutputData(M, N, K)
    