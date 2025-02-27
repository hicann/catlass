import numpy as np
import argparse
import sys
NP_DATA_TYPE = np.float32

def compareOutputData(M, N):
    with open("./data/output.txt","a") as f:
        print("---- FP32 -- ColumnMajor -- Gemv ---- \n", file=f)
        print("  M : {} , N : {} \n".format(M , N), file=f)
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        h_exp = np.fromfile("./data/output/exp_res.bin",dtype=NP_DATA_TYPE)
        h_res = np.fromfile("./data/output/our_res.bin",dtype=NP_DATA_TYPE)
        h_exp.reshape((M,1))  # 都是一列数据
        h_res.reshape((M,1))
        # atol 是绝对容忍度
        # rtol 是相对容忍度
        compare_res = np.isclose(h_exp,h_res,atol=1e-3,rtol=1e-3,equal_nan=False)
        allCnt = M
        print("===========================================================", file=f)
        false_count = (~compare_res).sum().item()
        print("errorCnt:", false_count, file=f)
        print("Total Count:", allCnt, file=f)
        print("Accuracy: {:.2f}%".format(((allCnt - false_count) / allCnt) * 100), file=f)
        print("===========================================================", file=f)

        # 打印出错误的索引和对应的值
        error_indices = np.where(~compare_res)[0]  # 找出不匹配的索引
        if len(error_indices) > 0:
            print("Errors at the following indices and values:", file=f)
            for idx in error_indices:
                print(f"Index: {idx}, Expected: {h_exp[idx]}, Got: {h_res[idx]}", file=f)
                
        # print("\n", file=f)
        # print(h_res.reshape((M,1)), file=f)
        # print("\n", file=f)
        # print(h_exp.reshape((M,1)), file=f)
if __name__ == "__main__":
    M = 32
    N = 32
    if len(sys.argv) > 1:
        M = int(sys.argv[1])
    if len(sys.argv) > 2:
        N = int(sys.argv[2])

    compareOutputData(M, N)
    