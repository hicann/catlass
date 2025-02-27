import numpy as np
import sys
from ml_dtypes import bfloat16
NP_DATA_TYPE = bfloat16

def get_thresholds(M, N, K, dtype=np.half):
    """根据 calc_times (M * N * K) 计算 atol 和 rtol"""
    calc_times = M * N
    
    if dtype == np.float32:
        if calc_times < 2048:
            atol = 1.0 / (1 << 11)
        elif calc_times < 16384:
            atol = 1.0 / (1 << 10)
        else:
            atol = 1.0 / (1 << 9)
        rtol = 1.0 / (1 << 14)
    
    elif dtype == np.half:
        if calc_times < 2048:
            atol = 1.0 / (1 << 8)
        else:
            atol = 1.0 / (1 << 7)
        rtol = 1.0 / (1 << 10)
        
    elif dtype == bfloat16:
        if calc_times < 2048:
            atol = 1.0 / (1 << 9)  # 比 float32 宽松但比 float16 严格
        elif calc_times < 16384:
            atol = 1.0 / (1 << 8)
        else:
            atol = 1.0 / (1 << 7)
        rtol = 1.0 / (1 << 12)  # 介于 float32 和 float16 之间

    elif dtype == np.int32:
        atol = 1e-2
        rtol = 1e-2

    else:
        raise ValueError("Unsupported data type!")

    return atol, rtol

def printDimension(M, N, K, groupIdx):
    with open("./data/output.txt", "a") as f:
        print(f"  M : {M} , N : {N} , K : {K} , groupIdx : {groupIdx} \n", file=f)

def compareOutputData(h_exp, h_res, allMNCnt):
    atol, rtol = get_thresholds(M, N, K, dtype=NP_DATA_TYPE)  # 计算 atol 和 rtol
    np.set_printoptions(threshold=np.inf, precision=3, suppress=True)
    
    # 重新调整形状
    h_exp = h_exp.reshape((allMNCnt, 1))
    h_res = h_res.reshape((allMNCnt, 1))

    h_exp = h_exp.astype(np.float32)
    h_res = h_res.astype(np.float32)
    # 进行误差比较
    compare_res = np.isclose(h_exp, h_res, atol=atol, rtol=rtol, equal_nan=False)

    false_count = (~compare_res).sum().item()

    with open("./data/output.txt", "a") as f:
        print("===========================================================", file=f)
        print(f"errorCnt: {false_count}", file=f)
        print(f"Total Count: {allMNCnt}", file=f)
        print(f"Accuracy: {((allMNCnt - false_count) / allMNCnt) * 100:.2f}%", file=f)
        print("===========================================================", file=f)

if __name__ == "__main__":
    groupCnt = 8
    
    if len(sys.argv) > 1:
        groupCnt = int(sys.argv[1])
    
    M_array = np.fromfile("./data/input/M_array.bin", dtype=np.int32)
    N_array = np.fromfile("./data/input/N_array.bin", dtype=np.int32)
    K_array = np.fromfile("./data/input/K_array.bin", dtype=np.int32)
    
    allMNCnt = 0
    h_exp = np.fromfile("./data/output/exp_res.bin", dtype=NP_DATA_TYPE)
    h_res = np.fromfile("./data/output/our_res.bin", dtype=NP_DATA_TYPE)
    
    print("--- Test Begin ---\n")
    
    for i in range(groupCnt):
        M = M_array[i]
        N = N_array[i]
        K = K_array[i]
        allMNCnt += M * N  # 计算所有组的总数量
        printDimension(M, N, K, i)
    
    compareOutputData(h_exp, h_res, allMNCnt)  # 将 h_exp 和 h_res 传递进去
    print("--- Test End ---\n")
