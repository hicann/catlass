import numpy as np
import sys

NP_DATA_TYPE = np.float32  # 选择 FP16 数据类型

def get_thresholds(M, N, K, dtype=np.half):
    """根据 calc_times (M * N * K) 计算 atol 和 rtol"""
    calc_times = M * N * K
    
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
    
    else:
        raise ValueError("Unsupported data type!")

    return atol, rtol

def compareOutputData(M, N, K):
    print("--- Test Begin ---\n")

    atol, rtol = get_thresholds(M, N, K, dtype=NP_DATA_TYPE)  # 计算 atol 和 rtol

    with open("./data/output.txt", "a") as f:
        print("---- FP32 -- ColumnMajor -- EpilogueGemm ---- \n", file=f)
        print(f"  M : {M} , N : {N} , K : {K} \n", file=f)
        np.set_printoptions(threshold=np.inf, precision=3, suppress=True)

        # 读取二进制文件
        h_exp = np.fromfile("./data/output/exp_res.bin", dtype=NP_DATA_TYPE)
        h_res = np.fromfile("./data/output/our_res.bin", dtype=NP_DATA_TYPE)

        # 重新调整形状
        h_exp = h_exp.reshape((M * N, 1))
        h_res = h_res.reshape((M * N, 1))

        # 进行误差比较
        compare_res = np.isclose(h_exp, h_res, atol=atol, rtol=rtol, equal_nan=False)

        allCnt = M * N
        false_count = (~compare_res).sum().item()

        print("===========================================================", file=f)
        print(f"errorCnt: {false_count}", file=f)
        print(f"Total Count: {allCnt}", file=f)
        print(f"Accuracy: {((allCnt - false_count) / allCnt) * 100:.2f}%", file=f)
        print("===========================================================", file=f)

    print("--- Test End ---\n")

if __name__ == "__main__":
    M, N, K = 32, 32, 32  # 默认值
    if len(sys.argv) > 1:
        M = int(sys.argv[1])
    if len(sys.argv) > 2:
        N = int(sys.argv[2])
    if len(sys.argv) > 3:
        K = int(sys.argv[3])
    
    compareOutputData(M, N, K)
