import numpy as np
import sys
from ml_dtypes import bfloat16

NP_DATA_TYPE = bfloat16  # 选择 FP16 数据类型

def get_thresholds(M, N, dtype=np.half):
    """根据 calc_times (M * N) 计算 atol 和 rtol"""
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
    
    else:
        raise ValueError("Unsupported data type!")

    return atol, rtol

def compareOutputData(M, N):
    print("--- Test Begin ---\n")

    atol, rtol = get_thresholds(M, N, dtype=NP_DATA_TYPE)  # 计算 atol 和 rtol

    with open("./data/output.txt", "a") as f:
        print("---- BF16 -- RowMajor -- EpilogueGemv ---- \n", file=f)
        print(f"  M : {M} , N : {N} \n", file=f)
        np.set_printoptions(threshold=np.inf, precision=3, suppress=True)

        # 读取二进制文件
        h_exp = np.fromfile("./data/output/exp_res.bin", dtype=NP_DATA_TYPE)
        h_res = np.fromfile("./data/output/our_res.bin", dtype=NP_DATA_TYPE)

        # 重新调整形状
        h_exp = h_exp.reshape((M, 1))
        h_res = h_res.reshape((M, 1))

        # 全部转成float32比较，避免bfloat16计算的精度问题
        h_exp = h_exp.astype(np.float32)
        h_res = h_res.astype(np.float32)

        # 进行误差比较
        compare_res = np.isclose(h_exp, h_res, atol=atol, rtol=rtol, equal_nan=False)

        allCnt = M
        false_count = (~compare_res).sum().item()

        print("===========================================================", file=f)
        print(f"errorCnt: {false_count}", file=f)
        print(f"Total Count: {allCnt}", file=f)
        print(f"Accuracy: {((allCnt - false_count) / allCnt) * 100:.2f}%", file=f)
        print("===========================================================", file=f)

        # 找出错误的行数
        error_rows = np.where(~compare_res)[0]
        print("错误的行数:", error_rows, file=f)

        # 输出错误位置对应的两个数据
        if len(error_rows) > 0:
            print("错误位置对应的预期数据和实际数据：", file=f)
            for row in error_rows:
                expected_value = h_exp[row][0]
                actual_value = h_res[row][0]
                print(f"行号: {row}, 预期数据: {expected_value}, 实际数据: {actual_value}", file=f)

    print("--- Test End ---\n")

if __name__ == "__main__":
    M, N = 32, 32  # 默认值
    if len(sys.argv) > 1:
        M = int(sys.argv[1])
    if len(sys.argv) > 2:
        N = int(sys.argv[2])

    compareOutputData(M, N)
