import numpy as np
import argparse
import sys

NP_DATA_TYPE = np.float16
def calculate_thresholds(M):
    calc_times = M * 1

    # 这里假设传入的 NP_DATA_TYPE 用 Python 数据类型来表示
    # 可以通过传入的数据类型进行判断
    def check_type(data_type):
        if data_type == np.float32:
            if calc_times < 2048:
                thres = 1.0 / (1 << 11)
            elif calc_times < 16384:
                thres = 1.0 / (1 << 10)
            else:
                thres = 1.0 / (1 << 9)
            eb_thres = 1.0 / (1 << 14)
        elif data_type == np.float16:
            if calc_times < 2048:
                thres = 1.0 / (1 << 8)
            else:
                thres = 1.0 / (1 << 7)
            eb_thres = 1.0 / (1 << 10)
        else:
            raise ValueError("不支持的数据类型，仅支持 np.float32 和 np.float16")
        return thres, eb_thres

    # 示例：假设这里使用 np.float32 类型，你可以根据实际情况修改
    thres, eb_thres = check_type(NP_DATA_TYPE)
    return thres, eb_thres
def compareOutputData(M, N):
    thres, eb_thres = calculate_thresholds(M)
    with open("./data/output.txt", "a") as f:
        print("---- FP16 -- RowMajor -- Gemv_aiv ---- \n", file=f)
        print("  M : {} , N : {} \n".format(M, N), file=f)
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        h_exp = np.fromfile("./data/output/exp_res.bin", dtype=NP_DATA_TYPE)
        h_res = np.fromfile("./data/output/our_res.bin", dtype=NP_DATA_TYPE)
        # 正确重塑数组形状
        h_exp = h_exp.reshape((M, 1))  # 都是一列数据
        h_res = h_res.reshape((M, 1))
        # atol 是绝对容忍度
        # rtol 是相对容忍度
        compare_res = np.isclose(h_exp, h_res, atol=thres, rtol=thres, equal_nan=False)
        allCnt = M * 1
        print("===========================================================", file=f)
        false_count = (~compare_res).sum().item()
        print("errorCnt:", false_count, file=f)
        print("Total Count:", allCnt, file=f)
        print("Accuracy: {:.2f}%".format(((allCnt - false_count) / allCnt) * 100), file=f)
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


if __name__ == "__main__":
    M = 32
    N = 32
    if len(sys.argv) > 1:
        M = int(sys.argv[1])
    if len(sys.argv) > 2:
        N = int(sys.argv[2])
    compareOutputData(M, N)
    