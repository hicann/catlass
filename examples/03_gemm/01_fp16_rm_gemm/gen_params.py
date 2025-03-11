# 生成单矩阵
import sys
import numpy as np
import csv
import os

def gen_test_data(times, mode):
    if times < 0:
        print("times must be greater than or equal to 0!")
        sys.exit(1)
    # 保存M N K的路径
    test_dim_csv_filepath = f"./params/MNK_data.csv"

    # 检查并创建 result 目录
    os.makedirs(os.path.dirname(test_dim_csv_filepath), exist_ok=True)
    #生成随机矩阵
    if mode == 0:
        low_M = 1
        high_M = 10000
        low_N = 1
        high_N = 10000
        low_K = 1
        high_K = 10000

        with open(test_dim_csv_filepath, "w") as f_output:
            f_output.write("M,N,K\n")  # 单个矩阵就只需要M,N,K

        # 生成数据并写入 CSV 文件
        for i in range(times):
            M = np.random.randint(low_M, high_M, dtype=np.uint32)
            N = np.random.randint(low_N, high_N, dtype=np.uint32)
            K = np.random.randint(low_K, high_K, dtype=np.uint32)

            with open(test_dim_csv_filepath, "a") as f_output:
                writer = csv.writer(f_output)
                writer.writerow([M, N, K])

    # 生成规整矩阵
    if mode == 1:
        low_M = 256
        high_M = 9984
        low_N = 256
        high_N = 9984
        low_K = 256
        high_K = 9984

        with open(test_dim_csv_filepath, "w") as f_output:
            f_output.write("M,N,K\n")  # 单个矩阵就只需要M,N,K

        # 生成数据并写入 CSV 文件
        for i in range(times):
            M = np.random.randint(low_M, high_M, dtype=np.uint32)
            N = np.random.randint(low_N, high_N, dtype=np.uint32)
            K = np.random.randint(low_K, high_K, dtype=np.uint32)

            with open(test_dim_csv_filepath, "a") as f_output:
                writer = csv.writer(f_output)
                writer.writerow([M, N, K])
    pass

if __name__ == "__main__":
    times = int(sys.argv[1])
    mode = int(sys.argv[2])
    gen_test_data(times, mode)
    pass