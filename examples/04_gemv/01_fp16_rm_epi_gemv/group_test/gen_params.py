#做批量测试时  生成指定次数的随机M N K   保存在params文件夹下  格式是csv
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
            f_output.write("problemCount,mList,nList,kList\n")

        # 生成数据并写入 CSV 文件
        for i in range(times):
            problemCount = np.random.randint(2, 10, dtype=np.int32)
            # M = np.random.randint(low_M, high_M, dtype=np.int32)
            mList = np.random.randint(low_M, high_M, size=problemCount, dtype=np.int32)
            mStr = ','.join(map(str, mList))
            # N = np.random.randint(low_N, high_N, dtype=np.int32)
            nList = np.random.randint(low_N, high_N, size=problemCount, dtype=np.int32)
            nStr = ','.join(map(str, nList))
            # K = np.random.randint(low_K, high_K, dtype=np.int32)
            kList = np.random.randint(low_K, high_K, size=problemCount, dtype=np.int32)
            kStr = ','.join(map(str, kList))

            with open(test_dim_csv_filepath, "a") as f_output:
                writer = csv.writer(f_output)
                writer.writerow([problemCount, mStr, nStr, kStr])
    # 生成规整矩阵
    if mode == 1:
        low_M = 256
        high_M = 9984
        low_N = 256
        high_N = 9984
        low_K = 256
        high_K = 9984

        with open(test_dim_csv_filepath, "w") as f_output:
            f_output.write("problemCount,mList,nList,kList\n")

        # 生成数据并写入 CSV 文件
        for i in range(times):
            problemCount = np.random.randint(2, 10, dtype=np.int32)
            # M = np.random.randint(low_M, high_M, dtype=np.int32)
            mList = np.random.randint(low_M, high_M, size=problemCount, dtype=np.int32)
            # N = np.random.randint(low_N, high_N, dtype=np.int32)
            nList = np.random.randint(low_N, high_N, size=problemCount, dtype=np.int32)
            # K = np.random.randint(low_K, high_K, dtype=np.int32)
            kList = np.random.randint(low_K, high_K, size=problemCount, dtype=np.int32)
            # M = int(M / 256) * 256
            for i in range(len(mList)):
                mList[i] = int(mList[i] / 256) * 256
            mStr = ','.join(map(str, mList))
            # N = int(N / 256) * 256
            for i in range(len(nList)):
                nList[i] = int(nList[i] / 256) * 256
            nStr = ','.join(map(str, nList))
            # K = int(K / 256) * 256
            for i in range(len(kList)):
                kList[i] = int(kList[i] / 256) * 256
            kStr = ','.join(map(str, kList))

            with open(test_dim_csv_filepath, "a") as f_output:
                writer = csv.writer(f_output)
                writer.writerow([problemCount, mStr, nStr, kStr])

if __name__ == "__main__":
    times = int(sys.argv[1])
    mode = int(sys.argv[2])
    gen_test_data(times, mode)
