import numpy as np
import argparse
import sys
DATA_TYPE = np.int32
NP_DATA_TYPE = np.int8

# 定义生成数据函数
def calc_expect_func(M, N, K):
    # 设置默认矩阵形状
    matSizeA = (K, M)
    matSizeB = (N, K) 

    A = np.random.uniform(-5,5,size=matSizeA).astype(NP_DATA_TYPE)
    B = np.random.uniform(-5,5,size=matSizeB).astype(NP_DATA_TYPE)

    A.tofile("./data/input/A.bin")
    B.tofile("./data/input/B.bin")
    # print(A)
    # print(B)
    expect_res = np.array(np.matmul(B.astype(DATA_TYPE),A.astype(DATA_TYPE))).astype(DATA_TYPE)
    expect_res.tofile("./data/output/exp_res.bin")


if __name__ == "__main__":
    # 初始化参数
    M = 32
    N = 32
    K = 32

    # 解析参数
    if len(sys.argv) > 1:
        M = int(sys.argv[1])
    if len(sys.argv) > 2:
        N = int(sys.argv[2])
    if len(sys.argv) > 3:
        K = int(sys.argv[3])
    calc_expect_func(M,N,K)