import numpy as np
import argparse
import sys
DATA_TYPE = float
NP_DATA_TYPE = np.float32

# 定义生成数据函数
def calc_expect_func(M, N, K):
    # 设置默认矩阵形状
    matSizeA = (K, M)
    matSizeB = (N, K) # 首先全是行优先
    matSizeC = (N, M)
    alpha = np.random.uniform(-1,1, 1).astype(NP_DATA_TYPE)
    beta  = np.random.uniform(-1,1, 1).astype(NP_DATA_TYPE)
    A = np.random.uniform(-1,1,size=matSizeA).astype(NP_DATA_TYPE)
    B = np.random.uniform(-1,1,size=matSizeB).astype(NP_DATA_TYPE)
    C = np.random.uniform(-1,1,size=matSizeC).astype(NP_DATA_TYPE)
    alpha.tofile("./data/input/alpha.bin")
    beta.tofile("./data/input/beta.bin")
    A.tofile("./data/input/A.bin")
    B.tofile("./data/input/B.bin")
    C.tofile("./data/input/C.bin")
    # print(A)
    # print(B)
    # print(C)
    tmp = np.matmul(B.astype(DATA_TYPE),A.astype(DATA_TYPE)).astype(NP_DATA_TYPE)
    # D = alpha * B * A + beta * C
    expect_res = np.array(tmp * alpha.astype(DATA_TYPE) + beta.astype(DATA_TYPE) * C.astype(DATA_TYPE)).astype(NP_DATA_TYPE)
    # expect_res = np.array(beta.astype(NP_DATA_TYPE) * C.astype(NP_DATA_TYPE)).astype(NP_DATA_TYPE)
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