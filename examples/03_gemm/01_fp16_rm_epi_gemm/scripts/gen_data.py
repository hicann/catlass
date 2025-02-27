import numpy as np
import argparse
import sys
DATA_TYPE = np.float32
NP_DATA_TYPE = np.half

# 定义生成数据函数
def calc_expect_func(M, N, K):
    # 设置默认矩阵形状
    matSizeA = (M, K)
    matSizeB = (K, N) # 首先全是行优先
    matSizeC = (M, N)
    # alpha = np.random.uniform(-0.125,0.125, 1).astype(DATA_TYPE)
    # beta  = np.random.uniform(-0.125,0.125, 1).astype(DATA_TYPE)
    # A = np.random.uniform(-0.125,0.125,size=matSizeA).astype(NP_DATA_TYPE)
    # B = np.random.uniform(-0.125,0.125,size=matSizeB).astype(NP_DATA_TYPE)
    # C = np.random.uniform(-0.125,0.125,size=matSizeC).astype(NP_DATA_TYPE)
    alpha = np.random.uniform(-0.5,0.5, 1).astype(DATA_TYPE)
    beta  = np.random.uniform(-0.5,0.5, 1).astype(DATA_TYPE)
    A = np.random.uniform(-0.5,0.5,size=matSizeA).astype(NP_DATA_TYPE)
    B = np.random.uniform(-0.5,0.5,size=matSizeB).astype(NP_DATA_TYPE)
    C = np.random.uniform(-0.5,0.5,size=matSizeC).astype(NP_DATA_TYPE)
    # alpha = np.random.uniform(-1,1, 1).astype(DATA_TYPE)
    # beta  = np.random.uniform(-1,1, 1).astype(DATA_TYPE)
    # A = np.random.uniform(-1,1,size=matSizeA).astype(NP_DATA_TYPE)
    # B = np.random.uniform(-1,1,size=matSizeB).astype(NP_DATA_TYPE)
    # C = np.random.uniform(-1,1,size=matSizeC).astype(NP_DATA_TYPE)
    with open("./data/input/alpha.bin", "w") as alpha_file:
        alpha.tofile(alpha_file)
    
    with open("./data/input/beta.bin", "w") as beta_file:
        beta.tofile(beta_file)

    with open("./data/input/A.bin", "w") as A_file:
        A.tofile(A_file)

    with open("./data/input/B.bin", "w") as B_file:
        B.tofile(B_file)

    with open("./data/input/C.bin", "w") as C_file:
        C.tofile(C_file)

    # print(A)
    # print(B)
    # print(C)
    tmp = np.matmul(A.astype(DATA_TYPE),B.astype(DATA_TYPE)).astype(DATA_TYPE)
    # D = alpha * A * B + beta * C
    expect_res = np.array(tmp * alpha.astype(DATA_TYPE) + beta.astype(DATA_TYPE) * C.astype(DATA_TYPE)).astype(NP_DATA_TYPE)
    # expect_res = np.array(beta.astype(NP_DATA_TYPE) * C.astype(NP_DATA_TYPE)).astype(NP_DATA_TYPE)
    # expect_res = C
    with open("./data/output/exp_res.bin", "w") as exp_res_file:
        expect_res.tofile(exp_res_file)


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