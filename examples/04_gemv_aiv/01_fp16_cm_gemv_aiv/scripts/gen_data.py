import numpy as np
import argparse
import sys
DATA_TYPE = float
NP_DATA_TYPE = np.half

# 定义生成数据函数
def calc_expect_func(M, N):
    # # 设置默认矩阵形状
    # matSizeA = (M, N)
    # matSizeX = (N, 1) # 首先全是行优先

    # A = np.random.uniform(-1,1,size=matSizeA).astype(NP_DATA_TYPE)
    # X = np.random.uniform(-1,1,size=matSizeX).astype(NP_DATA_TYPE)

    # A.tofile("./data/input/matrix_gm.bin")
    # X.tofile("./data/input/vector_gm.bin")
    # expect_res = np.array(np.matmul(A.astype(DATA_TYPE),X.astype(DATA_TYPE))).astype(NP_DATA_TYPE)
    # expect_res.tofile("./data/output/exp_res.bin")
    matrix_gm = np.random.uniform(-1, 1, (M, N)).astype(NP_DATA_TYPE)
    vector_gm = np.random.uniform(-1, 1, (N, 1)).astype(NP_DATA_TYPE)
    vector_y_gm = np.random.uniform(-1, 1, (M, 1)).astype(NP_DATA_TYPE)
    # matrix_gm = np.random.randint(-10, 11, (M, N)).astype(NP_DATA_TYPE)
    # vector_gm = np.random.randint(-10, 11, (N, 1)).astype(NP_DATA_TYPE)
    # vector_y_gm = np.random.randint(-10, 11, (M, 1)).astype(NP_DATA_TYPE)
    golden = np.matmul(matrix_gm.astype(DATA_TYPE), vector_gm.astype(DATA_TYPE)).astype(NP_DATA_TYPE)
    matrix_gm = matrix_gm.T
    vector_gm = vector_gm.T
    vector_y_gm = vector_y_gm.T
    golden = golden.T
    matrix_gm.tofile('./data/input/matrix_gm.bin')
    vector_gm.tofile('./data/input/vector_gm.bin')
    vector_y_gm.tofile('./data/input/vector_y_gm.bin')
    golden.tofile('./data/output/exp_res.bin')


if __name__ == "__main__":
    # 初始化参数
    M = 32
    N = 32

    # 解析参数
    if len(sys.argv) > 1:
        M = int(sys.argv[1])
    if len(sys.argv) > 2:
        N = int(sys.argv[2])

    calc_expect_func(M,N)