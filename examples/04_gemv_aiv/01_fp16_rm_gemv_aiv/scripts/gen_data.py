import numpy as np
import argparse
import sys
DATA_TYPE = np.float32
NP_DATA_TYPE = np.float16

# 定义生成数据函数
def calc_expect_func(M, N,alpha,beta):
    matrix_gm = np.random.uniform(-0.125, 0.125, (M, N)).astype(NP_DATA_TYPE)
    vector_gm = np.random.uniform(-0.125, 0.125, (N, 1)).astype(NP_DATA_TYPE)
    vector_y_gm = np.random.uniform(-0.125, 0.125, (M, 1)).astype(NP_DATA_TYPE)
    # matrix_gm = np.random.randint(-10, 11, (M, N)).astype(NP_DATA_TYPE)
    # vector_gm = np.random.randint(-10, 11, (N, 1)).astype(NP_DATA_TYPE)
    # vector_y_gm = np.random.randint(-10, 11, (M, 1)).astype(NP_DATA_TYPE)
    matrix_gm.tofile('./data/input/matrix_gm.bin')
    vector_gm.tofile('./data/input/vector_gm.bin')
    vector_y_gm.tofile('./data/input/vector_y_gm.bin')
    golden = (alpha.astype(np.float32) * np.matmul(matrix_gm.astype(np.float32),vector_gm.astype(np.float32) ).astype(NP_DATA_TYPE) 
              + beta.astype(np.float32) * vector_y_gm.astype(np.float32)).astype(NP_DATA_TYPE)
    
    golden.tofile('./data/output/exp_res.bin')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('M', action='store', type=int)
    parser.add_argument('N', action='store', type=int)
    parser.add_argument('alpha', action='store', type=np.float32)
    parser.add_argument('beta', action='store', type=np.float32)

    args = parser.parse_args()

    M = args.M
    N = args.N
    alpha = args.alpha
    beta = args.beta

    calc_expect_func(M, N,alpha,beta)