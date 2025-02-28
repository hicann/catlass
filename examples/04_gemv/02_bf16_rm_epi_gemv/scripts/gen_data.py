import numpy as np
import argparse
import sys
from ml_dtypes import bfloat16
DATA_TYPE = np.float32
NP_DATA_TYPE = bfloat16

# 定义生成数据函数
def calc_expect_func(M, N):
    # 设置默认矩阵形状
    Sizex = (1, N)
    SizeA = (M, N) # 首先全是行优先
    Sizey = (1, M)
    alpha = np.random.uniform(-1,1, 1).astype(NP_DATA_TYPE)
    beta  = np.random.uniform(-1,1, 1).astype(NP_DATA_TYPE)
    x = np.random.uniform(-1,1,size=Sizex).astype(NP_DATA_TYPE)
    A = np.random.uniform(-1,1,size=SizeA).astype(NP_DATA_TYPE)
    y = np.random.uniform(-1,1,size=Sizey).astype(NP_DATA_TYPE)
    alpha.tofile("./data/input/alpha.bin")
    beta.tofile("./data/input/beta.bin")
    x.tofile("./data/input/X.bin")
    A.tofile("./data/input/A.bin")
    y.tofile("./data/input/Y.bin")
    # print(A)
    # print(B)
    # print(C)
    tmp = np.matmul(A.astype(DATA_TYPE),x.astype(DATA_TYPE).T).astype(NP_DATA_TYPE)
    tmp = tmp.T
    # D = alpha * A * x + beta * y
    expect_res = np.array(tmp * alpha.astype(DATA_TYPE) + beta.astype(DATA_TYPE) * y.astype(DATA_TYPE)).astype(NP_DATA_TYPE)
    # expect_res = np.array(beta.astype(NP_DATA_TYPE) * C.astype(NP_DATA_TYPE)).astype(NP_DATA_TYPE)
    # expect_res = C
    expect_res.tofile("./data/output/exp_res.bin")


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