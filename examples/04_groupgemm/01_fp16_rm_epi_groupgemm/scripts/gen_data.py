import numpy as np
import sys
import argparse

DATA_TYPE = np.float32
NP_DATA_TYPE = np.float16

def gen_data(groupCnt):
    # 读取 M, N, K 数组
    M_array = np.fromfile("./data/input/M_array.bin", dtype=np.uint32)
    N_array = np.fromfile("./data/input/N_array.bin", dtype=np.uint32)
    K_array = np.fromfile("./data/input/K_array.bin", dtype=np.uint32)
    # print(M_array)
    # print(N_array)
    # print(K_array)
    for i in range(groupCnt):
        M = M_array[i]
        N = N_array[i]
        K = K_array[i]
        
        # 创建 A, B, C 数据
        # A = np.random.uniform(-0.25, 0.25, size=(M, K)).astype(NP_DATA_TYPE)
        # B = np.random.uniform(-0.25, 0.25, size=(K, N)).astype(NP_DATA_TYPE)
        # C = np.random.uniform(-0.25, 0.25, size=(M, N)).astype(NP_DATA_TYPE)
        
        A = np.random.uniform(-0.125, 0.125, size=(M, K)).astype(NP_DATA_TYPE)
        B = np.random.uniform(-0.125, 0.125, size=(K, N)).astype(NP_DATA_TYPE)
        C = np.random.uniform(-0.125, 0.125, size=(M, N)).astype(NP_DATA_TYPE)

        # 写入文件
        with open("./data/input/A.bin", "a") as a:
            A.tofile(a)
        with open("./data/input/B.bin", "a") as b:
            B.tofile(b)
        with open("./data/input/C.bin", "a") as c:
            C.tofile(c)
        
        # alpha 和 beta 是标量，可以直接生成并写入文件
        alpha = np.random.uniform(1, 1, 1).astype(DATA_TYPE)
        beta = np.random.uniform(1, 1, 1).astype(DATA_TYPE)
        
        with open("./data/input/alpha.bin", "a") as alpha_a:
            alpha.tofile(alpha_a)
        with open("./data/input/beta.bin", "a") as beta_b:
            beta.tofile(beta_b)
        
        # 计算期望结果
        tmp = np.matmul(A.astype(DATA_TYPE), B.astype(DATA_TYPE)).astype(DATA_TYPE)
        expect_res = np.array(tmp * alpha + beta * C.astype(DATA_TYPE)).astype(NP_DATA_TYPE)
        
        # 写入期望结果
        with open("./data/output/exp_res.bin", "a") as exp_res:
            expect_res.tofile(exp_res)

if __name__ == "__main__":
    groupCnt = 8
    
    if len(sys.argv) > 1:
        groupCnt = int(sys.argv[1])
    
    gen_data(groupCnt)
