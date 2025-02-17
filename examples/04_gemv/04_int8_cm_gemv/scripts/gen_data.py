import numpy as np
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "16"

def gen_golden_data(M, N):
    gm_dtype = np.int8

    matrix_gm = np.random.uniform(-5, 5, size=(N,M)).astype(gm_dtype)
    vector_gm = np.random.uniform(-5, 5, size=(1,N)).astype(gm_dtype)
    golden = np.matmul(vector_gm.astype(np.int32), matrix_gm.astype(np.int32)).astype(np.int32)



    # 保存文件
    matrix_gm.tofile('./data/input/A.bin')
    vector_gm.tofile('./data/input/X.bin')
    golden.tofile('./data/output/exp_res.bin')

if __name__ == '__main__':

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('M', action='store', type=int)
    parser.add_argument('N', action='store', type=int)


    args = parser.parse_args()
  
    # 更新 M, N, transA 的值
    M = args.M
    N = args.N


    # 生成数据
    gen_golden_data(M, N)