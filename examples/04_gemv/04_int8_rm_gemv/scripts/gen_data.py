import numpy as np
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "16"

def gen_golden_data(M, N):
    A_dtype = np.int8
    B_dtype = np.int8
    Compute_dtype = np.int32  # 中间计算使用 int32 提升精度
    C_dtype = np.int32

    # 固定矩阵形状为 (M, N)，存储顺序由 transA 决定
    matrix_size = (M, N)
    vectorIn_size = (N, 1)

    # 行优先存储 (C 顺序)
    x1_gm = np.random.uniform(-5, 5, size=matrix_size).astype(A_dtype)
    # x1_gm = np.ones(matrix_size, dtype=A_dtype)
    x1_c = np.ascontiguousarray(x1_gm)  # 确保内存连续且行优先


    # 生成向量（始终列优先）
    x2_gm = np.random.uniform(-5, 5, size=vectorIn_size).astype(B_dtype)
    # x2_gm = np.ones(vectorIn_size, dtype=B_dtype)  # 假设 x2 全为 1 向量
    x2_c = np.ascontiguousarray(x2_gm)

    print("x2_c's shape: ", x2_c.shape)     
    print("x2_c: " + str(x2_c))  # 打印示例数据

    # 矩阵乘法（显式提升计算精度）
    golden = np.dot(
        x1_c.astype(Compute_dtype),  # 转换为 int8 计算
        x2_c.astype(Compute_dtype)
    ).astype(C_dtype)  

    # 转置为 (1, M) 形状
    golden = golden.T  # shape: (1, M)

    print("golden's shape") 
    print(golden.shape)
    print("golden: " + str(golden))

    # 保存文件
    x1_gm.tofile('./data/input/A.bin')
    x2_gm.tofile('./data/input/X.bin')
    golden.tofile('./data/output/exp_res.bin')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('M', type=int)
    parser.add_argument('N', type=int)
    args = parser.parse_args()
    gen_golden_data(args.M, args.N)