import numpy as np
import sys
import random

def gen_data_MNK_array(groupCnt):
    # 预分配数组
    M_array = np.empty(groupCnt, dtype=np.uint32)
    N_array = np.empty(groupCnt, dtype=np.uint32)
    K_array = np.empty(groupCnt, dtype=np.uint32)
    
    # 生成随机维度并存入数组
    for i in range(groupCnt):
        M_array[i] = random.randint(1, 10000)
        N_array[i] = random.randint(1, 10000)
        K_array[i] = random.randint(1, 10000)
        # M_array[i] = 32
        # N_array[i] = 32
        # K_array[i] = 32
    # print(M_array)
    # print(N_array)
    # print(K_array)
    # 保存数组到文件
    M_array.tofile("./data/input/M_array.bin")
    N_array.tofile("./data/input/N_array.bin")
    K_array.tofile("./data/input/K_array.bin")

if __name__ == "__main__":
    groupCnt = 8
    
    if len(sys.argv) > 1:
        groupCnt = int(sys.argv[1])
    
    gen_data_MNK_array(groupCnt)
