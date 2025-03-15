import pandas as pd
import numpy as np
import sys
import ast
import argparse

if __name__ == '__main__':
    # sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('groupCnt', action='store', type=int)
    parser.add_argument('mList', action='store', type=str)
    parser.add_argument('nList', action='store', type=str)
    parser.add_argument('kList', action='store', type=str)
    args = parser.parse_args()

    groupCnt = args.groupCnt
    mList = list(map(np.uint32, args.mList.split(',')))
    M_array = np.empty(groupCnt, dtype=np.uint32)
    nList = list(map(np.uint32, args.nList.split(',')))
    N_array = np.empty(groupCnt, dtype=np.uint32)
    kList = list(map(np.uint32, args.kList.split(',')))
    K_array = np.empty(groupCnt, dtype=np.uint32)

    for i in range(groupCnt):
        M_array[i] = mList[i]
        N_array[i] = nList[i]
        K_array[i] = kList[i]
    # print(M_array)
    # 保存数组到文件
    M_array.tofile("./data/input/M_array.bin")
    N_array.tofile("./data/input/N_array.bin")
    K_array.tofile("./data/input/K_array.bin")
