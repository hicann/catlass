import csv
import sys
import os
import ast
import numpy as np

size = len(sys.argv) - 1
problemCount = int(sys.argv[size - 3])
mStr = str(sys.argv[size - 2])
nStr = str(sys.argv[size - 1])
kStr = str(sys.argv[size])

mArray = list(ast.literal_eval(mStr))
mList = np.array(mArray, dtype=np.uint32)
nArray = list(ast.literal_eval(nStr))
nList = np.array(nArray, dtype=np.uint32)
kArray = list(ast.literal_eval(kStr))
kList = np.array(kArray, dtype=np.uint32)

time_us_total = 0

for i in range(5): 
    with open(sys.argv[i + 1], newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        time_us_list = [float(row['Task Duration(us)']) for row in reader]
        
        time_us = sum(time_us_list[0:]) / len(time_us_list[0:])

        time_us_total += time_us

aic_mac_ratio_total = 0

for i in range(5): 
    with open(sys.argv[i + 6], newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        aic_mac_ratio_list = [float(row['aic_cube_ratio']) for row in reader if row['sub_block_id'] == "cube0"]
        
        aic_mac_ratio = sum(aic_mac_ratio_list[0:]) / len(aic_mac_ratio_list[0:])

        aic_mac_ratio_total += aic_mac_ratio


# Mflops = 2.0 * M * N * K * 1e-6

Mflops = 0.0
for i in range(problemCount):
    Mflops += 2.0 * mList[i] * nList[i] * kList[i] * 1e-6

print("problemCount:", problemCount, "mList: ", mStr, "nList: ", nStr, "kList: ", kStr, "time_us: ", time_us_total / 5, "Tflops: ", Mflops / time_us_total * 5, "aic_mac_ratio: ", aic_mac_ratio_total * 100 / 5, "%") 