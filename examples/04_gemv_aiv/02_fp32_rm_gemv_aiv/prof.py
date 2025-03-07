import csv
import sys
import os
import ast
import numpy as np
# 这个文件存在问题
size = len(sys.argv) - 1
M = int(sys.argv[size - 1])
N = int(sys.argv[size])
print(sys.argv)
time_us_total = 0
 
with open(sys.argv[1], newline='') as csvfile:
    reader = csv.DictReader(csvfile, skipinitialspace=True)
    time_us_list = [float(row['Task Duration(us)']) for row in reader]
    
    time_us = sum(time_us_list[0:]) / len(time_us_list[0:])

    time_us_total += time_us

aiv_mac_ratio_total = 0

with open(sys.argv[2], newline='') as csvfile: # 一共两个文件内容
    reader = csv.DictReader(csvfile, skipinitialspace=True)

    aiv_mac_ratio_list = [float(row['aiv_vec_ratio']) for row in reader if row['sub_block_id'] == "vector0"]
    
    aiv_mac_ratio = sum(aiv_mac_ratio_list[0:]) / len(aiv_mac_ratio_list[0:])

    aiv_mac_ratio_total += aiv_mac_ratio


Mflops = (M * N ) * 1e-6

print("M:", M, "N: ", N,  "time_us: ", time_us_total, "Tflops: ", Mflops / time_us_total, "aic_mac_ratio: ", aiv_mac_ratio_total * 100, "%") 