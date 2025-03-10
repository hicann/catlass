import pandas as pd
import subprocess
import sys
import os

def run(times, device, mode):
    #执行gen_params.py
    # command = "python3 ./script/gen_params.py {} {}".format(times, mode)
    # result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # command = "../../../scripts/build.sh 01_fp16_rm_epi_gemm"
    # result = subprocess.run(command, shell=True, capture_output=True, text=True)

    data = pd.read_csv('./params/fp32GemvCM_Test.csv')  # 

    operator_path = os.getcwd()

    results = pd.DataFrame(columns=["M", "N", "time_us", "Tflops", "utilization_ratio"])
    prof_data_path = "./batch_prof_data.csv"
    for index, row in data.iterrows():
        if index > 100:
            break
        col1 = row.iloc[0]
        col2 = row.iloc[1]

        command = "./run_profiling.sh {} {} {} {}".format(col1, col2, device, mode) # M N device mode
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        last_line1 = result.stdout.strip().splitlines()[-1]
        parts = last_line1.split()  
        M = parts[1]          
        N = parts[3]            
        time_us = parts[5]  
        Tflops = parts[7]
        utilization_ratio = parts[9]

        print(format(last_line1))
        frame =  pd.DataFrame([[M, N, time_us, Tflops, utilization_ratio]], 
                              columns=["M", "N",  "time_us", "Tflops", "utilization_ratio"])
        
        if index == 0:
            frame.to_csv(prof_data_path, mode='w', header=True, index=False)
        else:
            frame.to_csv(prof_data_path, mode='a', header=not os.path.exists(prof_data_path), index=False)
        
    # command = "rm -rf ./prof/*"
    # subprocess.run(command, shell=True, capture_output=True, text=True)

if __name__ == "__main__":
    times = int(sys.argv[1])
    device = int(sys.argv[2])
    mode = int(sys.argv[3])
    run(times, device, mode)