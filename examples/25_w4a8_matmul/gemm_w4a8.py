import os
import sys
import glob
import sqlite3
import torch
import numpy as np
import pandas as pd
from enum import Enum

CONST_16 = 16
RECORD_COUNT = 10
DATA_RANGE = (-1.0, 1.0)
WORKSPACE = os.getcwd()

os.environ["WORKSPACE"] = WORKSPACE
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"


class CubeFormat(Enum):
    ND = 0
    NZ = 1
    ZN = 2
    ZZ = 3
    NN = 4
    VECTOR = 5

    def __repr__(self) -> str:
        return self.__name__

class OpParam:
    def __init__(self) -> None:
        self.b = 0
        self.m = 0
        self.k = 0
        self.n = 0
        self.transA = False
        self.transB = False
        self.enBias = False
        self.enScale = False
        self.enResidual = False
        self.layoutA = CubeFormat.ND
        self.layoutB = CubeFormat.ND
        self.layoutC = CubeFormat.ND
    
    def __str__(self) -> str:
        return f"Shape: ({self.b}, {self.m}, {self.k}, {self.n}) \n" + \
               f"Transpose: A {self.transA}, B {self.transB} \n" + \
               f"(De)Quant: Bias {self.enBias}, Scale {self.enScale}, Residual {self.enResidual} \n" + \
               f"Layout: layoutA {self.layoutA}, layoutB {self.layoutB}, layoutC {self.layoutC}"
# 生成一个mXn的矩阵，其中的值介于low和high之间
def gen_rand(msize, nsize, low, high):
    return low + (high - low) * torch.rand((msize, nsize),dtype=torch.float32)

def build(workspace : str) -> None:
    os.system("cd ./scripts && bash build.sh".format(workspace))
# 生成一个row x col的二维数组，数值介于-8到7之间
def gen_data_int8(row, col):
    data = np.random.randint(-8, 8, size=(row, col), dtype=np.int8)
    return data
# 生成数据并做数据压缩
def gen_data_int4(row, col, trans):
    # 生成原始数据，范围严格在 [-8, 7]
    data_int8_origin = np.random.randint(-8, 8, size=(row, col), dtype=np.int8)
    if trans:
        data_int8_origin = data_int8_origin.T
        # 如果列数为奇数，则补零列
        data_int8 = data_int8_origin
        if row % 2 != 0:
            zero_row = np.zeros((col, 1), dtype=np.int8)
            data_int8 = np.hstack((data_int8_origin, zero_row))
        
        quantized = data_int8.reshape(-1, 2)
        print(quantized.shape)
        high_quantized = (quantized[:, 0] & 0x0F)
        low_quantized = (quantized[:, 1] & 0x0F) << 4
        data_int4 = low_quantized | high_quantized
        print("kkkkkk")

        data_int4_array = np.array(data_int4, dtype=np.int8)
        print(data_int4_array.shape)
        return data_int8_origin.T, data_int4_array

    else:
        data_int8 = data_int8_origin
        if col % 2 != 0:
            zero_column = np.zeros((row, 1), dtype=np.int8)
            data_int8 = np.hstack((data_int8_origin, zero_column))
        
        quantized = data_int8.reshape(-1, 2)
        print(quantized.shape)
        high_quantized = (quantized[:, 0] & 0x0F)
        low_quantized = (quantized[:, 1] & 0x0F) << 4
        data_int4 = low_quantized | high_quantized

        data_int4_array = np.array(data_int4, dtype=np.int8)
        print("kkkkkk")
        print(data_int4.shape)
        return data_int8_origin, data_int4_array

def gen_testcase(path: str, param: OpParam) -> None:
    bsize, msize, ksize, nsize = param.b, param.m, param.k, param.n
    transA, transB = param.transA, param.transB

    a_int8 = gen_data_int8(msize, ksize)
    print("int8 a done")

    b_int8, b_int4 = gen_data_int4(ksize, nsize, transB)
    b_int4.tofile(WORKSPACE +"/build/data/inputB.dat")
    print("int4 b done")

    print(a_int8.shape)
    print(b_int8.shape)
    
    # numpy int32矩阵乘法非常慢，此处用float32的矩阵乘法代替
    c_int32 = np.dot(a_int8.astype(np.float32), b_int8.astype(np.float32))
    c_int32 = np.float32(1.5) * c_int32

    print("matmul done")

    if transA:
        a_int8 = a_int8.T
    a_int8.tofile(WORKSPACE +"/build/data/inputA.dat")

    c_float = c_int32.astype(np.float32)
    c_half = c_int32.astype(np.float16)
    c_half.tofile(WORKSPACE +"/build/data/inputC.dat")
    c_float.tofile(WORKSPACE +"/build/data/expected.dat")

def compare(param: OpParam) -> None:
    ref = np.fromfile(WORKSPACE +"/build/data/inputC.dat", dtype=np.float16).astype(np.float32)
    ret = np.fromfile(WORKSPACE +"/build/data/outputC.dat", dtype=np.float16).astype(np.float32)
    expect = np.fromfile(WORKSPACE + "/build/data/expected.dat", dtype=np.float32).astype(np.float32)
    rdiff = np.abs(ret - ref) / np.abs(ref + 1e-6)
    precis = len(np.where(rdiff < 0.001)[0]) / len(ref)
    ret = torch.from_numpy(ret)
    ref = torch.from_numpy(ref)
    print(ret)
    print(ref)
    print("[SUCCESS]:\n" if torch.allclose(ret, ref, 0.001, 0.001) else "[FAILED]:\n", param, precis)
    f = open(WORKSPACE + '/log/ac.log','a')
    if torch.allclose(ret, ref, 0.001, 0.001):
        f.write("[SUCCESS]! M:{0},N:{1},K:{2},TRANSA:{3},TRANSA:{4}\n".format(param.m, param.n, param.k, param.transA, param.transB))
    else:
        f.write("[FAILED]! M:{0},N:{1},K:{2},TRANSA:{3},TRANSA:{4}\n".format(param.m, param.n, param.k, param.transA, param.transB))
    f.close()
    return


def write_profop_data(csv_name: str, resPath: str) -> None:
    PROF_HOME = os.path.join(WORKSPACE, "prof")
    time_us_list = []
    freq_list = []
    profPath = resPath
    if os.path.exists(resPath + "/_Z10GemmLaunchPhS_S_S_S_mm_mix_aic"):
        profPath = resPath + "/_Z10GemmLaunchPhS_S_S_S_mm_mix_aic"
        for path in sorted(os.listdir(profPath), key=lambda s: int(s)):
            profcsv = glob.glob(profPath + "/" + path + "/Op*.csv")
            if len(profcsv) == 0:
                print("ERROR: NO PROF DATA!!!")
                exit()
            op_basic_info = pd.read_csv(profcsv[0])
            for index, row in op_basic_info.iterrows():
                timeus = row['Task Duration(us)']
                time_us_list.append(timeus)
                freq_list.append(row['Current Freq'])
    else:
        profPath = resPath
        profcsv = glob.glob(profPath + "/Op*.csv")
        if len(profcsv) == 0:
                print("ERROR: NO PROF DATA!!!")
                exit()
        op_basic_info = pd.read_csv(profcsv[0])
        for index, row in op_basic_info.iterrows():
            timeus = row['Task Duration(us)']
            time_us_list.append(timeus)
            freq_list.append(row['Current Freq'])

    with open(os.path.join(WORKSPACE, 'log', f'res_{csv_name}'), 'a') as f:
        lines = []
        # line = "duration_time\n"
        for index in range(len(time_us_list)):
            lines.append(str(time_us_list[index]) + "," + str(freq_list[index]) + "\n")
        f.writelines(lines)

def accuracy_check(deviceId: int = 0) -> None:
    csv_list = ["test_cases.csv"]
    for csv in csv_list:
        data = pd.read_csv(os.path.join(WORKSPACE, "csv", csv), delimiter=",")
        for _, testcase in data.iterrows():
            param = OpParam()
            param.b = testcase["B"]
            param.m = testcase["M"]
            param.k = testcase["K"]
            param.n = testcase["N"]
            param.transA = bool(testcase['Transpose A'])
            param.transB = bool(testcase['Transpose B'])
            param.layoutA = CubeFormat(testcase["Data Format A"])
            param.layoutB = CubeFormat(testcase["Data Format B"])

            gen_testcase(os.path.join(WORKSPACE, 'build/data'), param)
            cmd = f"cd ../../build/examples/25_w4a8_matmul && ./25_w4a8_matmul {deviceId} {param.b} {param.m} {param.k} {param.n} {int(param.transA)} {int(param.transB)} | tee -a {WORKSPACE}/log/accur.log"
            print(cmd)
            os.system(cmd)
            compare(param)
    if os.path.exists(WORKSPACE + "/build/data"):
        os.system("rm -f {}/build/data/*".format(WORKSPACE))

def performance_test(deviceId: int = 0):
    if os.path.exists(WORKSPACE + "/log"):
        os.system("rm -f {}/log/*".format(WORKSPACE))
    if os.path.exists(WORKSPACE + "/prof"):
        os.system("rm -rf {}/prof/*".format(WORKSPACE))
    csv_list = ["1_600_general_top60.csv",
                "2_1200_random.csv",
                "3_200_random_large-m.csv",
                "4_700_general_model.csv",
                "5_800_matmul-fa.csv",
                "model.csv"]
    csv_list = ["6_four_model.csv"]
    csv_list = ["test_cases.csv"]
    # csv_list = ["rand_0_10000.csv"]
    csv_list = ["rand_1_10000_round.csv"]
    # csv_list = ["model.csv"]
    for csv in csv_list:
        rowNum = len(open(WORKSPACE + "/csv/" + csv).readlines()) - 1
        launchCount = 100
        for index in range(0, int((rowNum + launchCount - 1) / launchCount), 1):
            csvIndex = index * launchCount
            csvRowNum = min(launchCount, rowNum - csvIndex)
            cmd = f"./w4a8 {deviceId} {csv} {csvIndex}"
            print(cmd)
            # os.system(f"cd {WORKSPACE}/build/bin && {cmd}")
            os.system(f"cd {WORKSPACE}/build/bin && msprof op \
                --kernel-name=_Z10GemmLaunchPhS_S_S_S_mm    \
                --warm-up=50 \
                --launch-count={csvRowNum}                    \
                --application=\"{cmd}\"                       \
                --output={WORKSPACE}/prof | tee {WORKSPACE}/log/output.log")
            
            with open(f"{WORKSPACE}/log/output.log", 'r') as f:
                lines = f.readlines()  # 读取所有行
                resPath = lines[-3].split(" ")[-1].replace('\n', '')
                # print(resPath)
                write_profop_data(csv, resPath)

def clean_space() :
    if os.path.exists(WORKSPACE + "/build/data"):
        os.system("rm -f {}/build/data/*".format(WORKSPACE))
    if os.path.exists(WORKSPACE + "/build/bin"):
        os.system("rm -f {}/build/bin/*".format(WORKSPACE))
    if os.path.exists(WORKSPACE + "/prof"):
        os.system("rm -rf {}/prof/*".format(WORKSPACE))
    if os.path.exists(WORKSPACE + "/log"):
        os.system("rm -f {}/log/*".format(WORKSPACE))

if __name__ == "__main__":

    os.makedirs(WORKSPACE + "/build", exist_ok=True)
    os.makedirs(WORKSPACE + "/build/bin", exist_ok=True)
    os.makedirs(WORKSPACE + "/build/data", exist_ok=True)
    os.makedirs(WORKSPACE + "/prof", exist_ok=True)
    os.makedirs(WORKSPACE + "/log", exist_ok=True)
    
    op = str(sys.argv[1])
    if op == "clean":
        clean_space()
        exit()
    if op == "build":
        build(WORKSPACE)
        exit()
    
    deviceId = int(sys.argv[2])
    if op == "accur" :
        accuracy_check(deviceId)
    elif op == "prof":
        performance_test(deviceId)
    else:
        pass