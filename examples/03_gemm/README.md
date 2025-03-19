## 程序运行步骤

## 1.编译指令 
../../scripts/build.sh 03_gemm
## 2.运行指令：其中，m,n,k分别是两个矩阵的维度，device_id为运行的npu设备号
../../build/bin/03_gemm m n k deviceId

## 注意：examples/CMakeLists.txt里CCEC_COMPILER_OPTIONS若为-O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220-vec不能直接运行
## 需要修改为-O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220