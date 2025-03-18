## 程序运行步骤

## 1.编译指令 
../../scripts/build.sh 04_groupgemm
## 2.运行指令：其中，groupCnt为矩阵个数，mlist、nlist、klist为进行每个矩阵计算的shape，device_id为运行的npu设备号
../../build/bin/04_groupgemm groupCnt mlist nlist klist deviceId

## 注意：examples/CMakeLists.txt里CCEC_COMPILER_OPTIONS若为-O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220-vec不能直接运行
## 需要修改为-O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220