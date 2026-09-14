# MultiCoreSplitkMatmul Example Readme

## 代码组织

```text
├── 68_ascend950_multi_core_splitk_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── multi_core_splitk_matmul.cpp # 主文件
```

## 模板说明

该模板为多核切K模板，通过切分`K`，划分出更多的任务块，从而利用更多的计算核心。

```sh
# 编译指定用例
bash scripts/build.sh 68_ascend950_multi_core_splitk_matmul -DCATLASS_ARCH=3510
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./68_ascend950_multi_core_splitk_matmul 256 512 1024 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```


## 模板推荐场景

推荐 MNK 范围：该模板主要用于解决负载不均衡，如果划分出来的任务块比核心数量还少，那么可以通过切分`K`，划分出更多计算任务，从而让各个核心都能参与计算。约定L1上的分块大小为`m1 n1 k1`，设 `blocks = ceil(M/m1)×ceil(N/n1)`，推荐 `blocks ≤ C/2 且 K > 1024`（C 为 AI Core 核数）的块少深 K 场景；切 K 因子由框架按 shape 与核数自动推导。
