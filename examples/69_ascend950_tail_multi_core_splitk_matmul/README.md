# TailMultiCoreSplitkMatmul Example Readme

## 代码组织

```text
├── 69_ascend950_tail_multi_core_splitk_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── tail_multi_core_splitk_matmul.cpp # 主文件
```

## 模板说明

该模板为切尾轮基本块的多核切K模板，通过切分尾轮基本块的`K`，划分出更多的任务块，从而利用更多的核心参与尾轮的基本块计算。

```sh
# 编译指定用例
bash scripts/build.sh 69_ascend950_tail_multi_core_splitk_matmul -DCATLASS_ARCH=3510
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./69_ascend950_tail_multi_core_splitk_matmul 256 512 1024 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```


## 模板推荐场景

推荐 MNK 范围：约定L1上的分块大小为`m1 n1 k1`，设 `blocks = ceil(M/m1)×ceil(N/n1)`，推荐 `C < blocks < 1.5C 且 K > 1024`（C 为 AI Core 核数），即 1~1.5 波且尾轮有残余块的场景。
