# BasicMatmulTlaGemv Example Readme

## 功能说明

- 算子功能：完成基础矩阵向量乘（GEMV）计算
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{j} &= \Sigma_{k} A_{k}B_{k,j}
    \end{aligned}
  $$

  其中$A$是形如`(1,k)`的输入向量，$B$是形如`(k,n)`的输入矩阵，$C$是形如`(1,n)`的输出向量。
- 支持产品型号：Ascend 950PR&DT 系列产品

## 样例参数说明

| 参数       | 属性 |shape|dtype| 说明                                                             |
| ---------- | ------ |--------|---------|----------------------------------------------- |
| `A`        | Input  | `(1,k)` | `float` | 左向量，layout为`VectorLayout`  |
| `B`        | Input  | `(k,n)` | `float` | 右矩阵，layout为`ColumnMajor` |
| `C`        | Output | `(1,n)` | `float` | 矩阵向量乘结果，layout为`RowMajor` |

## 使用范围说明

推荐范围：
- `k ≤ 7168`

## 使用示例

### 命令行参数

```bash
50_ascend950_basic_matmul_gemv [m] [n] [k] [deviceId]
```
上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | A 矩阵的 m 轴大小，固定为1 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。本样例为 Ascend 950 算子，编译时需追加 `-DCATLASS_ARCH=3510`，详见[快速开始](../../docs/zh/1_Practice/01_quick_start.md#编译执行)。

    ```bash
    bash scripts/build.sh 50_ascend950_basic_matmul_gemv -DCATLASS_ARCH=3510
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./50_ascend950_basic_matmul_gemv 1 128 127 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```

## 使用说明

BasicMatmul默认使用的DispatchPolicy MmadPingpong支持以下几个模板参数：

| 模板参数         | 默认值 | 参数说明                                         |
| ---------------- | ------ | ------------------------------------------------ |
| ArchTag          | 无     | 指定架构型号                                     |
| enableUnitFlag   | false  | 是否开启Unitflag，开启L0C多缓冲时必须设置为false |
| useHF32          | false  | 是否开启HF32，仅float类型支持                    |
| l0CStages        | 1      | 指定L0C的缓冲区数量，设置为2即可开启L0C双缓冲    |
| enableL1Resident | false  | 是否开启L1常驻                                   |
| l1AStages        | 1      | L1上加载矩阵A的Buffer数量                        |
| l1BStages        | 1      | L1上加载矩阵B的Buffer数量                        |
| l0AStages        | 1      | L0上加载矩阵A的Buffer数量                        |
| l0BStages        | 1      | L0上加载矩阵B的Buffer数量                        |

设矩阵Shape为`M N K`, L1上的分块大小为`m1 n1 k1`，M方向的分块数量`mTiles = CeilDiv(M, m1)`，N方向的分块数量`nTiles = CeilDiv(N, n1)`，总任务数为`taskBlocks = mTiles * nTiles`，在以下两种情况下可以选择开启enableL1Resident：

1.`mTiles = 1`，且`nTiles > CoreNum`，且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`n1`设置为原来的一半。

2.`nTiles = 1`，且`mTiles > CoreNum`, 且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`m1`设置为原来的一半。
