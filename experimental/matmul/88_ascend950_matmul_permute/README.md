# MatmulPermute Example Readme

## 功能说明

- 算子功能：完成矩阵乘并将结果从二维布局转换为四维布局
- 支持产品型号：Ascend 950PR&950DT系列产品
- 计算公式：

  $$
    C = A \times B
  $$

  其中 `A`、`B` 的形状分别为 `(M,K)`、`(K,N)`，中间结果 `C` 的形状为
  `(M,N)`。

- 输出变换：

  $$
    D = C.reshape(M / 8, 8, 4, N / 4).permute(0, 2, 1, 3)
  $$

  输出 `D` 的逻辑形状为 `(M / 8, 4, 8, N / 4)`。算子在 matmul epilogue
  阶段直接按照变换后的地址写回，避免额外启动 permute kernel。

## 参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(M,K)` | `float16` | 左矩阵，RowMajor |
| `B` | Input | `(K,N)` | `float16` | 右矩阵，RowMajor |
| `D` | Output | `(M / 8, 4, 8, N / 4)` | `float16` | MatmulPermute 输出结果 |

## 约束说明

输入矩阵的维度需要满足以下约束：

| 参数 | 约束 | 说明 |
| --- | --- | --- |
| `M` | `M % 8 == 0` | 满足输出布局变换要求 |
| `N` | `N % 4 == 0` | 满足输出布局变换要求 |

## 使用示例

### 命令行参数

```bash
88_ascend950_matmul_permute [M] [N] [K] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `M` | 无 | 左矩阵的行数 |
| `N` | 无 | 右矩阵的列数 |
| `K` | 无 | 左矩阵的列数 |
| `deviceId` | `0` | 指定运行设备 ID |

### 执行示例

1. 将样例加入 `examples` 构建目录后，进入项目根目录编译样例代码：

    ```bash
    bash scripts/build.sh 88_ascend950_matmul_permute -DCATLASS_ARCH=3510
    ```

2. 切换到可执行文件目录并执行样例：

    ```bash
    cd output/bin
    ./88_ascend950_matmul_permute 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
