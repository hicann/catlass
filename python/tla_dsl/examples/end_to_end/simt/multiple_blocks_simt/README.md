# MultipleBlocksSimt Example Readme

## 功能说明

- 算子功能：以 SIMT 模式完成一维向量加法，同时在 block 与 thread 两级上切分数据。
- 计算公式：

  $$
    C_{i} = A_{i} + B_{i}
  $$

  其中 $A$、$B$、$C$ 均为形如 `(n,)` 的一维向量。

- 本样例的重点不在算法本身，而在两级并行的写法：

  | 层级 | 接口 | 查询位置 |
  | --- | --- | --- |
  | block 间 | `tla.arch.block_idx()` / `tla.arch.block_num()` | SIMT 区域**之外**，以启动参数的形式捕获进区域 |
  | block 内 | `tla.arch.thread_idx()` / `tla.arch.thread_block_dim()` | SIMT 区域**之内** |

  线程几何为三维的 `(64, 4, 2)`，即每个 block 512 个线程，用于同时验证 y、z 分量。kernel 内先把三维线程号线性化为 `tid`，再按 `gid = block_idx * tdim + tid`、`stride = block_num * tdim` 跨步遍历，使整个 launch 的所有线程共同覆盖一次数组，而不是每个 block 各自重复计算全部数据。

- 支持产品型号：Ascend 950 系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `gm_a` | Input | `(65536,)` | `f32` | 左加数，layout 为 `RowMajor` |
| `gm_b` | Input | `(65536,)` | `f32` | 右加数，layout 为 `RowMajor` |
| `gm_c` | Output | `(65536,)` | `f32` | 加法结果，layout 为 `RowMajor` |

元素个数 `N_ELE`、线程几何 `THREADS` 均为样例内的常量，修改后需重新运行以触发重新编译。

## 使用范围说明

本样例为 SIMT 多 block 切分的最小示例，不含 tiling、double buffer 等优化。

输出缓冲区在下发前被填充为哨兵值 `-999.0`，运行结束后单独统计仍为哨兵值的元素个数（`untouched`）。跨步写错时表现为 `untouched` 不为 0，这是切分错误而非算术错误，因此与精度校验分开报告。

### 命令行参数

```bash
multiple_blocks_simt.py [--device deviceId] [--block-dim blockDim] [--atol atol] [--cache-dir cacheDir] [--force-recompile] [--no-cache]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `deviceId` | `0` | 指定运行设备 ID |
| `blockDim` | `-1` | 使用的 AIV 核数，`-1` 表示取设备的 `vector_core_num` |
| `atol` | `1e-4` | 精度比对的绝对误差阈值 |
| `cacheDir` | 无 | 编译缓存目录，等价于设置 `CATLASS_DSL_CACHE_DIR`；不指定时使用默认缓存位置 |
| `--force-recompile` | 关闭 | 强制重新编译，等价于设置 `CATLASS_DSL_FORCE_RECOMPILE=1` |
| `--no-cache` | 关闭 | 禁用编译缓存，等价于设置 `CATLASS_DSL_CACHE=0` |

### 执行示例

1. 进入项目根目录，参考[快速上手](../../../../docs/zh/quick_start.md)准备样例代码的运行环境。

2. 在项目根目录下执行算子样例程序。

    ```bash
    python python/tla_dsl/examples/end_to_end/simt/multiple_blocks_simt/multiple_blocks_simt.py --device 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    untouched=0/65536
    passed=True cache_key=...
    ```
