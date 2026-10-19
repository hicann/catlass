# MultipleBlocksSimt Example Readme

## 功能说明

- 算子功能：以SIMT模式完成一维向量加法，同时在block与thread两级上切分数据。
- 计算公式：

  $$
    C_{i} = A_{i} + B_{i}
  $$

  其中 $A$、$B$、$C$ 均为形如 `(n,)` 的一维向量。

- 本样例的重点不在算法本身，而在两级并行的写法：

  | 层级 | 接口 | 查询位置 |
  | --- | --- | --- |
  | block间 | `tla.arch.block_idx()` / `tla.arch.block_num()` | SIMT区域**外**获取block编号和数量，block_num值由host侧launch kernel时以启动参数的形式传入 |
  | block内 | `tla.arch.thread_idx()` / `tla.arch.thread_block_dim()` | SIMT区域**内**获取线程索引和线程块几何(三维) |

  本样例中，线程块为三维的`(64, 4, 2)`，即每个block内512个线程。kernel内先把block内三维线程索引展平为`tid`，再按SIMT区域内全局线程索引`gid = block_idx * tdim + tid`、SIMT区域内线程总数作为步长`stride = block_num * tdim`跨步遍历，将整个数组分发给所有线程。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `gm_a` | Input | `(n,)` | `f32` | 左加数，一维连续张量 |
| `gm_b` | Input | `(n,)` | `f32` | 右加数，一维连续张量 |
| `gm_c` | Output | `(n,)` | `f32` | 加法结果，一维连续张量 |
| `THREADS` | 启动参数 | `(tx, ty, tz)` | int 三元组 | 线程块几何，乘积为每block线程数；样例默认 `(64, 4, 2)` 即512线程，经`tla.vec.func`的`thread_block_dim`传入 |
| `block_num` | 启动参数 | 标量 | int | launch时指定的线程块数，即`--block-dim`参数（默认取设备 AIV 核数） |

> [注]
> `n` 即样例内的元素个数常量 `N_ELE`（样例默认 65536），`tx`/`ty`/`tz` 与使用范围说明中的线程几何记号一致。`N_ELE`、`THREADS` 均为样例内的常量，修改后需重新运行以触发重新编译。
>
> 一维张量经 `from_dlpack` 绑定时 `layout_tag` 固定取 `RowMajor`（一维无行列次序之分，该标签仅是绑定接口的必填形式）。

## 使用范围说明

本样例为 SIMT 多 block 切分的最小示例，不含 tiling、double buffer 等优化。

使用约束：

- 元素类型：`f32`（fp32 加法在 Host 与设备上逐位一致，可精确比对）。
- 输入为一维`(n,)`，`1 ≤ n ≤ 65536`。
- 启动参数受官方上限约束：grid中线程块总数`block_num`不超过65535；单个线程块的线程数`tx·ty·tz`不超过2048。
- 在上述上限内，线程几何 `(tx, ty, tz)` 与线程块数任意取值均支持，包括`n`小于总线程数（多数线程空转一轮）的场景。
- 线程总数`tx·ty·tz`建议为32的倍数，属官方性能建议而非正确性约束，非整倍数的几何已验证支持。
- 官方文档给出线程数与每线程可用寄存器数的对应关系（如 1025~2048 线程时每线程 16 个寄存器），在大线程数下编写寄存器占用较高的 kernel 需自行评估溢出风险。

输出缓冲区在下发前被填充为哨兵值`-999.0`，运行结束后单独统计仍为哨兵值的元素个数（`untouched`）。跨步写错时表现为`untouched`不为0，这是切分错误而非算术错误，因此与精度校验分开报告。

## 使用示例

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
