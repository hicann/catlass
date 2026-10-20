# MLA Example Readme

## 代码组织

```text
├── 19_mla
│   ├── CMakeLists.txt # CMake编译文件
│   ├── gen_data.py
│   ├── kernel_common.hpp #两个不同的kernel实现中的共同变量与宏
│   ├── mla.cpp
│   ├── mla_kernel.cpp # MLA TP 2/4/8 模板
│   ├── mla_kernel_tp1_spec.cpp # MLA TP 1 模板
│   ├── amla_kernel_tp1_spec.cpp # AMLA TP 1 模板
│   └── README.md
```

## 使用范围说明

以下是本样例的参数约束及推荐场景：

| 参数 | 说明 | 约束 | 推荐范围 |
| --- | --- | --- | --- |
| `headDim`（`D`） | Query/Key 非 Rope 部分头维度 | 固定为 `512` | `512` |
| `qkRopeDim`（`D_r`） | Query/Key Rope 部分头维 | 固定为 `64` | `64` |
| `numHeads`（`N_h`） | Query 头数 | 取值 {16, 32, 64, 128} | `16` / `32` / `64` / `128` （取128时走 TP1-spec kernel） |
| `kvHeads` | KV 头数 | 固定为 `1`（整除 `numHeads`） | `1` |
| `batch` | batch 数量 | `≥ 1`；通用 kernel（`N_h ∈ {16,32,64}`）受 AIC 核数限制，TP1-spec kernel（`N_h = 128`）`batch ≤ 40` | `1` ~ `16` |
| `qSeqlen`（`S_q`） | 每 batch 的 Query 序列长度 | `1 ≤ qSeqlen ≤ 4` | `1` ~ `4` |
| `kvSeqlen`（`S_kv`） | 每 batch 的 KV 序列长度 | `128 ≤ kvSeqlen ≤ 16384` 且 `kvSeqlen ≥ qSeqlen` | `128` ~ `16384` |
| `blockSize`（`B`） | Paged KV cache 分块大小 | 固定为 `128` | `128` |
| `numBlocks` | KV cache 物理块数 | `numBlocks ≥ batch * ceil(kvSeqlen / blockSize)` | — |
| `sparseMode` | 掩码模式 | `0`：无 mask；`1`：causal mask | `0` / `1` |
| `dtype` | 数据类型 | `half` / `bf16` | `half` / `bf16` |

推荐范围：
- 使用 Paged KV cache承载长上下文，`blockSize` 固定为 `128`, `kvSeqlen` 可覆盖至 `16384`。
- `Q`/`K`（以及其 Rope 部分）张量的数值范围推荐在 `[-5, 5]`以内，避免过大的动态范围致使绝对误差超阈值上限

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)
- 第一步，首先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入。

```bash
# 在./examples/19_mla目录下执行
python gen_data.py 4 "1,2,3,4" "128,256,512,1024" 16 64 128 half
# 输入参数分别对应 batchSize，qSeqlen，kvSeqlen, qheadNum，numBlock, blockSize
# qSeqlen表示需要推理的token个数，支持范围为1~4，即常规decode与mtp场景
# kvSeqlen表示输入的序列长度
# blockSize参数当前支持128
# 最后一个参数需要指明数据类型为“half”或“bf16”
```

执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据

```text
├── data
│   ├── block_table.bin
│   ├── golden.bin
│   ├── k.bin
│   ├── k_rope.bin
│   ├── kv_seqlen.bin
│   ├── q.bin
│   ├── q_ntokens.bin
│   ├── q_rope.bin
│   └── q_seqlen.bin
```

第二步，执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。

```bash
# 编译指定用例（在CATLASS代码仓目录下）
bash scripts/build.sh 19_mla
cd output/bin
./19_mla 4 "1,2,3,4" "128,256,512,1024" 16 64 128
# 此处的参数和生成数据的参数保持一致
# 完整参数为 batch "qSeqlenList" "kvSeqlenList" numHeads numBlocks blockSize [--dtype DTYPE --datapath DATA_PATH --device DEVICE_ID]，dtype默认为half, datapath默认为../../examples/19_mla/data, device默认为0。
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## 精度回归

参考计算统一由 `gen_data.py` 提供。批量用例使用同一个 attention 参考实现的 FP64 模式；
原有单用例命令保留 FP32 参考。脚本只负责数据生成，不执行算子、比较输出或维护 tiling key。
需要安装 `numpy` 和 `ml_dtypes`。

```bash
# 在代码仓根目录运行；--suite smoke 仅生成基础用例
python examples/19_mla/gen_data.py --suite full --binary ./output/bin/19_mla --output-dir ./mla_test_data --device 0
./output/bin/19_mla --test ./mla_test_data --device 0
# 单独运行某个用例
./output/bin/19_mla --test ./mla_test_data --device 0 --case bf16_repeated_underflow
```

以下为 Ascend 910B2（24 个 AIC 核）上使用默认种子的成功运行输出，`...` 表示省略的日志。
数据生成完成后，各用例会打印 `Generated`，并在输出目录写入 `cases.txt`：

```text
Generated half_zero
Generated half_cancel
...
Generated bf16_tail_tasks
```

完整回归通过时，所有用例显示 `[       OK ]`，最终汇总为：

```text
[ RUN      ] half_zero
...
Compare success.
[       OK ] half_zero
...
[       OK ] bf16_tail_tasks
MLA tests: 37 run, 0 failed
```

单独运行 `bf16_repeated_underflow` 的结果如下：

```text
[ RUN      ] bf16_repeated_underflow
TILING_FORMERTASKNUM = 1
TILING_TAILTASKNUM = 0
TILING_KVSPLIT = 1536
TILING_KVCORENUM = 1
tilingKey : 5
elements=65536 mismatches=0 max_abs=2.95331e-42 rmse=2.95331e-42
Compare success.
[       OK ] bf16_repeated_underflow
MLA tests: 1 run, 0 failed
```

`mismatches=0` 表示该用例没有超出容差的元素，`0 failed` 表示所有执行的用例通过，程序退出码为 0。
具体 tiling 参数、元素数量和误差数值可能随设备核数和生成数据而变化。

生成器查询设备实际 AIC 核数，构造跨任务和尾任务输入，并在所有参考结果生成完成后写入 `cases.txt`。
切换到不同核数的设备时需要重新生成数据。测试入口位于 `mla.cpp`，直接调用现有 tiling 和 kernel
启动路径，使用 `common/golden` 的逐元素比较；下溢用例额外要求非零结果及正确符号。
数值错误、未覆盖预期 kernel、输入文件缺失或尺寸错误均返回非零退出码。
原有单用例执行的精度比较失败也返回非零退出码。
