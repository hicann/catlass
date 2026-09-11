# XAttention Example

## 功能说明

本样例将 xllm-ops 的 `x_attention` 迁移为 CATLASS 可独立编译、运行和精度验证的 example。算子同时计算两部分 KV：

- shared KV：同一 batch 内的 beam 共享上下文；
- unshared KV：每个 beam 独立维护的 decode 上下文；
- 最后使用两部分 softmax 的 row max/row sum 合并输出。

支持 fp16、bf16，以及以下两种互斥缓存组合：

| cacheMode | shared KV | unshared KV |
| --- | --- | --- |
| `0` | Paged Attention | 连续内存 |
| `1` | 连续内存 | Paged Attention |

## 代码组织

```text
78_x_attention/
├── CMakeLists.txt
├── README.md
├── gen_data.py
├── x_attention.cpp            # Host 侧数据加载、kernel 启动和精度校验
├── x_attention_device.hpp     # Device 入口
├── x_attention_helper.hpp     # 公共参数、Tiling 数据与 CATLASS 模板组装
├── x_attention_kernel.hpp     # shared/unshared/combine kernel 调度
└── x_attention_tiling.hpp     # Example 形态的 Host tiling
```

## 约束

以下是当前 example 与 ATK 交付件共用的输入约定。example 参数使用 camelCase，Python/ATK 使用对应的 snake_case。

| 参数（example / Python、ATK） | 约束 |
| --- | --- |
| `batch`、`beamSize / beam_size` | 正整数 |
| `sharedKvSeqLen / shared_kv_seq_len` | 正整数，不要求是 128 的倍数 |
| `numHeads / num_heads`、`kvHeads / kv_heads` | 正整数，`num_heads % kv_heads == 0`，且 `1 <= num_heads / kv_heads <= 128` |
| `embeddingSize / embedding_size`（head dim） | 固定为 128；ATK 根据该常量生成 shape |
| `decodeStep / decode_step`、`maxDecodeStep / max_decode_step` | `1 <= decode_step <= max_decode_step <= 256`，分别表示有效 decode 长度和缓存容量 |
| `cacheMode / cache_mode` | 只能为 0 或 1，含义见上表 |
| Q/K/V dtype | 所有 Q/K/V 使用同一种 FP16 或 BF16；example 命令行名称为 `half` / `bf16` |

- 运行平台：Atlas A2/Atlas A3（`CATLASS_ARCH=2201`），Host tiling 要求至少两个 Cube 核；当前不支持 attention mask。
- **128-token 分页块仅指 `cache_mode=0` 的 shared KV**：shape 为 `(num_blocks, 128, kv_heads, 128)`，每个 batch 分配 `ceil(shared_kv_seq_len / 128)` 个块，最后一块允许不满。`cache_mode=1` 的 unshared KV 按 request 索引，shape 为 `(request_count, beam_size, kv_heads, max_decode_step, 128)`，其 decode 容量不固定为 128。
- example 与 ATK 均生成 `shared_kv_lens[:] = shared_kv_seq_len`，即各 batch 元素使用相同的有效 shared KV 长度，同一 batch 元素内的所有 beam 共享该 KV。这里描述的是这两个数据生成入口的等长约定，不代表底层分页 kernel 只能读取等长序列。
- `128` 是 GQA 分组大小的上限，不是 `num_heads`、`kv_heads` 或 `beam_size` 的独立上限。ATK YAML 中列举的取值是测试采样集合，不是算子的完整支持范围；实际规模还受设备内存和内部整数索引范围限制。

### PyTorch 接口的附加约定

`torch_catlass.x_attention` 从缓存 shape 和 block table 推导上述参数，不接收 `cache_mode` 标量：两个 block table 必须恰好一个非空，另一个为 `None` 或空张量。所有传入张量应在同一 NPU 上且连续；K/V 的对应 shape 必须一致。

- `shared_kv_lens` 为 int32、shape `(batch,)`；`decode_step` 为 int32、shape `(1,)`，所有 batch/beam 共用这个有效 decode 长度。
- block table 为 int32。shared table 的 shape 为 `(batch, max_blocks_per_batch)`，当前接口要求 `num_blocks == batch * max_blocks_per_batch`，有效块索引在 `[0, num_blocks)`；unshared table 的 shape 为 `(batch,)`，request 索引在 `[0, request_count)`。
- shared 分页模式下，每个有效 shared 长度须在 `[1, max_blocks_per_batch * 128]`；本交付件采用上述等长数据。shared 连续模式按有效长度累加定位下一 batch，使用本交付件的等长布局时，长度必须与缓存的 `shared_kv_seq_len` 一致，不能把 padding 当作 batch 间间隔。
- `scale_value >= 0`，其中 `0` 表示默认的 `1 / sqrt(128)`；example 和 ATK 使用默认 scale。

上述是调用方必须满足的约定，并非每一项都有运行时检查：当前 C++ wrapper 检查 shape/dtype 等信息，不读取 NPU 上的长度和索引值来检查其范围。自定义输入也必须保证这些值有效；example 的 `.bin` 元数据必须与命令行参数一致。

核对依据：[`gen_data.py`](gen_data.py)、[`x_attention_tiling.hpp`](x_attention_tiling.hpp)、[`x_attention_helper.hpp`](x_attention_helper.hpp)、[`x_attention_kernel.hpp`](x_attention_kernel.hpp) 及 [PyTorch C++ wrapper](../../tests/optest/src/include/template/x_attention.h)。ATK 对应说明位于 `catlass-test-scripts/atk/78_x_attention/README.md`。

## 使用示例

在 CATLASS 仓库根目录执行：

```bash
batch=1
beamSize=4
sharedKvSeqLen=512
numHeads=8
kvHeads=2
embeddingSize=128
maxDecodeStep=32
decodeStep=16
cacheMode=0
dtype=half

python3 examples/78_x_attention/gen_data.py \
    ${batch} ${beamSize} ${sharedKvSeqLen} ${numHeads} ${kvHeads} ${embeddingSize} \
    ${maxDecodeStep} ${decodeStep} ${cacheMode} ${dtype}

bash scripts/build.sh 78_x_attention -DCATLASS_ARCH=2201

cd output/bin
./78_x_attention \
    ${batch} ${beamSize} ${sharedKvSeqLen} ${numHeads} ${kvHeads} ${embeddingSize} \
    ${maxDecodeStep} ${decodeStep} ${cacheMode} --dtype ${dtype}
```

执行成功时输出：

```text
Compare success.
```
