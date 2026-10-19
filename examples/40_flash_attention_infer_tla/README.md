# FlashAttentionInferTla Example Readme

## 代码组织

```text
├── 40_flash_attention_infer_tla
│   ├── CMakeLists.txt # CMake构建配置文件
│   ├── gen_data.py
│   ├── kernel_common.hpp
│   ├── fai.cpp
│   ├── fai_kernel.cpp
│   ├── fai_tiling.cpp
│   └── README.md
```

## 使用范围说明

以下是本样例的参数约束及推荐场景：

| 参数 | 说明 | 约束 | 推荐范围 |
| --- | --- | --- | --- |
| `headDim`（`embeddingSize`） | 头维度 | `128`（当前模板仅以 `128` 实例化） | `128` |
| `numHeads` | Query 头数 | `≥ kvHeads` 且为 `kvHeads` 的整数倍（GQA/MQA） | `1` ~ `16` |
| `kvHeads` | KV 头数 | `≥ 1` 且整除 `numHeads` | `1` ~ `16` |
| `batch` | batch 数量 | `≥ 1` | `1` ~ `16` |
| `qSeqlen` | 每 batch 的 Query 序列长度 | `1 ≤ qSeqlen ≤ kvSeqlen` | `1` ~ `16384` |
| `kvSeqlen` | 每 batch 的 KV 序列长度 | `≥ qSeqlen` | `1` ~ `16384` |
| `blockSize` | Paged KV cache 分块大小 | 固定为 `128` | `128` |
| `numBlocks` | KV cache 物理块数 | `numBlocks ≥ batch * ceil(kvSeqlen / blockSize)` | — |
| `maskType`（`sparseMode`） | 掩码模式 | `0`：无 mask；`1`：causal mask | `0` / `1` |
| `dtype` | 数据类型 | `half` / `bf16` | `half` / `bf16` |

推荐范围：
- `Q`/`K`/`V` 张量的数值范围推荐在 `[-1, 1]`以内，避免过大的动态范围致使绝对误差超阈值上限

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)

- 接下来，先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入, 执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据。
- 然后执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。

以下是一个完整的shell脚本示例

```text
batch=1           # batch大小
qSeqlen=177       # query序列长度
kvSeqlen=512      # key/value序列长度
numHeads=1        # query head数量
kvHeads=1         # key/value head数量
headSize=128      # embeddingSize
isVariedLen=0     # 是否使用变长序列，当前仅支持0
maskType=1        # mask类型，0表示无mask，1表示使用mask
dtype="bf16"      # 数据类型，支持"half"或"bf16"
cacheMode=1       # 缓存模式，0表示非Paged Attention，1表示Paged Attention
device=0          # 设备ID

function build() {
    rm -rf build
    rm -rf output
    bash scripts/build.sh 40_flash_attention_infer_tla
}

function gen_data() {
    python3 examples/40_flash_attention_infer_tla/gen_data.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType "$dtype" $cacheMode
    echo "Data gen finished"
}

function run_kernel() {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads ' D=' $headSize ' mask=' $maskType
    cd output/bin/
    ./40_flash_attention_infer_tla $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType --device $device --dtype $dtype
}

build
gen_data
run_kernel
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```
