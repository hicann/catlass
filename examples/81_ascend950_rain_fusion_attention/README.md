# FlashAttentionInfer Example Readme

## 代码组织

```text
├── 81_ascend950_rain_fusion_attention
│   ├── CMakeLists.txt  # CMake编译文件
│   ├── rfa_kernel.cpp
│   ├── rfa_kernel_utils.cpp
│   ├── rfa_tiling.cpp
│   ├── rfa_tilingdata.h
│   ├── rfa.cpp
│   ├── gen_data.py
│   └── README.md
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#算子编译)

- 接下来，先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入, 执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据。
- 然后执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。

以下是一个完整的shell脚本示例

```text
batch=1
qSeqlen=512
kvSeqlen=1024
numHeads=8
kvHeads=1
headSize=128
blockShapeX=128
blockShapeY=256
dtype="half"  # "half", "bf16"
qInputLayout="TND"   # "TND", "BNSD"
kvInputLayout="TND"  # "TND", "BNSD"
isVariedLen=0
device=0
innerPrec=0 # 仅gen_data.py使用，对应kernel逻辑固定为0

function build() {
    rm -rf build
    rm -rf output
    bash scripts/build.sh -DCATLASS_ARCH=3510 81_ascend950_rain_fusion_attention --clean
}

function gen_data() {
    rm -rf examples/81_ascend950_rain_fusion_attention/data
    python3 examples/81_ascend950_rain_fusion_attention/gen_data.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $blockShapeX $blockShapeY \
        "$dtype" $qInputLayout $kvInputLayout $isVariedLen
    echo "Data gen finished"
}

function run_kernel {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads ' headSize=' $headSize ' bX=' $blockShapeX ' bY=' $blockShapeY
    cd output/bin/
    ./81_ascend950_rain_fusion_attention $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $blockShapeX $blockShapeY \
        $dtype $qInputLayout $kvInputLayout $isVariedLen --device $device
}

build
gen_data
run_kernel
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## 已支持特性

|            特性             |          对应参数            |
| :-------------------------: | :------------------------: |
|          数据类型            |    dtype="half"/"bf16"     |
|          输入布局            |  qInputLayout="TND"/"BNSD" |
|          输入布局            | kvInputLayout="TND"/"BNSD" |
|       不同batch序列可变       |      isVariedLen=0/1       |
|          headSize           |      qkHeadSize=64/128     |
|        sequence大小         |      kvSeqlen > qSeqlen    |
