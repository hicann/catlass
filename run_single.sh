batch=24
qSeqlen=1024
kvSeqlen=4096
numHeads=80
kvHeads=8
headSize=128
isVariedLen=0
maskType=0
dtype="half"
device=6

function build() {
    rm -rf build
    bash scripts/build.sh 23_flash_attention_infer
}

function gen_data() {
    python3 examples/23_flash_attention_infer/gen_data.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType "$dtype"
    echo "Data gen finished"
}

function run_kernel {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads ' D=' $headSize ' mask=' $maskType
    cd build/bin
    ./23_flash_attention_infer $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType --dType $dtype --device $device
    # msprof op --application="./23_flash_attention_infer $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType --dType $dtype --device $device" --output=../../prof
    # msprof op simulator --application="./23_flash_attention_infer $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType --dType $dtype --device $device" --output=../../simu
}