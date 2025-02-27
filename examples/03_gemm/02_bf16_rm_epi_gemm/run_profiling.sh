M=${1}
N=${2}
K=${3}
device=${4}
# 这个文件存在问题
# rm -rf ./quant_gemm/*
# python3 test.py $M $N $K
# python3 prof.py `find ./quant_gemm/ -name "op_summary*.csv" ` $M $N $K
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
rm -rf ./prof/*
mkdir -p ./prof ./data/input ./data/output
python3 ./scripts/gen_data.py $M $N $K
../../../scripts/build.sh 02_bf16_rm_epi_gemm  # 编译命令
# msprof op --application="python3 test.py $problemCount $mList $nList $kList" --output=../prof --launch-count=10 --launch-skip-before-match=5 --kernel-name=GroupedMatmul 
msprof op --output=./prof --launch-count=5 ../../../build/bin/02_bf16_rm_epi_gemm $M $N $K $device
python3 ./prof.py `find ./prof/** -name "OpBasicInfo.csv" ` `find ./prof/** -name "PipeUtilization.csv" ` $M $N $K
rm -rf ./data/input/* ./data/output/*