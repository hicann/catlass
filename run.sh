source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh 32_matmul_add_evt
cd output/bin
./32_matmul_add_evt 256 512 1024 0
