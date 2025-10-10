source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh 32_matmul_add_evt
if [ $? -eq 0 ]; then
    # export ASCEND_SLOG_PRINT_TO_STDOUT=1
    cd output/bin
    ./32_matmul_add_evt 256 512 1024 0
fi
