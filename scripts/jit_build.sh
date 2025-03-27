#!/bin/sh

# default input file
launch_src_file="_gen_launch.cpp"
output_lib_file="_gen_module.so"

if [ $# -ge 1 ] ; then
    launch_src_file=$1
fi
if [ $# -ge 2 ]; then
    output_lib_file=$2
fi

launch_obj_file="${launch_src_file%.cpp}.o"
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

cd "$(dirname "$0")"

# customize your own compile options here
# e.g. add an additional -I parameter to include custom header file directory
bisheng -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220 \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -I$ASCEND_HOME_PATH/compiler/tikcpp \
    -I$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw \
    -I$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/impl \
    -I$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/interface \
    -I$ASCEND_HOME_PATH/include \
    -I$ASCEND_HOME_PATH/include/experiment/runtime \
    -I$ASCEND_HOME_PATH/include/experiment/msprof \
    -I$PYTHON_INCLUDE \
    -I../include \
    -I../examples/common \
    -Wno-macro-redefined -Wno-ignored-attributes \
    -fPIC -c $launch_src_file -o $launch_obj_file

compile_ret=$? 
if [ $compile_ret -ne 0 ] ; then
    exit $compile_ret
fi

# customize your own linking options here
# e.g. add an additional -l parameter to link to a custom shared library
bisheng --cce-fatobj-link -fPIC --cce-aicore-arch=dav-c220 -shared -o $output_lib_file $launch_obj_file \
    -L$ASCEND_HOME_PATH/lib64 \
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase

exit $?
