#!/bin/bash

# default input file
LAUNCH_SRC_FILE="_gen_launch.cpp"
OUTPUT_LIB_FILE="_gen_module.so"

if [ $# -ge 1 ] ; then
    LAUNCH_SRC_FILE=$1
fi
if [ $# -ge 2 ]; then
    OUTPUT_LIB_FILE=$2
fi

LAUNCH_OBJ_FILE="${LAUNCH_SRC_FILE%.cpp}.o"
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
    -I../../../include \
    -I../../common \
    -Wno-macro-redefined -Wno-ignored-attributes \
    -fPIC -c $LAUNCH_SRC_FILE -o $LAUNCH_OBJ_FILE

COMPILE_RET=$? 
if [ $COMPILE_RET -ne 0 ] ; then
    exit $COMPILE_RET
fi

# customize your own linking options here
# e.g. add an additional -l parameter to link to a custom shared library
 bisheng --cce-fatobj-link -fPIC  -shared -o $OUTPUT_LIB_FILE $LAUNCH_OBJ_FILE \
     -L$ASCEND_HOME_PATH/lib64 \
     -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase

exit $?