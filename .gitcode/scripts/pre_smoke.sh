#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 不使用 set -e：单个测试失败后继续执行其他测试，最后统一返回失败数量。
set -uo pipefail

DRY_RUN=${CI_DRY_RUN:-0}
FAILED=0

WORKSPACE=$1
cd "${WORKSPACE}"

# ============================================================================
# 1. 获取本次提交修改的文件
# ============================================================================

CURRENT_HASH=$(git rev-parse HEAD) || exit 1
DIFFLIST_FILE="/tmp/difflist_${CURRENT_HASH:0:16}.txt"
git diff --name-only HEAD~1 HEAD > "${DIFFLIST_FILE}" 2>/dev/null || true

# 判断本次提交是否修改了匹配路径的文件。
has_change() {
    grep -qE "$1" "${DIFFLIST_FILE}"
}

# 统一执行一个测试，并记录结果。
# 执行前打印完整命令（含参数，dry-run 与真实执行都可见），失败时保留命令上下文。
run_test() {
    local name=$1
    shift

    echo
    echo "=== [${name}] ==="
    printf '    >>> '
    printf '%q ' "$@"
    echo

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "    [DRY RUN]"
        return 0
    fi

    if "$@"; then
        echo "    [OK] ${name}"
        return 0
    fi

    echo "    [FAIL] ${name}"
    FAILED=$((FAILED + 1))
    return 1
}

echo "============================================================"
echo "CATLASS CI itemized tests"
echo "difflist: ${DIFFLIST_FILE}"
echo "============================================================"
echo "changed files:"
cat "${DIFFLIST_FILE}"

# 只有文档类文件变化时，不需要执行测试。
# CMakeLists.txt 虽然以 .txt 结尾，但属于构建配置，不能跳过。
if [ -s "${DIFFLIST_FILE}" ]; then
    if grep -qE '(^|/)CMakeLists\.txt$' "${DIFFLIST_FILE}"; then
        : # 修改了 CMakeLists.txt，继续执行 CI
    elif grep -qvE '\.(md|rst|txt)$' "${DIFFLIST_FILE}"; then
        : # 存在非文档文件，继续执行 CI
    else
        echo
        echo "Only doc change, skip all tests."
        exit 0
    fi
fi

# ============================================================================
# 2. 根据修改文件决定需要执行哪些测试
# ============================================================================

RUN_SELF_CONTAINED=false
RUN_UNITTEST=false
RUN_PYEXT=false
RUN_CPPGEN=false
RUN_MSTUNER=false
RUN_DSL=false

EXAMPLE_CASES=()

# include/ 会影响头文件自包含测试、unittest 和相关 example。
if has_change '^include/'; then
    RUN_SELF_CONTAINED=true
    RUN_UNITTEST=true
fi

# include/ 或 examples/NN_* 变化时，计算受影响的 example。
if has_change '^include/|^examples/[0-9]'; then
    example_output=$(python3 "${WORKSPACE}/tests/tools/get_test_example_case_list.py" \
        --difflist "${DIFFLIST_FILE}") || exit 1

    # case 名不包含空格，用数组保存，后面传命令参数更安全。
    example_output=${example_output//$'\n'/ }
    read -r -a EXAMPLE_CASES <<< "${example_output}"
fi

if has_change '^tests/unittest'; then
    RUN_UNITTEST=true
fi

if has_change '^python/tla_dsl'; then
    RUN_DSL=true
fi

if has_change '^examples/python_extension/'; then
    RUN_PYEXT=true
fi

if has_change '^python/catlass_cppgen'; then
    RUN_CPPGEN=true
fi

if has_change '^tools/tuner'; then
    RUN_MSTUNER=true
fi

if has_change '^tests/test_torch_lib\.py$|^include/catlass/torch/|^src/torch/'; then
    RUN_PYEXT=true
fi

# 这几个 example 同时被 python extension 覆盖。
for case_name in "${EXAMPLE_CASES[@]}"; do
    case "${case_name}" in
        00_basic_matmul|02_grouped_matmul_slice_m|24_conv_bias)
            RUN_PYEXT=true
            ;;
    esac
done

echo
echo "============================================================"
echo "Test items to run:"
echo "  example_cases : ${EXAMPLE_CASES[*]:-<none>}"
echo "  pyext         : ${RUN_PYEXT}"
echo "  cppgen        : ${RUN_CPPGEN}"
echo "  mstuner       : ${RUN_MSTUNER}"
echo "  unittest      : ${RUN_UNITTEST}"
echo "  self_contained: ${RUN_SELF_CONTAINED}"
echo "  dsl           : ${RUN_DSL}"
echo "============================================================"

# ============================================================================
# 3. DSL CI：ut_type=ir 或 lit 时，只执行 DSL 测试
# ============================================================================

if [ "${ut_type:-}" = "ir" ] || [ "${ut_type:-}" = "lit" ]; then
    if [ "${RUN_DSL}" != "true" ]; then
        echo "no dsl ut (python/tla_dsl unchanged)"
        exit 0
    fi

    export CATLASS_DSL_PREBUILT_ASCENDNPU_IR=/opt/AscendNPU-IR
    cd "${WORKSPACE}/python/tla_dsl" || exit 1

    if [ "${ut_type}" = "ir" ]; then
        run_test "dsl_build" bash "${WORKSPACE}/python/tla_dsl/build.sh" || true
        run_test "dsl_pytest" python -m pytest -q "${WORKSPACE}/python/tla_dsl/tests" || true
    else
        export PATH="/opt/AscendNPU-IR/build/bin:${PATH}"
        run_test "dsl_build" bash "${WORKSPACE}/python/tla_dsl/build.sh" || true
        run_test "dsl_lit" llvm-lit -sv "${WORKSPACE}/python/tla_dsl/csrc/mlir/build/tests/lit" || true
    fi

    exit "${FAILED}"
fi

# ============================================================================
# 4. 准备常规测试环境
# ============================================================================

if [ "${DRY_RUN}" -ne 1 ]; then
    pip3 install ml_dtypes expecttest pybind11-stubgen pytest pytest-xdist || exit 1

    # shellcheck source=/dev/null
    source /usr/local/Ascend/ascend-toolkit/set_env.sh || exit 1

    export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
    unset ASCEND_RT_VISIBLE_DEVICES
    export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
fi

cd "${WORKSPACE}" || exit 1

# ============================================================================
# 5. 执行常规分项测试
# ============================================================================

if [ "${RUN_SELF_CONTAINED}" = "true" ]; then
    run_test "self_contained_includes" \
        bash "${WORKSPACE}/scripts/build.sh" --clean --tests test_self_contained_includes || true
fi

if [ "${RUN_UNITTEST}" = "true" ]; then
    run_test "unittest" bash "${WORKSPACE}/tests/run_unittest.sh" || true
fi

# 受影响 example：跑 test_example.py（C++ bin）+ 对应 optest（torch_catlass 算子测试）
if [ "${#EXAMPLE_CASES[@]}" -gt 0 ]; then
    EXAMPLE_FILTER=""
    OPTEST_KEYWORDS=()
    for case_name in "${EXAMPLE_CASES[@]}"; do
        if [ -n "${EXAMPLE_FILTER}" ]; then
            EXAMPLE_FILTER="${EXAMPLE_FILTER} or "
        fi
        EXAMPLE_FILTER="${EXAMPLE_FILTER}${case_name}"
        # optest 关键词：去掉数字前缀（00_basic_matmul -> basic_matmul）
        OPTEST_KEYWORDS+=("${case_name#*_}")
    done

    run_test "example" \
        python3 -m pytest -q "${WORKSPACE}/tests/test_example.py" -k "${EXAMPLE_FILTER}" || true

    if [ "${#OPTEST_KEYWORDS[@]}" -gt 0 ]; then
        run_test "optest" \
            bash "${WORKSPACE}/tests/run_optest.sh" || true
    fi
fi

# python extension / torch lib 
if [ "${RUN_PYEXT}" = "true" ]; then
    run_test "python_extension" bash "${WORKSPACE}/tests/run_python_extension.sh" || true
fi

if [ "${RUN_CPPGEN}" = "true" ]; then
    run_test "cppgen" bash "${WORKSPACE}/tests/run_cppgen.sh" || true
fi

if [ "${RUN_MSTUNER}" = "true" ]; then
    run_test "mstuner" python3 "${WORKSPACE}/tools/tuner/test/test_mstuner.py" || true
fi

# ============================================================================
# 6. 汇总结果
# ============================================================================

echo
echo "============================================================"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "DRY RUN complete."
elif [ "${FAILED}" -eq 0 ]; then
    echo "All targeted tests passed."
else
    echo "SUMMARY: ${FAILED} item(s) failed."
fi
echo "============================================================"

exit "${FAILED}"
