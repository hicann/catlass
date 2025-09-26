SCRIPT_PATH=$(dirname "$(realpath "$0")")
DIFF_LIST=${SCRIPT_PATH}/../../difflist.txt
git diff --name-only --output=${DIFF_LIST}
examples=$(python3 ${SCRIPT_PATH}/../prec_test.py ${DIFF_LIST} " or ")
echo "pytest filter is ${examples}"
pytest --collect-only -k "${examples}"