SCRIPT_PATH=$(dirname "$(realpath "$0")")
BUILD_SCRIPT_PATH=$(realpath "$SCRIPT_PATH"/../scripts/build.sh)

bash "$BUILD_SCRIPT_PATH" --clean catlass_examples || exit 1
python3 "$SCRIPT_PATH/test_example.py"