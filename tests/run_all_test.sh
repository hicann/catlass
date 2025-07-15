SCRIPT_PATH=$(dirname "$(realpath "$0")")
BUILD_SCRIPT_PATH=$(realpath "$SCRIPT_PATH"/../scripts/build.sh)

# example test
bash "$BUILD_SCRIPT_PATH" --clean catlass_examples || exit 1
python3 "$SCRIPT_PATH/test_example.py"

# python extension
bash "$BUILD_SCRIPT_PATH" --clean python_extension || exit 1
WHEEL_DIR="$SCRIPT_PATH/../output/python_extension/"
WHEEL_FILE=$(find "$WHEEL_DIR" -type f -name "torch_catlass-*.whl" 2>/dev/null | head -n 1)
if [ -z "$WHEEL_FILE" ]; then
    echo "Error: No .whl file found in $WHEEL_DIR"
    exit 1
fi
pip install "$WHEEL_FILE"
python3 "$SCRIPT_PATH/test_python_extension.py"
pip uninstall torch_catlass

# torch lib
bash "$BUILD_SCRIPT_PATH" --clean torch_library || exit 1
python3 "$SCRIPT_PATH/test_torch_lib.py"