#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 使用 .venv 中的 Python
PYTHON="$SCRIPT_DIR/optest/.venv/bin/python"
echo "Using python: $($PYTHON --version 2>&1) at $PYTHON"

# 1. 清理旧的编译产物
echo "============================================"
echo "Step 1: Cleaning old build artifacts..."
echo "============================================"
rm -rf optest/build optest/dist optest/*.egg-info optest/_skbuild

# 2. 编译 optest
echo ""
echo "============================================"
echo "Step 2: Building optest..."
echo "============================================"
cd "$SCRIPT_DIR/optest"
bash build.sh

# 3. 安装 optest（从 dist）
echo ""
echo "============================================"
echo "Step 3: Installing optest from dist..."
echo "============================================"
cd "$SCRIPT_DIR/optest"
WHEEL_FILE=$(ls -1 dist/*.whl | head -1)
echo "Installing: $WHEEL_FILE"
$PYTHON -m pip install "$WHEEL_FILE" --force-reinstall --no-deps

# 4. 运行测试
echo ""
echo "============================================"
echo "Step 4: Running tests..."
echo "============================================"
cd "$SCRIPT_DIR/optest"
$PYTHON -m pytest tests/ -v

# 5. 卸载 optest
echo ""
echo "============================================"
echo "Step 5: Uninstalling optest..."
echo "============================================"
$PYTHON -m pip uninstall torch-catlass -y

echo ""
echo "============================================"
echo "All steps completed successfully!"
echo "============================================"