# 构建 API 文档

TLA DSL API 文档由脚本读取 `catlass.core_api` 的运行时对象生成 Markdown，再由
MkDocs 构建为静态 HTML。

## 1. 安装文档构建依赖

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python -m pip install -r requirements.txt
```

## 2. 生成 API Reference Markdown

生成脚本会导入 `catlass.core_api`。执行前，`PYTHONPATH` 必须包含已构建 AscendNPU-IR 提供的 `mlir_core`：

```bash
export ASCEND_NPU_IR_ROOT="${CATLASS_ROOT}/python/tla_dsl/3rdparty/AscendNPU-IR"
export PYTHONPATH="${ASCEND_NPU_IR_ROOT}/build/install/python_packages/mlir_core${PYTHONPATH:+:${PYTHONPATH}}"
```

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python tools/generate_api_reference.py
```

生成结果：`docs/api-reference.md`

## 3. 构建静态 HTML

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python -m mkdocs build --strict
```

构建结果：`site/index.html`（已 `.gitignore` 忽略）

## 4. 本地实时预览

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python -m mkdocs serve
```

打开终端输出的地址（通常 `http://127.0.0.1:8000/`）即可在浏览器预览。
