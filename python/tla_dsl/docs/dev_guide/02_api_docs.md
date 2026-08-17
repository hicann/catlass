# 构建 API 文档

TLA DSL API 文档由脚本通过 AST 静态解析 `catlass.core_api` 生成 Markdown，再由
MkDocs 构建为静态 HTML。

## 1. 安装文档构建依赖

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python -m pip install -r requirements.txt
```

## 2. 生成 API Reference Markdown

源码 docstring 使用英文段落标题，并带目录标签：

- `Directory:`：该 API 在参考文档 TOC 中的路径（可用 `/` 表示嵌套）
- `Description:` / `Parameters:` / `Constraints:` / `Example:`

生成器根据 `Directory:` 建目录树；章节顺序与章节简介写在脚本的
`DIRECTORY_SECTIONS`（API docstring 只保留归属路径与接口说明）。
同节内 API 顺序跟随源码定义顺序。
生成器**只产出**英文参考文档 `docs/kernel-api-reference.md`（自动生成，勿手改）。
中文版 `docs/kernel-api-reference.zh.md` 为**手工维护**（不以术语表 / glossary 自动生成）；英文稿变更后请同步翻译更新中文稿。

生成脚本通过 AST 解析源码，不要求导入已构建的 `mlir_core`：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python tools/generate_api_reference.py
```

生成结果（仅英文）：`docs/kernel-api-reference.md`

手工维护的中文参考（不由上述命令生成）：`docs/kernel-api-reference.zh.md`

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
