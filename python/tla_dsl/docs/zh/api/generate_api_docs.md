---
nav_order: 20
---

# 生成 API 文档

CATLASS DSL API 文档由脚本通过 AST 静态解析源码生成 Markdown，再由仓库根目录
MkDocs 构建为静态站点。

- Kernel API：`core_api.py`、`tla/tensor.py` → `docs/en/api/kernel_api_reference.md`
- Host API：Host 源文件 → `docs/en/api/host_api_reference.md`

公共解析 / Markdown 渲染在 `tools/common.py`。

## 前置条件

生成脚本仅做 AST 解析。构建 MkDocs 站点前安装文档依赖：

```bash
# /path/to/catlass 需替换为你 clone 的 CATLASS 仓库实际路径
cd /path/to/catlass/python/tla_dsl
python -m pip install -r requirements-docs.txt
```

## 生成 API Reference Markdown

源码 docstring 使用英文段落标题，并带目录标签：

- `Directory:`：该 API 在参考文档 TOC 中的路径（可用 `/` 表示嵌套）
- `Description:` / `Parameters:` / `Constraints:` / `Example:`

生成器根据 `Directory:` 建目录树；章节顺序与章节简介写在脚本的
`DIRECTORY_SECTIONS`（Kernel）与 `HOST_DIRECTORY_SECTIONS`（Host）。
同节内 API 顺序跟随源码定义顺序。

生成器产出英文参考文档（自动生成，勿手改）：

- `docs/en/api/kernel_api_reference.md`（`tools/generate_kernel_api_reference.py`）
- `docs/en/api/host_api_reference.md`（`tools/generate_host_api_reference.py`）

中文版 `docs/zh/api/kernel_api_reference.md` 与 `docs/zh/api/host_api_reference.md`
为**手工维护**；英文稿变更后请同步翻译更新中文稿。

环境变量见 [环境变量](../kernel_development/core_concepts/env_vars.md)，**不由** Host API
生成器扫描。

```bash
cd /path/to/catlass/python/tla_dsl
python tools/generate_kernel_api_reference.py
python tools/generate_host_api_reference.py
```

检查生成文件是否与当前代码一致，但不改写文件：

```bash
python tools/generate_kernel_api_reference.py --check
python tools/generate_host_api_reference.py --check
```

`--check` 在文件过期时输出 diff 并返回非零状态，适合用于提交前检查。
