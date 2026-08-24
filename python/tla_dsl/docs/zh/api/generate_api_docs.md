---
nav_order: 20
---

# 生成 API 文档

CATLASS DSL API 文档由脚本通过 AST 静态解析 `catlass.core_api` 生成 Markdown。

## 前置条件

生成脚本会导入 `catlass.core_api`，因此需要先完成 [Debug 开发构建](../dsl_development/build_guide/index.md#development-模式构建开发态)：

## 生成 API Reference Markdown

源码 docstring 使用英文段落标题，并带目录标签：

- `Directory:`：该 API 在参考文档 TOC 中的路径（可用 `/` 表示嵌套）
- `Description:` / `Parameters:` / `Constraints:` / `Example:`

生成器根据 `Directory:` 建目录树；章节顺序与章节简介写在脚本的
`DIRECTORY_SECTIONS`（API docstring 只保留归属路径与接口说明）。
同节内 API 顺序跟随源码定义顺序。
生成器**只产出**英文参考文档 `docs/en/api/kernel_api_reference.md`（自动生成，勿手改）。
中文版 `docs/zh/api/kernel_api_reference.md` 为**手工维护**（不以术语表 / glossary 自动生成）；英文稿变更后请同步翻译更新中文稿。

生成脚本通过 AST 解析源码，不要求导入已构建的 `mlir_core`：

## 生成 Core API Reference

```bash
cd /path/to/catlass/python/tla_dsl
python tools/generate_api_reference.py
```

生成结果（仅英文）：`docs/en/api/kernel_api_reference.md`

手工维护的中文参考（不由上述命令生成）：`docs/zh/api/kernel_api_reference.md`

检查生成文件是否与当前代码一致，但不改写文件：

```bash
python tools/generate_api_reference.py --check
```

`--check` 在文件过期时输出 diff 并返回非零状态，适合用于提交前检查。
