"""Pre-build hook: auto-generate nav from directory structure."""

import re
from pathlib import Path

SECTION_TITLES = {
    "1_Practice": "实践指南",
    "2_Design": "设计总结",
    "3_API": "API 参考",
    "0x_new_versions": "新版本",
    "evaluation": "调测工具",
    "others": "其他实践",
    "01_kernel_design": "算法设计",
    "00_basics": "基础知识",
    "02_tla": "TLA 设计",
    "03_evg": "EVG 设计",
}

ROOT_PAGES = [
    ("../README.md", "项目介绍"),
    ("../CHANGELOG.md", "版本日志"),
    ("../CONTRIBUTING.md", "贡献指南"),
    ("../SECURITYNOTE.md", "安全须知"),
]


def _h1_title(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
        m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return _prettify(path.stem)


def _prettify(stem: str) -> str:
    name = re.sub(r"^\d+_", "", stem)
    name = name.replace("_", " ")
    return " ".join(w[0].upper() + w[1:] if w else "" for w in name.split())


def _scan_dir(dir_path: Path, rel_to: Path, depth: int = 0) -> list:
    """Scan *dir_path*, emit paths relative to *rel_to*."""
    if depth > 5:
        return []
    nav = []
    try:
        entries = sorted(dir_path.iterdir(), key=lambda p: p.name)
    except PermissionError:
        return []

    has_readme = (dir_path / "README.md").exists()

    for path in entries:
        name = path.name
        if name.startswith(".") or name in ("figures", "README.md", "index.md"):
            continue
        if path.suffix not in (".md",) and not path.is_dir():
            continue

        rel = path.relative_to(rel_to)

        if path.is_dir():
            children = _scan_dir(path, rel_to, depth + 1)
            if not children:
                continue
            title = SECTION_TITLES.get(name, _prettify(name))
            if len(children) == 1 and isinstance(children[0], dict):
                nav.append({title: list(children[0].values())[0]})
            else:
                nav.append({title: children})
        else:
            title = _h1_title(path)
            nav.append({title: str(rel)})

    if not nav and has_readme:
        nav.append(
            {
                _prettify(dir_path.name): str(
                    (dir_path / "README.md").relative_to(rel_to)
                )
            }
        )

    return nav


def on_pre_build(config, **kwargs):
    docs_dir = Path(config["docs_dir"])  # docs/
    root_dir = docs_dir.parent  # repo root
    nav = []

    # Home: docs hub (docs/zh/README.md)
    nav.append({"Home": "zh/README.md"})

    # Root pages (path relative to docs_dir: ../README.md etc.)
    for path, title in ROOT_PAGES:
        nav.append({title: path})

    # Scan docs/zh/ sections
    zh = docs_dir / "zh"
    if zh.is_dir():
        for entry in sorted(zh.iterdir()):
            if (
                not entry.is_dir()
                or entry.name.startswith(".")
                or entry.name == "figures"
            ):
                continue
            children = _scan_dir(entry, docs_dir)
            if children:
                nav.append(
                    {SECTION_TITLES.get(entry.name, _prettify(entry.name)): children}
                )
        # Q&A
        qa = zh / "Q&A.md"
        if qa.exists():
            nav.append({"常见问题": str(qa.relative_to(docs_dir))})

    # Scan examples/ (outside docs_dir, paths use ../ prefix)
    examples_dir = root_dir / "examples"
    if examples_dir.is_dir():
        children = _scan_dir(examples_dir, root_dir)
        if children:
            # Rewrite paths relative to docs_dir by prepending ../
            def _rewrite(nodes):
                for item in nodes:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if isinstance(v, str):
                                item[k] = "../" + v
                            elif isinstance(v, list):
                                _rewrite(v)

            _rewrite(children)
            nav.append({"算子示例": children})

    config["nav"] = nav
