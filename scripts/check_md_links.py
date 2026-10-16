#!/usr/bin/env python3
# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------
"""Check that the links inside Markdown files are valid.

Two kinds of links are verified:

* repository-relative links: the target file must exist, and if the link carries
  a ``#anchor`` the target Markdown file must contain a matching heading;
* external ``http(s)`` links: they must be reachable (HTTP status < 400). Rate limiting
  (``429``), server errors (``5xx``) and transport failures are reported as *unverified*
  and do not fail the run, because they say nothing about the link itself.

Usage::

    python scripts/check_md_links.py [options] [FILE ...]

When no FILE is given, every Markdown file tracked by git is checked. When the
pre-commit framework drives the hook it passes the staged files explicitly.

Known limitation: some documentation sites are single page applications that
answer ``200`` for any path (for example ``www.hiascend.com``). Such links are
reported as reachable but their content cannot be validated from here.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import os
import re
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCHEMES = ("http://", "https://")
SKIP_SCHEMES = ("mailto:", "tel:", "data:", "javascript:", "#")
USER_AGENT = "catlass-md-link-check/1.0"

FENCE_RE = re.compile(r"^\s*(```|~~~)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
AUTOLINK_RE = re.compile(r"<((?:https?|mailto|tel):[^>\s]+)>")
BARE_URL_RE = re.compile(r"(?<![\w(\[])(https?://[^\s<>)\]\"]+)")
HTML_ANCHOR_RE = re.compile(r"<a\s+[^>]*?(?:id|name)=\"([^\"]+)\"", re.IGNORECASE)


# --------------------------------------------------------------------------- link parsing


def _mask_code(text: str) -> str:
    """Blank out fenced code blocks and inline code, keeping the line count."""
    out = []
    in_fence = False
    for line in text.split("\n"):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append("")
            continue
        if in_fence:
            out.append("")
            continue
        # drop inline code spans so that their content is not parsed as a link
        out.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


def _parse_inline(text: str, paren_idx: int) -> tuple[str, int]:
    """Parse the destination of an inline link whose ``(`` is at paren_idx."""
    i = paren_idx + 1
    n = len(text)
    while i < n and text[i] in " \t":
        i += 1
    if i < n and text[i] == "<":
        j = text.find(">", i + 1)
        if j == -1:
            return "", i
        return text[i + 1 : j], j + 1
    depth = 0
    buf = []
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            buf.append(text[i + 1])
            i += 2
            continue
        if c == "(":
            depth += 1
            buf.append(c)
            i += 1
            continue
        if c == ")":
            if depth == 0:
                return "".join(buf).strip(), i + 1
            depth -= 1
            buf.append(c)
            i += 1
            continue
        if c in " \t" and depth == 0:
            break
        buf.append(c)
        i += 1
    return "".join(buf).strip(), i


def extract_links(masked: str) -> list[tuple[int, str]]:
    """Return ``(offset, destination)`` pairs found in masked Markdown."""
    found: list[tuple[int, str]] = []
    spans: list[tuple[int, int]] = []

    # inline links and images: [text](dest) / ![alt](dest)
    for m in re.finditer(r"\]\(", masked):
        dest, end = _parse_inline(masked, m.end() - 1)
        spans.append((m.start(), end))
        if dest:
            found.append((m.end(), dest))

    # reference definitions: [label]: dest
    for m in re.finditer(r"^ {0,3}\[[^\]]+\]:\s*(\S+)", masked, re.MULTILINE):
        found.append((m.start(1), m.group(1)))

    # autolinks: <https://...>
    for m in AUTOLINK_RE.finditer(masked):
        found.append((m.start(1), m.group(1)))

    # bare URLs that are not already part of another link
    for m in BARE_URL_RE.finditer(masked):
        pos = m.start(1)
        if any(s <= pos < e for s, e in spans):
            continue
        found.append((pos, m.group(1)))

    return found


# --------------------------------------------------------------------------- anchors


def slugify(text: str) -> str:
    """Reproduce the anchor the documentation renderer derives from a heading.

    The renderer used by this repository slugifies the whole heading text, so a
    trailing ``{ #some-id }`` is kept as literal text rather than treated as a
    custom anchor. See ``docs/zh/1_Practice/evaluation/bottleneck_analysis_and_optimization.md``
    for examples such as ``#用msprof进行单算子性能分析--msprof-single-operator-analysis-``.
    """
    text = re.sub(r"`([^`]*)`", r"\1", text).strip().lower()
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "-_ " or cat[0] in {"L", "N"}:
            chars.append(ch)
    return "".join(chars).replace(" ", "-")


def collect_anchors(path: Path) -> tuple[set[str], set[str]]:
    """Return (exact anchors, loose anchors) of a Markdown file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set(), set()

    exact: set[str] = set()
    loose: set[str] = set()
    counts: dict[str, int] = {}
    in_fence = False
    for line in text.split("\n"):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in HTML_ANCHOR_RE.finditer(line):
            exact.add(m.group(1))
        m = HEADING_RE.match(line)
        if not m:
            continue
        base = slugify(m.group(2))
        idx = counts.get(base, 0)
        counts[base] = idx + 1
        exact.add(base if idx == 0 else f"{base}-{idx}")

    loose = {re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", a.lower()) for a in exact}
    return exact, loose


def check_anchor(target: Path, anchor: str, cache: dict) -> tuple[bool, str]:
    key = str(target)
    if key not in cache:
        cache[key] = collect_anchors(target)
    exact, loose = cache[key]
    want = urllib.parse.unquote(anchor).lower()
    if want in exact:
        return True, ""
    if re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", want) in loose:
        return True, ""
    return False, f"anchor '#{anchor}' not found in {target.relative_to(REPO_ROOT)}"


# --------------------------------------------------------------------------- external


#: HTTP statuses that only mean the server refused to serve us right now, not that the
#: link is dead: 429 (rate limit) and any 5xx are reported as "unverified" instead of
#: "broken" so that a flaky host does not block a commit.
TRANSIENT_STATUSES = (429,)


def _http_status(url: str, timeout: float) -> tuple[str, str]:
    """Return ``(kind, message)`` with kind in ``{"ok", "broken", "transient"}``."""
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(
            url, method=method, headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return "ok", f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            if method == "HEAD" and exc.code in (403, 405, 501):
                continue
            if exc.code < 400:
                return "ok", f"HTTP {exc.code}"
            if exc.code in TRANSIENT_STATUSES or exc.code >= 500:
                return "transient", f"HTTP {exc.code}"
            return "broken", f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001 - URLError, timeout, ssl, ...
            if method == "HEAD":
                continue
            return "transient", f"{type(exc).__name__}: {exc}"
    return "transient", "unreachable"


def check_external(url: str, timeout: float, retries: int) -> tuple[str, str]:
    kind, last = "transient", "unreachable"
    for _ in range(retries + 1):
        kind, last = _http_status(url, timeout)
        if kind == "ok":
            break
    return kind, last


# --------------------------------------------------------------------------- main


def line_of(starts: list[int], offset: int) -> int:
    return bisect.bisect_right(starts, offset)


def is_external(url: str) -> bool:
    return url.lower().startswith(SCHEMES)


def gather_files(argv_files: list[str]) -> list[Path]:
    if argv_files:
        return [Path(f) for f in argv_files if f.lower().endswith(".md")]
    out = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return sorted(REPO_ROOT.rglob("*.md"))
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check links inside Markdown files.")
    parser.add_argument("files", nargs="*", help="Markdown files to check")
    parser.add_argument("--no-external", action="store_true", help="skip http(s) links")
    parser.add_argument(
        "--timeout", type=float, default=20.0, help="per-request timeout in seconds"
    )
    parser.add_argument(
        "--retries", type=int, default=1, help="retries for a failing external link"
    )
    parser.add_argument("--jobs", type=int, default=8, help="parallel requests")
    parser.add_argument(
        "--skip-host", action="append", default=[], help="host to ignore (repeatable)"
    )
    args = parser.parse_args()

    if os.environ.get("MDLINK_NO_EXTERNAL") == "1":
        args.no_external = True

    files = gather_files(args.files)
    problems: list[str] = []
    warnings: list[str] = []
    ext_urls: dict[str, list[tuple[Path, int, str]]] = {}
    anchor_cache: dict = {}
    checked = 0

    for path in files:
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        masked = _mask_code(raw)
        starts = [0] + [m.end() for m in re.finditer("\n", masked)]
        for offset, dest in extract_links(masked):
            checked += 1
            line = line_of(starts, offset)
            dest = dest.strip()
            if not dest:
                continue
            low = dest.lower()
            if low.startswith(SKIP_SCHEMES):
                continue
            if is_external(dest):
                host = urllib.parse.urlsplit(dest).netloc
                if any(h in host for h in args.skip_host):
                    continue
                ext_urls.setdefault(dest, []).append((path, line, dest))
                continue

            target_part, _, anchor = dest.partition("#")
            if target_part:
                if target_part.startswith("/"):
                    target = REPO_ROOT / target_part.lstrip("/")
                else:
                    target = (path.parent / urllib.parse.unquote(target_part)).resolve()
                if not target.exists():
                    rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
                    problems.append(f"{rel}:{line}: missing target '{target_part}'")
                    continue
                if anchor and target.suffix.lower() in (".md", ".markdown"):
                    ok, msg = check_anchor(target, anchor, anchor_cache)
                    if not ok:
                        rel = (
                            path.relative_to(REPO_ROOT) if path.is_absolute() else path
                        )
                        problems.append(f"{rel}:{line}: {msg}")
            elif anchor and path.suffix.lower() in (".md", ".markdown"):
                ok, msg = check_anchor(path, anchor, anchor_cache)
                if not ok:
                    rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
                    problems.append(f"{rel}:{line}: {msg}")

    if ext_urls and not args.no_external:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(check_external, url, args.timeout, args.retries): url
                for url in ext_urls
            }
            for fut in concurrent.futures.as_completed(futures):
                url = futures[fut]
                kind, msg = fut.result()
                if kind == "ok":
                    continue
                for path, line, dest in ext_urls[url]:
                    rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
                    if kind == "transient":
                        warnings.append(
                            f"{rel}:{line}: unverified link '{dest}' ({msg})"
                        )
                    else:
                        problems.append(f"{rel}:{line}: broken link '{dest}' ({msg})")

    for item in sorted(problems) + sorted(warnings):
        print(item)

    scope = "files" if args.files else "all tracked Markdown files"
    print(
        f"\nchecked {checked} link(s) in {scope}: "
        f"{len(problems)} problem(s)"
        + ("" if args.no_external else f", {len(ext_urls)} unique external url(s)")
        + (f", {len(warnings)} unverified" if warnings else "")
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
