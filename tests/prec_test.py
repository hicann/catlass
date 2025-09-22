import glob
import logging
import os
import re
import sys
from typing import Dict, List, Set

import networkx as nx

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
# 正则表达式匹配包含"catlass"或"tla"的头文件包含语句
CATLASS_INCLUDE_PATTERN = re.compile(r'#include\s*"([^"]*(catlass|tla)[^"]*)"')
CATLASS_REPO_PATH = ""
CATLASS_EXAMPLES_PATH = os.path.join(CATLASS_REPO_PATH, "examples")
CATLASS_INCLUDE_PATH = os.path.join(CATLASS_REPO_PATH, "include")

# 缓存文件内容，避免重复读取
FILE_CONTENT_CACHE: Dict[str, List[str]] = {}


def get_example_list() -> List[str]:
    """
    获取所有示例目录的完整路径列表

    Returns:
        示例目录完整路径列表，按数字顺序排序
    """
    example_pattern = os.path.join(CATLASS_EXAMPLES_PATH, "[0-9][0-9]*")
    example_dirs = [d for d in glob.glob(example_pattern) if os.path.isdir(d)]
    example_dirs.sort()
    return example_dirs


def get_file_content(file_path: str) -> List[str]:
    """
    获取文件内容，使用缓存避免重复读取

    Args:
        file_path: 文件完整路径

    Returns:
        文件内容行列表
    """
    if file_path in FILE_CONTENT_CACHE:
        return FILE_CONTENT_CACHE[file_path]

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.readlines()
            FILE_CONTENT_CACHE[file_path] = content
            return content
    except IOError as e:
        logging.error(f"警告: 无法读取文件 {file_path}: {e}")
        return []


def get_src_dep_file_list(src_file: str) -> List[str]:
    """
    获取源文件依赖的catlass头文件完整路径列表

    Args:
        src_file: 源文件完整路径

    Returns:
        该文件依赖的头文件完整路径列表
    """
    src_dep_list = []
    content = get_file_content(src_file)

    for line in content:
        if (match := CATLASS_INCLUDE_PATTERN.search(line)) is not None:
            # 将相对路径转换为完整路径
            include_path = match.group(1)
            abs_include_path = os.path.join(CATLASS_INCLUDE_PATH, include_path)
            src_dep_list.append(abs_include_path)

    return src_dep_list


def get_example_file_list(example_path: str) -> List[str]:
    """
    获取示例目录中的所有C++源文件完整路径

    Args:
        example_path: 示例目录完整路径

    Returns:
        示例目录中的所有C++源文件完整路径列表
    """
    example_file_list = []

    # 查找所有C++源文件
    for ext in ["*.h", "*.hpp", "*.cpp"]:
        pattern = os.path.join(example_path, "**", ext)
        file_list = glob.glob(pattern, recursive=True)
        example_file_list.extend(file_list)

    return example_file_list


def build_dep_graph() -> nx.DiGraph:
    """
    构建依赖关系图，使用完整路径作为节点

    Returns:
        有向图，节点表示示例或头文件的完整路径，边表示依赖关系
    """
    G = nx.DiGraph()

    # 添加示例节点及其依赖
    example_dirs = get_example_list()
    for example_path in example_dirs:
        G.add_node(example_path)
        logging.debug(f"add {example_path}")
        example_file_list = get_example_file_list(example_path)

        for example_file in example_file_list:
            G.add_edge(example_path, example_file)
            _add_file_with_deps(G, example_file)

    # 添加头文件节点及其依赖
    catlass_header_files = glob.glob(
        os.path.join(CATLASS_INCLUDE_PATH, "**/*.hpp"), recursive=True
    )

    for header_file in catlass_header_files:
        _add_file_with_deps(G, header_file)

    return G


def _add_file_with_deps(G: nx.DiGraph, file_path: str) -> None:
    """
    添加文件及其依赖关系到图中

    Args:
        G: 依赖关系图
        file_path: 文件路径
    """
    if not G.has_node(file_path):
        G.add_node(file_path)
        logging.debug(f"add {file_path}")

    dep_file_list = get_src_dep_file_list(file_path)
    for dep_file in dep_file_list:
        if not G.has_node(dep_file):
            G.add_node(dep_file)
            logging.debug(f"add {dep_file}")
        if not G.has_edge(file_path, dep_file):
            G.add_edge(file_path, dep_file)
            logging.debug(f"add {file_path} -> {dep_file}")


def get_inferred_example_list(src_file: str, graph: nx.DiGraph) -> Set[str]:
    """
    获取受给定头文件影响的所有示例完整路径

    Args:
        src_file: 头文件完整路径
        graph: 依赖关系图

    Returns:
        受影响的示例完整路径集合
    """
    inferred_example_list = set()

    # 确保src_file是完整路径
    if not os.path.isabs(src_file):
        src_file = os.path.join(CATLASS_REPO_PATH, src_file)

    # 检查所有示例是否依赖于该头文件
    for example in get_example_list():
        if nx.has_path(graph, example, src_file):
            inferred_example_list.add(example)
    return inferred_example_list


# 测试代码
if __name__ == "__main__":
    diff_file_list_file = sys.argv[1]
    delimeter = sys.argv[2]
    CATLASS_REPO_PATH = sys.argv[3]
    diff_file_list = []
    with open(diff_file_list_file, mode="r+") as diff_file_list_file_handle:
        for line in diff_file_list_file_handle.readlines():
            line_strip = line.strip()
            if line_strip.startswith("examples") or line_strip.startswith("include"):
                diff_file_list.append(line_strip)
    graph = build_dep_graph()
    affected_examples = set()
    for diff_file in diff_file_list:
        affected_examples = affected_examples.union(
            get_inferred_example_list(diff_file, graph)
        )
    print(delimeter.join(list(map(lambda s: s.split("/")[-1], affected_examples))))
