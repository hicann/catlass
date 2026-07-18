import os
import glob
import shutil
import subprocess
import sys

import lit.formats

config.name = "ascend-catlass-dsl"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir", ".test"]
config.excludes = ["lit.cfg.py", "lit.site.cfg.py", "regression-template.mlir"]

config.test_source_root = config.tla_lit_source_dir
config.test_exec_root = config.tla_lit_binary_dir

path_entries = []
if getattr(config, "llvm_tools_dir", ""):
    path_entries.append(config.llvm_tools_dir)
conda_prefix = os.environ.get("CONDA_PREFIX", "")
if conda_prefix:
    conda_llvm_libexec = os.path.join(conda_prefix, "libexec", "llvm")
    if os.path.isdir(conda_llvm_libexec):
        path_entries.append(conda_llvm_libexec)
for tool in (config.tla_compile_tool,):
    parent = os.path.dirname(tool)
    if parent:
        path_entries.append(parent)

filecheck_tool = getattr(config, "filecheck_tool", "") or "FileCheck"
if os.path.isabs(filecheck_tool):
    path_entries.append(os.path.dirname(filecheck_tool))
else:
    filecheck_on_path = shutil.which(filecheck_tool, path=os.pathsep.join(path_entries))
    if filecheck_on_path:
        filecheck_tool = filecheck_on_path
    else:
        llvm_config = shutil.which("llvm-config")
        if llvm_config:
            try:
                llvm_bindir = subprocess.check_output(
                    [llvm_config, "--bindir"],
                    text=True,
                ).strip()
            except Exception:
                llvm_bindir = ""
            if llvm_bindir:
                path_entries.append(llvm_bindir)
                candidate = os.path.join(llvm_bindir, "FileCheck")
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    filecheck_tool = candidate
        if not os.path.isabs(filecheck_tool):
            for candidate in glob.glob("/usr/lib/llvm-*/bin/FileCheck"):
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    path_entries.append(os.path.dirname(candidate))
                    filecheck_tool = candidate
                    break
        if not os.path.isabs(filecheck_tool):
            for candidate in (
                "/opt/conda-env/libexec/llvm/FileCheck",
                "/opt/conda-env/bin/FileCheck",
            ):
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    path_entries.append(os.path.dirname(candidate))
                    filecheck_tool = candidate
                    break

existing_path = os.environ.get("PATH", "")
if existing_path:
    path_entries.append(existing_path)
config.environment["PATH"] = os.pathsep.join(path_entries)

python_path_entries = [os.path.dirname(os.path.dirname(config.tla_lit_source_dir))]
ascendnpu_ir_root = getattr(config, "tla_ascendnpu_ir_root", "")
mlir_python_root = os.path.join(
    ascendnpu_ir_root, "build", "install", "python_packages", "mlir_core"
)
if os.path.isdir(mlir_python_root):
    python_path_entries.append(mlir_python_root)
existing_python_path = os.environ.get("PYTHONPATH", "")
if existing_python_path:
    python_path_entries.append(existing_python_path)
config.environment["PYTHONPATH"] = os.pathsep.join(python_path_entries)

config.filecheck_tool = filecheck_tool
config.substitutions.append(("%tla_compile", config.tla_compile_tool))
config.substitutions.append(("%filecheck", config.filecheck_tool))
config.substitutions.append(("%tla_lit_src", config.tla_lit_source_dir))
config.substitutions.append(("%python", sys.executable))

config.environment["FILECHECK_OPTS"] = "-enable-var-scope --allow-unused-prefixes=false"
