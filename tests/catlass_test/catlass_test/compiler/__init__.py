import os
import re
import subprocess
from typing import Dict

from inflection import underscore
from loguru import logger

from catlass_test import (
    ASCEND_HOME_PATH,
    CATLASS_INCLUDE_PATH,
    CATLASS_TEST_INCLUDE_PATH,
    CATLASS_TEST_KERNEL_PATH,
    CATLASS_TEST_PATH,
)

CATLASS_KERNEL_ENTRY_FILE = os.path.join(CATLASS_TEST_PATH, "csrc", "kernel.cpp")

COMPILER_INCLUDE_DIRECTORIES = [
    f"-I{CATLASS_TEST_INCLUDE_PATH}",
    f"-I{ASCEND_HOME_PATH}/include",
    f"-I{ASCEND_HOME_PATH}/include/experiment/runtime",
    f"-I{ASCEND_HOME_PATH}/include/experiment/msprof",
    f"-I{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw",
    f"-I{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl",
    f"-I{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface",
    f"-I{CATLASS_INCLUDE_PATH}",
]

COMPILER_COMPILE_OPTIONS = [
    "-xcce",
    "-std=c++17",
    "-Xhost-start",
    "-O0",
    "-g",
    "-Xhost-end",
    "-Wno-macro-redefined",
]
COMPILER_LLVM_COMPILE_OPTIONS = [
    "--cce-aicore-arch=dav-c220",
    "-mllvm",
    "-cce-aicore-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-function-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-record-overflow=true",
    "-mllvm",
    "-cce-aicore-addr-transform",
    "-mllvm",
    "-cce-aicore-dcci-insert-for-scalar=false",
]


def list_to_dict(s):
    return {
        parts[1]: parts[0]
        for item in [x.strip() for x in s.split(",")]
        for parts in [item.split()]
        if len(parts) == 2
    }


COMPILER_LINK_DIRECTORIES = [
    f"-L{ASCEND_HOME_PATH}/lib64",
]
COMPILER_LINK_LIBRARIES = [
    "-ltiling_api",
    "-lascendcl",
    "-lstdc++",
]


class TemplateCompiler:
    def __init__(self, kernel_template_src: str):
        self.kernel_template_src = kernel_template_src
        self.__init_kernel_name_and_params()

    def __init_kernel_name_and_params(self):
        pattern = re.compile(
            r"template\s*<([^>]+)>\s*(?:inline\s+)?int32_t\s+(\w+)\s*\(([^)]+)\)"
        )
        with open(self.kernel_template_src, mode="r+") as kernel_template_src_handle:
            match = pattern.search(kernel_template_src_handle.read())
            if match is not None:
                self.compile_params = list_to_dict(match.group(1))
                self.kernel_name = match.group(2)
                self.orig_runtime_params = match.group(3)
                runtime_params = list_to_dict(match.group(3))
                self.runtime_params = {}
                for var_name, var_type in runtime_params.items():
                    if var_name.startswith("*") or var_name.startswith("&"):
                        self.runtime_params[var_name[1:]] = f"{var_type}*"
                    else:
                        self.runtime_params[var_name] = var_type

    @property
    def runtime_params_call(self):
        return ",".join(self.runtime_params.keys())

    def compile(self, compile_definitions: Dict[str, str]) -> str:
        """编译算子"""
        # logger.info(f"compiling kernel {kernel_name}")

        dcompile_params = []
        for var_name in self.compile_params.keys():
            dcompile_params.append(compile_definitions.get(var_name))
        compile_params = ",".join(dcompile_params)
        kernel_name = (
            "_".join(
                [f"lib{underscore(self.kernel_name)}"]
                + [
                    dcompile_param.split("::")[-1].lower()
                    for dcompile_param in dcompile_params
                ]
            )
            + ".so"
        )
        kernel_full_path = os.path.join(CATLASS_TEST_KERNEL_PATH, kernel_name)
        if os.path.exists(kernel_full_path):
            return kernel_full_path
        COMPILER_DEFINATIONS = [
            f"-DKERNEL_TEMPLATE_FILE={self.kernel_template_src}",
            f"-DCOMPILE_PARAM={compile_params}",
            f"-DRUNTIME_PARAM={self.orig_runtime_params}",
            f"-DRUNTIME_PARAM_CALL={self.runtime_params_call}",
            f"-DKERNEL_TEMPLATE_NAME={self.kernel_name}",
            f"-DTILING_KEY_VAR",
        ]
        command = (
            ["ccec"]
            + COMPILER_COMPILE_OPTIONS
            + COMPILER_DEFINATIONS
            + COMPILER_LLVM_COMPILE_OPTIONS
            + COMPILER_INCLUDE_DIRECTORIES
            + COMPILER_LINK_DIRECTORIES
            + COMPILER_LINK_LIBRARIES
            + [
                "-fPIC",
                "--shared",
                CATLASS_KERNEL_ENTRY_FILE,
                "-o",
                kernel_full_path,
            ]
        )

        # 执行命令
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(
                "Compile failed! The return code of compiler is not 0. Error info: "
            )
            logger.error(result.stderr)
        return kernel_full_path
