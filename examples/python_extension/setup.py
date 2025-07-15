
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import time


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
            self.generate_pyi(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir + "/torch_catlass",
            "-DPython3_EXECUTABLE=" + sys.executable,
            "-DBUILD_PYBIND=True"
        ]

        build_args = []
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", os.path.join(ext.sourcedir, "../../")] +
                              cmake_args, cwd=self.build_temp)
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "_C", "-j"] + build_args, cwd=self.build_temp)

    def generate_pyi(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        module_name = ext.name.split(".")[-1]
        stubgen_args = [module_name, "--output-dir", extdir]
        subprocess.check_call(["pybind11-stubgen"] + stubgen_args, cwd=extdir)


version = f"0.1.0.{time.strftime('%Y%m%d%H%M%S')}"

setup(
    name="torch_catlass",
    version=version,
    author="Huawei Technologies Co., Ltd.",
    description="A PyTorch extension for AscendC Tenplates with pybind11 bindings",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=["torch_catlass"],
    ext_modules=[CMakeExtension("torch_catlass")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[],
    include_package_data=True,
)
