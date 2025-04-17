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

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))+"/torch_act"
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPython3_EXECUTABLE=" + sys.executable,
            "-DBUILD_PYBIND=True"
        ]

        build_args = []
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", ".", "-j"] + build_args, cwd=self.build_temp)

version = f"0.1.0.{time.strftime('%Y%m%d%H%M%S')}"

setup(
    name="torch_act",
    version="0.1.0",
    author="HW",
    description="A PyTorch extension with pybind11 bindings",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=["torch_act"],  # 如果有纯 Python 模块
    ext_modules=[CMakeExtension("torch_act")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires = [],
    include_package_data=True,
)