from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    from setuptools.command.editable_wheel import editable_wheel
except ImportError:

    class editable_wheel:
        pass


PROJECT_ROOT = Path(__file__).resolve().parent


def _detect_compiler() -> dict[str, str]:
    """Detect clang/clang++ and pin the CMake compilers to clang 19.x.

    The CC/CXX environment variables take precedence over auto-detection.
    A missing clang or a non-19.x version only prints a warning.
    """
    env: dict[str, str] = {}

    cc = os.environ.get("CC")
    cxx = os.environ.get("CXX")

    if cc and cxx:
        env["CMAKE_C_COMPILER"] = cc
        env["CMAKE_CXX_COMPILER"] = cxx
        return env

    clang = shutil.which("clang-19") or shutil.which("clang")
    clangxx = shutil.which("clang++-19") or shutil.which("clang++")

    if clang and clangxx:
        proc = subprocess.run(
            [clang, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r"clang version (\S+)", proc.stdout)
        version = match.group(1) if match else "<unknown>"

        if version.startswith("19."):
            env["CMAKE_C_COMPILER"] = cc or clang
            env["CMAKE_CXX_COMPILER"] = cxx or clangxx
        else:
            print(
                f"WARNING: clang version is {version}, expected 19.x. "
                "While GCC-based builds may work in practice, clang is the "
                "recommended compiler for TLA / MLIR development.",
                file=sys.stderr,
            )
    else:
        print(
            "WARNING: clang/clang++ not found in PATH. "
            "While GCC-based builds may work in practice, clang is the "
            "recommended compiler for TLA / MLIR development.",
            file=sys.stderr,
        )

    return env


class CMakeExtension(Extension):
    """A setuptools extension whose build is delegated to CMake."""

    def __init__(
        self,
        name: str,
        *,
        sourcedir: str,
        target: str,
    ) -> None:
        super().__init__(name=name, sources=[])
        self.sourcedir = sourcedir
        self.cmake_target = target


class CMakeBuild(build_ext):
    """Invoke CMake instead of the setuptools compiler abstraction."""

    def run(self) -> None:
        if shutil.which("cmake") is None:
            raise RuntimeError("CMake was not found. Install CMake or add it to PATH.")

        self._inplace = self.inplace
        super().run()

    def copy_extensions_to_source(self) -> None:
        pass

    def _install(self, src: Path, dst: Path) -> None:
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                symlinks=True,
                ignore=shutil.ignore_patterns("libCatlassAggregateCAPI.so"),
            )
        else:
            shutil.copy2(src, dst)

    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        source_dir = (PROJECT_ROOT / ext.sourcedir).resolve()
        extension_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        extension_dir.mkdir(parents=True, exist_ok=True)

        build_type = os.environ.get(
            "CMAKE_BUILD_TYPE",
            "Debug" if self.debug else "Release",
        )
        generator = os.environ.get("CMAKE_GENERATOR")
        if not generator and shutil.which("ninja"):
            generator = "Ninja"
        extra_cmake_args = shlex.split(os.environ.get("CMAKE_ARGS", ""))

        build_dir = self._get_build_directory(
            build_type=build_type,
            generator=generator,
            extra_cmake_args=extra_cmake_args,
        )
        build_dir.mkdir(parents=True, exist_ok=True)

        configure_command = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_INSTALL_PREFIX={build_dir / 'install'}",
            # Use the host sysroot; CMAKE_ARGS appended later can override it.
            "-DCMAKE_SYSROOT=/",
        ]

        for name in ("CATLASS_INCLUDE_DIR",):
            value = os.environ.get(name)
            if value:
                configure_command.append(f"-D{name}={value}")

        compiler_env = _detect_compiler()
        for key, value in compiler_env.items():
            configure_command.append(f"-D{key}={value}")

        if generator:
            configure_command.extend(["-G", generator])
        configure_command.extend(extra_cmake_args)

        subprocess.run(
            configure_command,
            cwd=PROJECT_ROOT,
            check=True,
        )

        # --- build → install → wheel pattern ---
        # cmake build
        build_command = [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            build_type,
            "--target",
            *shlex.split(ext.cmake_target),
        ]
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            jobs = self.parallel or os.cpu_count() or 1
            build_command.extend(["--parallel", str(jobs)])

        subprocess.run(
            build_command,
            cwd=PROJECT_ROOT,
            check=True,
        )

        # cmake install: all artifacts (.so, _mlir, bc stubs + headers) land in one prefix
        install_prefix = build_dir / "install"
        subprocess.run(
            ["cmake", "--install", str(build_dir)],
            cwd=PROJECT_ROOT,
            check=True,
        )

        installed_catlass = install_prefix / "catlass"
        if not installed_catlass.is_dir():
            raise RuntimeError(
                f"cmake --install produced no catlass/ tree at {installed_catlass}"
            )

        if not self._inplace:
            # Copy cmake install artifacts INTO the setuptools wheel staging dir.
            # Do NOT replace extension_dir — setuptools already populated it with
            # Python source files (__init__.py etc.) via normal package discovery.
            so_files = list(installed_catlass.glob("_tla_type_bridge_native*.so"))
            if not so_files:
                raise RuntimeError(
                    f"_tla_type_bridge_native*.so not found in {installed_catlass}"
                )
            for so in so_files:
                self._install(so, extension_dir / so.name)
            for subdir in ("_mlir", "csrc", "include"):
                src = installed_catlass / subdir
                if src.is_dir():
                    self._install(src, extension_dir / subdir)
        else:
            # dev mode: symlink/copy _mlir and .so back to source tree
            mlir_src = installed_catlass / "_mlir"
            if mlir_src.is_dir():
                self._install(mlir_src, PROJECT_ROOT / "catlass" / "_mlir")
            so_files = list(installed_catlass.glob("_tla_type_bridge_native*.so"))
            if so_files:
                self._install(so_files[0], PROJECT_ROOT / "catlass" / so_files[0].name)

    def _get_build_directory(
        self,
        *,
        build_type: str,
        generator: str | None,
        extra_cmake_args: list[str],
    ) -> Path:
        explicit_build_dir = os.environ.get("CMAKE_BUILD_DIR")
        if explicit_build_dir:
            path = Path(explicit_build_dir)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path.resolve()

        python_tag = (
            sys.implementation.cache_tag
            or f"py{sys.version_info.major}{sys.version_info.minor}"
        )
        platform_tag = sysconfig.get_platform()

        fingerprint_source = "\0".join(
            [
                os.environ.get("CC", ""),
                os.environ.get("CXX", ""),
                generator or "",
                os.environ.get("CMAKE_TOOLCHAIN_FILE", ""),
                os.environ.get("ASCEND_HOME_PATH", ""),
                os.environ.get("CATLASS_DSL_ASCENDNPU_IR_ROOT", ""),
                os.environ.get("CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR", ""),
                os.environ.get("CATLASS_INCLUDE_DIR", ""),
                *extra_cmake_args,
            ]
        )
        fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[
            :12
        ]

        return (
            PROJECT_ROOT
            / "build"
            / "cmake"
            / python_tag
            / (f"{platform_tag}-{build_type.lower()}-{fingerprint}")
        )


class EditableWheel(editable_wheel):
    """skip build_ext when installing editable wheel."""

    def _run_build_subcommands(self) -> None:
        build = self.get_finalized_command("build")
        for name in build.get_sub_commands():
            if name == "build_ext":
                continue
            self.run_command(name)


setup(
    use_scm_version={
        "write_to": "catlass/_version.py",
        "fallback_version": "0.0.0",
    },
    ext_modules=[
        CMakeExtension(
            "catlass._tla_type_bridge_native",
            sourcedir="csrc/mlir",
            target="tla-compiler CatlassPythonModules",
        ),
    ],
    cmdclass={
        "build_ext": CMakeBuild,
        "editable_wheel": EditableWheel,
    },
    zip_safe=False,
)
