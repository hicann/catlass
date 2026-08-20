from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repeated_editable_install_isolates_python_from_project_pythonpath(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "tla_dsl"
    project_root.mkdir()
    build_script = project_root / "build.sh"
    shutil.copy2(PROJECT_ROOT / "build.sh", build_script)

    npu_ir_root = tmp_path / "AscendNPU-IR"
    mlir_core = npu_ir_root / "build" / "install" / "python_packages" / "mlir_core"
    mlir_core.mkdir(parents=True)
    retained_path = tmp_path / "retained-pythonpath"
    retained_path.mkdir()

    log_path = tmp_path / "python-invocations.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\t%s\\t%s\\n\' "$PWD" "$*" "${PYTHONPATH-}" '
        '>> "$BUILD_TEST_LOG"\n'
        'if [[ "$*" == "setup.py build_ext --inplace" ]]; then\n'
        "  mkdir -p ascend_catlass_dsl.egg-info\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "ASCEND_HOME_PATH": str(tmp_path / "ascend-toolkit"),
            "BUILD_TEST_LOG": str(log_path),
            "CATLASS_DSL_PREBUILT_ASCENDNPU_IR": str(npu_ir_root),
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "PYTHONPATH": os.pathsep.join([str(project_root), str(retained_path)]),
        }
    )

    subprocess.run([build_script], check=True, env=env)
    subprocess.run([build_script], check=True, env=env)

    invocations = [
        line.split("\t", maxsplit=2) 
        for line in log_path
        .read_text()
        .splitlines()
    ]
    pip_invocations = [entry for entry in invocations if "-m pip " in entry[1]]
    assert pip_invocations == [
        [
            str(project_root.parent),
            f"-I -m pip install -e {project_root} --no-deps",
            os.pathsep.join([str(mlir_core), str(project_root), str(retained_path)]),
        ],
        [
            str(project_root.parent),
            f"-I -m pip install -e {project_root} --no-deps",
            os.pathsep.join([str(mlir_core), str(project_root), str(retained_path)]),
        ],
    ]
