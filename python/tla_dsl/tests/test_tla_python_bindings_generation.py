from __future__ import annotations

import pathlib
import subprocess
import sys


def test_generated_tla_python_bindings_are_current() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "generate_tla_python_bindings.py"
    assert script.exists()
    subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=repo_root)
