import os
import pathlib
import subprocess
import tempfile

import pytest


def _tla_compile_path() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tla_compile = repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    if not tla_compile.exists():
        pytest.skip("TlaCompile binary not found. Build csrc/mlir first.")
    return tla_compile


def _tla_compile_env() -> dict[str, str]:
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        lib_dir = pathlib.Path(conda_prefix) / "lib"
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)
    return env


def test_tla_sync_to_hivm_lowers_pipe_sync_to_hivm_ops() -> None:
    tla_compile = _tla_compile_path()
    env = _tla_compile_env()
    pipeline = subprocess.run(
        [str(tla_compile), "--print-pipeline=mlir", "/dev/null"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if "tla-sync-to-hivm" not in (pipeline.stdout + pipeline.stderr):
        pytest.skip("TlaCompile pipeline does not include tla-sync-to-hivm")

    mlir_text = """module {
  tla.func @flags() {
    %ready = tla.flag "ready" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>} -> !tla.flag
    tla.set_flag %ready : !tla.flag
    tla.wait_flag %ready : !tla.flag
    tla.pipe_barrier [#tla.pipe<all>]
    tla.return
  }
}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = pathlib.Path(tmpdir) / "input.mlir"
        input_path.write_text(mlir_text)
        result = subprocess.run(
            [str(tla_compile), str(input_path), "--emit=mlir"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    assert result.returncode == 0, result.stderr
    assert "hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]" in result.stdout
    assert "hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]" in result.stdout
    assert "hivm.hir.pipe_barrier[<PIPE_ALL>]" in result.stdout


def test_tla_sync_to_hivm_allocates_distinct_event_ids_for_same_pipe_pair() -> None:
    tla_compile = _tla_compile_path()
    env = _tla_compile_env()
    pipeline = subprocess.run(
        [str(tla_compile), "--print-pipeline=mlir", "/dev/null"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if "tla-sync-to-hivm" not in (pipeline.stdout + pipeline.stderr):
        pytest.skip("TlaCompile pipeline does not include tla-sync-to-hivm")

    mlir_text = """module {
  tla.func @flags() {
    %ready = tla.flag "ready" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>} -> !tla.flag
    %done = tla.flag "done" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>} -> !tla.flag
    tla.set_flag %ready : !tla.flag
    tla.wait_flag %ready : !tla.flag
    tla.set_flag %done : !tla.flag
    tla.wait_flag %done : !tla.flag
    tla.return
  }
}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = pathlib.Path(tmpdir) / "input.mlir"
        input_path.write_text(mlir_text)
        result = subprocess.run(
            [str(tla_compile), str(input_path), "--emit=mlir"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    assert result.returncode == 0, result.stderr
    assert (
        result.stdout.count("hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]")
        == 1
    )
    assert (
        result.stdout.count("hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]")
        == 1
    )
    assert (
        result.stdout.count("hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]")
        == 1
    )
    assert (
        result.stdout.count("hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]")
        == 1
    )
