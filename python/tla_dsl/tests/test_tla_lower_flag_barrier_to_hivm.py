import os
import pathlib
import subprocess
import tempfile

import pytest


def _tla_compile_path() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tla_compile = (
        repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    )
    if not tla_compile.exists():
        raise AssertionError("TlaCompile binary not found. Build csrc/mlir first.")
    return tla_compile


def _tla_compile_env() -> dict[str, str]:
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        lib_dir = pathlib.Path(conda_prefix) / "lib"
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)
    return env


def _compile_mlir(mlir_text: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = pathlib.Path(tmpdir) / "input.mlir"
        input_path.write_text(mlir_text)
        return subprocess.run(
            [str(_tla_compile_path()), str(input_path), "--emit=mlir"],
            check=False,
            capture_output=True,
            text=True,
            env=_tla_compile_env(),
        )


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
    if "tla-lower-flag-barrier-to-hivm" not in (pipeline.stdout + pipeline.stderr):
        pytest.skip("TlaCompile pipeline does not include tla-lower-flag-barrier-to-hivm")

    mlir_text = """module {
  tla.func @flags() {
    "tla.vector"() ({
      %ready = tla.flag "ready" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>} -> !tla.flag
      tla.set_flag %ready : !tla.flag
      tla.wait_flag %ready : !tla.flag
      tla.pipe_barrier [#tla.pipe<all>]
    }) : () -> ()
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
    if "tla-lower-flag-barrier-to-hivm" not in (pipeline.stdout + pipeline.stderr):
        pytest.skip("TlaCompile pipeline does not include tla-lower-flag-barrier-to-hivm")

    mlir_text = """module {
  tla.func @flags() {
    "tla.vector"() ({
      %ready = tla.flag "ready" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>} -> !tla.flag
      %done = tla.flag "done" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>} -> !tla.flag
      tla.set_flag %ready : !tla.flag
      tla.wait_flag %ready : !tla.flag
      tla.set_flag %done : !tla.flag
      tla.wait_flag %done : !tla.flag
    }) : () -> ()
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


def _cross_flag_module(body: str, *, core: str = "vector") -> str:
    opposite_core = "cube" if core == "vector" else "vector"
    return f"""module {{
  tla.func @flags() {{
    "tla.{core}"() ({{
{body}
    }}) : () -> ()
    "tla.{opposite_core}"() ({{
    }}) : () -> ()
    tla.return
  }}
}}
"""


def _assert_fully_lowered(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr
    assert "hivm.hir.sync_block_set" not in result.stdout
    assert "hivm.hir.sync_block_wait" not in result.stdout
    assert "tla.cross_flag" not in result.stdout
    assert "tla.cross_core_" not in result.stdout


def test_tla_sync_to_hivm_requires_cross_flag_function_core_type() -> None:
    result = _compile_mlir(
        """module {
  tla.func @flags() {
    "tla.vector"() ({
      %flag = tla.cross_flag "flag" -> !tla.cross_flag<2>
      tla.cross_core_set_flag %flag {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
    }) : () -> ()
    tla.return
  }
}
"""
    )
    assert result.returncode != 0
    assert "AIC/AIV hivm.func_core_type" in result.stderr


@pytest.mark.parametrize(("mode", "config"), [(0, 257), (1, 273)])
def test_tla_sync_to_hivm_lowers_static_device_modes(mode: int, config: int) -> None:
    flag_type = f"!tla.cross_flag<{mode}>"
    result = _compile_mlir(
        _cross_flag_module(
            f"""      %unused = tla.cross_flag "a" -> {flag_type}
      %flag = tla.cross_flag "b" -> {flag_type}
      tla.cross_core_wait_flag %unused {{pipe = #tla.pipe<scalar>}} : {flag_type}
      tla.cross_core_set_flag %flag {{pipe = #tla.pipe<mte3>}} : {flag_type}
      tla.cross_core_wait_flag %flag {{pipe = #tla.pipe<scalar>}} : {flag_type}"""
        )
    )
    _assert_fully_lowered(result)
    assert f"llvm.mlir.constant({config} : i64) : i64" in result.stdout
    assert '"hivm.intr.hivm.SET.CROSS.CORE"' in result.stdout
    assert "<{pipe = 5 : i64}> : (i64) -> ()" in result.stdout
    assert (
        '"hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.IMM"() '
        "<{flag_id = 1 : i64, pipe = 0 : i64}>" in result.stdout
    )


def test_tla_sync_to_hivm_lowers_static_mode2_on_aiv() -> None:
    flag_type = "!tla.cross_flag<2>"
    result = _compile_mlir(
        _cross_flag_module(
            f"""      %flag = tla.cross_flag "flag" -> {flag_type}
      tla.cross_core_set_flag %flag {{pipe = #tla.pipe<fix>}} : {flag_type}
      tla.cross_core_wait_flag %flag {{pipe = #tla.pipe<vector>}} : {flag_type}"""
        )
    )
    _assert_fully_lowered(result)
    assert (
        '"hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() '
        "<{pipe = 10 : i64, sync_id = 0 : i64}>" in result.stdout
    )
    assert (
        '"hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() '
        "<{pipe = 1 : i64, sync_id = 0 : i64}>" in result.stdout
    )


def test_tla_sync_to_hivm_duplicates_static_mode2_on_aic() -> None:
    flag_type = "!tla.cross_flag<2>"
    result = _compile_mlir(
        _cross_flag_module(
            f"""      %flag = tla.cross_flag "flag" -> {flag_type}
      tla.cross_core_set_flag %flag {{pipe = #tla.pipe<mte3>}} : {flag_type}
      tla.cross_core_wait_flag %flag {{pipe = #tla.pipe<mte1>}} : {flag_type}""",
            core="cube",
        )
    )
    _assert_fully_lowered(result)
    for operation, pipe in (("SET", 5), ("WAIT", 3)):
        for sync_id in (0, 16):
            assert (
                f'"hivm.intr.hivm.{operation}.INTRA.BLOCKI.mode"() '
                f"<{{pipe = {pipe} : i64, sync_id = {sync_id} : i64}}>" in result.stdout
            )


@pytest.mark.parametrize("aiv_id", [0, 1])
def test_tla_sync_to_hivm_lowers_static_mode4_on_aic(aiv_id: int) -> None:
    flag_type = "!tla.cross_flag<4>"
    result = _compile_mlir(
        _cross_flag_module(
            f"""      %flag = tla.cross_flag "flag" -> {flag_type}
      tla.cross_core_set_flag %flag {{aiv_id = {aiv_id} : i64, pipe = #tla.pipe<mte3>}} : {flag_type}
      tla.cross_core_wait_flag %flag {{aiv_id = {aiv_id} : i64, pipe = #tla.pipe<mte1>}} : {flag_type}""",
            core="cube",
        )
    )
    _assert_fully_lowered(result)
    sync_id = 16 * aiv_id
    assert f"pipe = 5 : i64, sync_id = {sync_id} : i64" in result.stdout
    assert f"pipe = 3 : i64, sync_id = {sync_id} : i64" in result.stdout


@pytest.mark.parametrize("aiv_id", [0, 1])
def test_tla_sync_to_hivm_guards_static_mode4_on_aiv(aiv_id: int) -> None:
    flag_type = "!tla.cross_flag<4>"
    result = _compile_mlir(
        _cross_flag_module(
            f"""      %flag = tla.cross_flag "flag" -> {flag_type}
      tla.cross_core_set_flag %flag {{aiv_id = {aiv_id} : i64, pipe = #tla.pipe<fix>}} : {flag_type}
      tla.cross_core_wait_flag %flag {{aiv_id = {aiv_id} : i64, pipe = #tla.pipe<vector>}} : {flag_type}"""
        )
    )
    _assert_fully_lowered(result)
    assert "hivm.hir.get_sub_block_idx" in result.stdout
    assert f"llvm.mlir.constant({aiv_id} : i64) : i64" in result.stdout
    assert 'llvm.icmp "eq"' in result.stdout
    assert "cf.cond_br" in result.stdout
    assert "pipe = 10 : i64, sync_id = 0 : i64" in result.stdout
    assert "pipe = 1 : i64, sync_id = 0 : i64" in result.stdout


@pytest.mark.parametrize("mode", [0, 1, 2, 4])
def test_tla_sync_to_hivm_lowers_dynamic_id_to_register_form(mode: int) -> None:
    flag_type = f"!tla.cross_flag<{mode}>"
    aiv_attr = ", aiv_id = 0 : i64" if mode == 4 else ""
    result = _compile_mlir(
        _cross_flag_module(
            f"""      %condition = arith.constant true
      %a = tla.cross_flag "a" -> {flag_type}
      %b = tla.cross_flag "b" -> {flag_type}
      %selected = scf.if %condition -> ({flag_type}) {{
        scf.yield %a : {flag_type}
      }} else {{
        scf.yield %b : {flag_type}
      }}
      tla.cross_core_set_flag %selected {{pipe = #tla.pipe<fix>{aiv_attr}}} : {flag_type}
      tla.cross_core_wait_flag %selected {{pipe = #tla.pipe<vector>{aiv_attr}}} : {flag_type}"""
        )
    )
    _assert_fully_lowered(result)
    if mode in (2, 4):
        assert '"hivm.intr.hivm.SET.INTRA.BLOCK.mode"(' in result.stdout
        assert '"hivm.intr.hivm.WAIT.INTRA.BLOCK.mode"(' in result.stdout
        assert "INTRA.BLOCKI.mode" not in result.stdout
    else:
        assert '"hivm.intr.hivm.SET.CROSS.CORE"(' in result.stdout
        assert '"hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.REG"(' in result.stdout
        assert "WAIT.FLAG.DEV.PIPE.IMM" not in result.stdout
        assert "llvm.shl" in result.stdout
        assert "llvm.or" in result.stdout


def test_tla_sync_to_hivm_duplicates_dynamic_mode2_on_aic() -> None:
    flag_type = "!tla.cross_flag<2>"
    result = _compile_mlir(
        _cross_flag_module(
            f"""      %condition = arith.constant true
      %a = tla.cross_flag "a" -> {flag_type}
      %b = tla.cross_flag "b" -> {flag_type}
      %selected = scf.if %condition -> ({flag_type}) {{
        scf.yield %a : {flag_type}
      }} else {{
        scf.yield %b : {flag_type}
      }}
      tla.cross_core_set_flag %selected {{pipe = #tla.pipe<fix>}} : {flag_type}""",
            core="cube",
        )
    )
    _assert_fully_lowered(result)
    assert result.stdout.count('"hivm.intr.hivm.SET.INTRA.BLOCK.mode"(') == 2
    assert "llvm.add" in result.stdout
    assert "llvm.mlir.constant(16 : i64)" in result.stdout


def test_tla_sync_to_hivm_offsets_dynamic_mode4_aiv1_on_aic() -> None:
    flag_type = "!tla.cross_flag<4>"
    result = _compile_mlir(
        _cross_flag_module(
            f"""      %condition = arith.constant true
      %a = tla.cross_flag "a" -> {flag_type}
      %b = tla.cross_flag "b" -> {flag_type}
      %selected = scf.if %condition -> ({flag_type}) {{
        scf.yield %a : {flag_type}
      }} else {{
        scf.yield %b : {flag_type}
      }}
      tla.cross_core_set_flag %selected {{aiv_id = 1 : i64, pipe = #tla.pipe<fix>}} : {flag_type}""",
            core="cube",
        )
    )
    _assert_fully_lowered(result)
    assert result.stdout.count('"hivm.intr.hivm.SET.INTRA.BLOCK.mode"(') == 1
    assert "llvm.add" in result.stdout
    assert "llvm.mlir.constant(16 : i64)" in result.stdout


@pytest.mark.parametrize(
    ("flag_type", "attrs", "message"),
    [
        ("!tla.cross_flag<4>", "pipe = #tla.pipe<fix>", "mode 4 requires aiv_id"),
        (
            "!tla.cross_flag<2>",
            "aiv_id = 0 : i64, pipe = #tla.pipe<fix>",
            "aiv_id is only valid for mode 4",
        ),
    ],
)
def test_cross_core_flag_verifier_rejects_invalid_aiv_id_ir(
    flag_type: str, attrs: str, message: str
) -> None:
    result = _compile_mlir(
        _cross_flag_module(
            f"""      %flag = tla.cross_flag "flag" -> {flag_type}
      tla.cross_core_set_flag %flag {{{attrs}}} : {flag_type}"""
        )
    )
    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize(
    "body",
    [
        """      %flag = tla.cross_flag "flag" -> !tla.cross_flag<2>
      tla.cross_core_set_flag %flag {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>""",
        """      %flag = tla.cross_flag "flag" -> !tla.cross_flag<2>
      tla.cross_core_wait_flag %flag {pipe = #tla.pipe<vector>} : !tla.cross_flag<2>""",
        """      %flag = tla.cross_flag "flag" -> !tla.cross_flag<2>
      tla.cross_core_set_flag %flag {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %flag {pipe = #tla.pipe<mte3>} : !tla.cross_flag<2>
      tla.cross_core_wait_flag %flag {pipe = #tla.pipe<vector>} : !tla.cross_flag<2>""",
    ],
)
def test_tla_sync_to_hivm_accepts_unmatched_and_repeated_cross_ops(body: str) -> None:
    result = _compile_mlir(_cross_flag_module(body))
    _assert_fully_lowered(result)


def test_tla_sync_to_hivm_emits_repeated_sets_from_distinct_pipes() -> None:
    flag_type = "!tla.cross_flag<2>"
    result = _compile_mlir(
        _cross_flag_module(
            f"""      %flag = tla.cross_flag "flag" -> {flag_type}
      tla.cross_core_set_flag %flag {{pipe = #tla.pipe<fix>}} : {flag_type}
      tla.cross_core_set_flag %flag {{pipe = #tla.pipe<mte3>}} : {flag_type}"""
        )
    )
    _assert_fully_lowered(result)
    assert result.stdout.count('"hivm.intr.hivm.SET.INTRA.BLOCKI.mode"()') == 2
    assert "pipe = 10 : i64, sync_id = 0 : i64" in result.stdout
    assert "pipe = 5 : i64, sync_id = 0 : i64" in result.stdout


def test_tla_sync_to_hivm_rejects_cross_mode_dynamic_selection() -> None:
    result = _compile_mlir(
        _cross_flag_module(
            """      %condition = arith.constant true
      %a = tla.cross_flag "a" -> !tla.cross_flag<2>
      %b = tla.cross_flag "b" -> !tla.cross_flag<4>
      %selected = scf.if %condition -> (!tla.cross_flag<2>) {
        scf.yield %a : !tla.cross_flag<2>
      } else {
        scf.yield %b : !tla.cross_flag<4>
      }
      tla.cross_core_set_flag %selected {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>"""
        )
    )
    assert result.returncode != 0
    assert "should match input type" in result.stderr


def _cross_flag_declarations(count: int) -> str:
    return "\n".join(
        f"""      %f{i} = tla.cross_flag "flag_{i:02d}" -> !tla.cross_flag<2>
      tla.cross_core_set_flag %f{i} {{pipe = #tla.pipe<fix>}} : !tla.cross_flag<2>"""
        for i in range(count)
    )


def test_tla_sync_to_hivm_accepts_full_cross_flag_id_range() -> None:
    result = _compile_mlir(_cross_flag_module(_cross_flag_declarations(16)))
    _assert_fully_lowered(result)
    assert result.stdout.count('"hivm.intr.hivm.SET.INTRA.BLOCKI.mode"()') == 16
    assert "pipe = 10 : i64, sync_id = 15 : i64" in result.stdout


def test_tla_sync_to_hivm_rejects_cross_flag_id_exhaustion() -> None:
    result = _compile_mlir(_cross_flag_module(_cross_flag_declarations(17)))
    assert result.returncode != 0
    assert "cross flag id exhausted (legal range 0-15; maximum 16 flags)" in result.stderr
