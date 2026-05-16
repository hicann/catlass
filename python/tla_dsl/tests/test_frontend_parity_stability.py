from __future__ import annotations

from pathlib import Path

import catlass as tla
import catlass.execution as execution
import catlass.runtime as runtime_mod


@tla.kernel
def fake_kernel() -> None:
    x = tla.make_shape(100, 200)
    _ = x


def test_full_demo_style_kernel_lowering_emits_mlir(compiler_tlair) -> None:
    mlir = compiler_tlair(fake_kernel)
    assert 'sym_name = "fake_kernel"' in mlir
    assert '"tla.make_shape"' in mlir
    assert '"tla.return"' in mlir


def test_full_demo_style_compile_routes_kernel(monkeypatch) -> None:
    artifact = execution.TlaKernelArtifact(
        cache_key="parity",
        cache_dir=Path("/tmp/cache"),
        tlair_mlir="module {}",
        lowered_llvm="; llvm",
        entrypoint="fake_kernel",
        compiler_bridge_path=Path("/tmp/_tla_type_bridge_native.so"),
        hivmc_path=Path("/tmp/hivmc-a5"),
        kernel_binary_path=Path("/tmp/kernel.o"),
    )
    calls: list[tuple[str, str]] = []

    def fake_compile(fn, *, kind, options, runtime, type_args=None):
        del options, runtime, type_args
        calls.append((fn.__name__, kind))
        return artifact

    monkeypatch.setattr(runtime_mod, "compile_kernel", fake_compile)
    result = tla.compile(fake_kernel, cache=False)

    assert result == artifact
    assert calls == [("fake_kernel", "kernel")]
