import pathlib
import subprocess

import pytest

import catlass as tla
import catlass.runtime as runtime_mod


def _require_hivm_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tla_compile = repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    if not tla_compile.exists():
        raise AssertionError("TlaCompile binary not found. Build csrc/mlir first.")
    return tla_compile


@tla.kernel
def copy_gm_to_cbuf_kernel(mem_in: tla.Tensor) -> None:
    tile = tla.tile_view(mem_in, tla.make_shape(32, 32), tla.make_coord(1, 1))
    allocator = tla.utils.LocalmemAllocator()
    ptr = allocator.allocate(32 * 32 * 4, 512, tla.AddressSpace.l1)
    ptr = tla.recast_ptr(ptr, dtype=tla.Float32)
    local = tla.make_tensor_like(ptr, tile, tla.arch.zN)
    with tla.cube():
        tla.copy(local, tile)


def test_frontend_copy_gm_to_cbuf_lowers_to_runtime_call(tmp_path) -> None:
    tla_compile = _require_hivm_tla_compile()
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(128, 128),
            tla.Float32,
            origin_shape=tla.make_shape(128, 128),
        )

    mlir = copy_gm_to_cbuf_kernel.dump_mlir(type_args=(mem,))
    assert "!tla.layout<!tla.shape<(16,2),(8,4)>" in mlir
    assert "!tla.ptr<f32, l1, 512>" in mlir
    assert "tla.copy" in mlir

    input_path = tmp_path / "copy_gm_to_cbuf.mlir"
    input_path.write_text(mlir)
    try:
        result = subprocess.run(
            [str(tla_compile), str(input_path), "-o", "-"],
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if "missing descriptor for tla.tile_view result" in exc.stderr:
            pytest.skip(
                "tla.tile_view descriptor lowering is not available in this build"
            )
        raise

    lowered = result.stdout
    assert "copy_gm_row_major_to_cbuf_zN_float" in lowered
    assert "_mlir_ciface_copy_gm_row_major_to_cbuf_zN_float" not in lowered
    assert "hacc.always_inline" in lowered
    assert "hivm.func_core_type = #hivm.func_core_type<AIC>" in lowered
    assert "llvm.emit_c_interface" in lowered
    assert "memref.subview" not in lowered
    assert "memref.cast %arg0 : memref<128x128xf32" in lowered
    assert lowered.count("hivm.hir.pointer_cast") == 1
    assert '"tla.copy"' not in lowered
    assert '"tla.alloc_ptr"' not in lowered
    assert '"tla.recast_ptr"' not in lowered
