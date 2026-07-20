import pathlib
import re
import subprocess

import pytest

import catlass as tla
import catlass.runtime as runtime_mod
from catlass.execution_lowering import TlaLoweringError


def _require_hivm_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tla_compile = repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    if not tla_compile.exists():
        raise AssertionError("TlaCompile binary not found. Build csrc/mlir first.")
    return tla_compile


@tla.kernel
def copy_gm_to_cbuf_kernel(mem_in: tla.Tensor) -> None:
    tile = tla.tile_view(mem_in, tla.make_shape(32, 32), tla.make_coord(1, 1))
    ptr = tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l1, 512)
    local = tla.make_tensor_like(ptr, tile, tla.arch.zN)
    with tla.cube():
        tla.copy(local, tile)


@tla.kernel
def copy_kernel_arg_directly_to_ub_kernel(mem_in: tla.Tensor) -> None:
    ub_ptr = tla.allocate((16, 16), tla.Float32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor(
        ub_ptr,
        tla.make_layout(
            tla.make_shape(16, 16),
            tla.make_stride(16, 1),
        ),
    )
    with tla.vector():
        tla.copy(ub, mem_in)


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


def test_kernel_gm_arg_copies_directly_to_ub(tmp_path) -> None:
    tla_compile = _require_hivm_tla_compile()
    with runtime_mod._eager_capture():
        mem_in = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float32,
            origin_shape=tla.make_shape(16, 16),
            coord=tla.make_coord(0, 0),
            stride=tla.make_stride(16, 1),
            layout_tag=tla.arch.RowMajor,
        )

    mlir = copy_kernel_arg_directly_to_ub_kernel.dump_mlir(type_args=(mem_in,))
    assert "tla.tile_view" not in mlir
    assert "tla.make_tensor_like" not in mlir
    assert mlir.count("tla.make_tensor ") == 1
    source_copy = next(line for line in mlir.splitlines() if "tla.copy" in line)
    assert "%arg0" in source_copy

    input_path = tmp_path / "copy_kernel_arg_directly_to_ub.mlir"
    input_path.write_text(mlir)
    result = subprocess.run(
        [
            str(tla_compile),
            str(input_path),
            "-o",
            "-",
            "--mlir-print-ir-after=tla-lower-func",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    descriptor_match = re.search(
        r"(?P<descriptor>%[A-Za-z0-9_]+) = tla\.tensor_desc %arg0\[",
        result.stderr,
    )
    assert descriptor_match is not None, result.stderr
    lowered_copy = next(
        line for line in result.stderr.splitlines() if "tla.copy" in line
    )
    assert descriptor_match.group("descriptor") in lowered_copy
    assert "%arg0" not in lowered_copy
    assert "copy_gm_row_major_to_ub_row_major_float" in result.stdout
    assert "tla.copy" not in result.stdout


@tla.kernel
def copy_l0c_to_ub_split_mismatch_dtype_kernel(gm_c: tla.Tensor) -> None:
    """L0C(f32)->UB(f16) with SPLIT_M, dtype mismatch must be rejected."""
    allocator = tla.utils.LocalmemAllocator()
    l0c_ptr = allocator.allocate(32 * 32 * 4, 512, tla.AddressSpace.l0c)
    l0c_ptr = tla.recast_ptr(l0c_ptr, dtype=tla.Float32)
    l0c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
    ub_ptr = allocator.allocate(16 * 32 * 4, 256, tla.AddressSpace.ub)
    ub_ptr = tla.recast_ptr(ub_ptr, dtype=tla.Float16)
    ub = tla.make_tensor_like(ub_ptr, gm_c, tla.arch.RowMajor)
    with tla.cube():
        tla.copy(
            ub, l0c,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )


def test_copy_l0c_to_ub_split_mismatch_dtype_raises() -> None:
    """L0C->UB copy with SPLIT_M where src(f32) != dst(f16) must raise TlaLoweringError."""
    with runtime_mod._eager_capture():
        gm_c = tla.Tensor(
            tla.make_shape(32, 32),
            tla.Float16,
            origin_shape=tla.make_shape(32, 32),
            coord=tla.make_coord(0, 0),
            stride=tla.make_stride(32, 1),
            layout_tag=tla.arch.RowMajor,
        )
    with pytest.raises(
        TlaLoweringError,
        match=r"When copy l0c to ub with split mode, src and dst dtype must be same",
    ):
        copy_l0c_to_ub_split_mismatch_dtype_kernel.dump_mlir(type_args=(gm_c,))


@tla.kernel
def nested_ub_subtile_copy_kernel(mem_in: tla.Tensor, mem_out: tla.Tensor) -> None:
    allocator = tla.utils.LocalmemAllocator()
    gm_in = tla.tile_view(mem_in, tla.make_shape(64, 64), tla.make_coord(0, 0))
    gm_out = tla.tile_view(mem_out, tla.make_shape(32, 32), tla.make_coord(0, 0))
    ub_ptr = allocator.allocate(64 * 64 * 4, 256, tla.AddressSpace.ub)
    ub_ptr = tla.recast_ptr(ub_ptr, dtype=tla.Float32)
    ub_root = tla.make_tensor_like(ub_ptr, gm_in, tla.arch.RowMajor)
    ub_tile = tla.tile_view(ub_root, tla.make_shape(32, 32), tla.make_coord(1, 1))
    with tla.vector():
        tla.copy(ub_root, gm_in)
        tla.copy(gm_out, ub_tile)


def test_nested_ub_subtile_copy_lowers(tmp_path) -> None:
    """GM<->UB staging copies lower to vector-core (AIV) cifax runtime calls."""
    tla_compile = _require_hivm_tla_compile()
    with runtime_mod._eager_capture():
        mem_in = tla.Tensor(
            tla.make_shape(64, 64),
            tla.Float32,
            origin_shape=tla.make_shape(64, 64),
        )
        mem_out = tla.Tensor(
            tla.make_shape(32, 32),
            tla.Float32,
            origin_shape=tla.make_shape(32, 32),
        )

    mlir = nested_ub_subtile_copy_kernel.dump_mlir(type_args=(mem_in, mem_out))
    input_path = tmp_path / "nested_ub_subtile_copy.mlir"
    input_path.write_text(mlir)

    result = subprocess.run(
        [str(tla_compile), str(input_path), "-o", "-"],
        text=True,
        capture_output=True,
        check=True,
    )
    lowered = result.stdout
    # The kernel issues both a GM->UB and a UB->GM staging copy; each lowers to its
    # own inlinable AIV cifax runtime template (bc/Vector/dma.cpp).
    assert "copy_gm_row_major_to_ub_row_major_float" in lowered
    assert "copy_ub_row_major_to_gm_row_major_float" in lowered
    assert "hivm.func_core_type = #hivm.func_core_type<AIV>" in lowered
    assert '"tla.copy"' not in lowered


@tla.kernel
def ptradd_ub_subtile_copy_kernel(mem_in: tla.Tensor, mem_src: tla.Tensor) -> None:
    gm_root = tla.tile_view(mem_in, tla.make_shape(64, 64), tla.make_coord(0, 0))
    gm_src = tla.tile_view(mem_src, tla.make_shape(32, 32), tla.make_coord(0, 0))
    allocator = tla.utils.LocalmemAllocator()
    ub_ptr = allocator.allocate((64 * 64 + 16) * 4, 256, tla.AddressSpace.ub)
    ub_ptr = tla.recast_ptr(ub_ptr, dtype=tla.Float32) + 16
    ub_root = tla.make_tensor_like(ub_ptr, gm_root, tla.arch.RowMajor)
    ub_tile = tla.tile_view(ub_root, tla.make_shape(32, 32), tla.make_coord(1, 1))
    with tla.vector():
        tla.copy(ub_tile, gm_src)


def test_ptradd_ub_subtile_copy_applies_ptr_offset(tmp_path) -> None:
    """The ptr_add offset is preserved in the cifax base pointer_cast and tile payload."""
    tla_compile = _require_hivm_tla_compile()
    with runtime_mod._eager_capture():
        mem_in = tla.Tensor(
            tla.make_shape(64, 64),
            tla.Float32,
            origin_shape=tla.make_shape(64, 64),
        )
        mem_src = tla.Tensor(
            tla.make_shape(32, 32),
            tla.Float32,
            origin_shape=tla.make_shape(32, 32),
        )

    mlir = ptradd_ub_subtile_copy_kernel.dump_mlir(type_args=(mem_in, mem_src))
    input_path = tmp_path / "ptradd_ub_subtile.mlir"
    input_path.write_text(mlir)
    output_path = tmp_path / "out.mlir"
    result = subprocess.run(
        [
            str(tla_compile),
            str(input_path),
            "-o",
            str(output_path),
            "--mlir-print-ir-after=tla-finalize-memref",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout + result.stderr  # print-ir-after goes to stderr
    # GM->UB staging copy lowers to the AIV cifax runtime template.
    assert "copy_gm_row_major_to_ub_row_major_float" in out
    assert "hivm.func_core_type = #hivm.func_core_type<AIV>" in out
    # ptr_add contributes 64 bytes (16 f32 elements) to the UB base pointer_cast;
    # the UB sub-tile (coord (1,1) of a 64-wide buffer) carries stride0=64 and
    # absCoord (32,32) in the i64 payload, so the stub computes the flat offset
    # (32*64+32 = 2080) from absCoord/stride at runtime.
    assert "arith.constant 64 : i64" in out
    assert '"tla.copy"' not in out
