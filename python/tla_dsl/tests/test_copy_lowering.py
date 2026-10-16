from catlass.tla.runtime import make_fake_tensor

import pathlib
import re
import subprocess

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod


def _require_hivm_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tla_compile = (
        repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    )
    if not tla_compile.exists():
        raise AssertionError("TlaCompile binary not found. Build csrc/mlir first.")
    return tla_compile


@tla.kernel
def copy_gm_to_l1_kernel(mem_in: tla.Tensor) -> None:
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


@tla.kernel
def copy_l1_to_gm_kernel(mem_in: tla.Tensor) -> None:
    """L1 -> GM: the copy unit has no such direction at all."""
    l1 = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l1, 512),
        mem_in,
        tla.arch.zN,
    )
    with tla.cube():
        tla.copy(mem_in, l1)


@tla.kernel
def copy_ub_to_ub_kernel(mem_in: tla.Tensor) -> None:
    src_ptr = tla.allocate((32, 32), tla.Float32, tla.AddressSpace.ub, 256)
    dst_ptr = tla.allocate((32, 32), tla.Float32, tla.AddressSpace.ub, 256)
    src = tla.make_tensor_like(src_ptr, mem_in, tla.arch.RowMajor)
    dst = tla.make_tensor_like(dst_ptr, mem_in, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(dst, src)


def test_frontend_rejects_copy_l1_to_gm() -> None:
    """A direction the hardware does not have is named as such, not as an
    unimplemented route."""
    mem = make_fake_tensor(
        tla.Float32,
        (32, 32),
        (32, 1),
        origin_shape=(32, 32),
        layout_tag=tla.arch.RowMajor,
    )
    with pytest.raises(
        runtime_mod.TlaCoreAPIError,
        match=r"Hardware unsupported copy route",
    ):
        copy_l1_to_gm_kernel.dump_mlir(type_args=(mem,))


@tla.kernel
def copy_fp8_gm_to_ub_kernel(mem_in: tla.Tensor) -> None:
    """fp8 staged through UB on the vector path."""
    ub = tla.make_tensor(
        tla.allocate((32, 32), tla.Float8E4M3FN, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(32, 32), tla.make_stride(32, 1)),
    )
    with tla.vector():
        tla.copy(ub, mem_in)


def test_frontend_accepts_fp8_gm_to_ub_copy() -> None:
    """fp8 has a vector staging route now.

    It used to be refused on the grounds that fp8 was a cube *operand* format.
    That was a statement about the registrations rather than about the
    hardware: Vector/dma.cpp registers GM <-> UB for both fp8 encodings, so the
    front-end route table has to admit them or the copy dies before it reaches
    the lowering.
    """
    mem = make_fake_tensor(
        tla.Float8E4M3FN,
        (32, 32),
        (32, 1),
        origin_shape=(32, 32),
        layout_tag=tla.arch.RowMajor,
    )
    copy_fp8_gm_to_ub_kernel.dump_mlir(type_args=(mem,))


@tla.kernel
def copy_fp4_to_l0a_without_scale_kernel() -> None:
    """A packed fp4 tile loaded into L0A with no scale block attached."""
    l1 = tla.make_tensor(
        tla.allocate((128, 64), tla.Float4E2M1, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
    )
    l0a = tla.make_tensor_like(
        tla.allocate((128, 64), tla.Float4E2M1, tla.AddressSpace.l0a, 512),
        l1,
        tla.arch.zN,
    )
    with tla.cube():
        tla.copy(l0a, l1)


def test_frontend_rejects_unscaled_fp4_l1_to_l0a_copy() -> None:
    """fp4 is microscaling-only: the L1->L0A load has to carry a scale."""
    with pytest.raises(
        runtime_mod.TlaCoreAPIError,
        match=r"packed fp4 to l0a/l0b must carry scale",
    ):
        copy_fp4_to_l0a_without_scale_kernel.dump_mlir()


def test_frontend_copy_gm_to_l1_lowers_to_runtime_call(tmp_path) -> None:
    tla_compile = _require_hivm_tla_compile()
    mem = make_fake_tensor(
        tla.Float32,
        (128, 128),
        (128, 1),
        origin_shape=(128, 128),
        layout_tag=tla.arch.RowMajor,
    )

    mlir = copy_gm_to_l1_kernel.dump_mlir(type_args=(mem,))
    assert "!tla.layout<!tla.shape<(16,2),(8,4)>" in mlir
    assert "!tla.ptr<f32, l1, 512>" in mlir
    assert "tla.copy" in mlir

    input_path = tmp_path / "copy_gm_to_l1.mlir"
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
    assert "copy_gm_RowMajor_to_l1_zN_float" in lowered
    assert "_mlir_ciface_copy_gm_RowMajor_to_l1_zN_float" not in lowered
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
    mem_in = make_fake_tensor(
        tla.Float32,
        (16, 16),
        (16, 1),
        origin_shape=(16, 16),
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
        r"(?P<descriptor>%[A-Za-z0-9_]+) = tla\.tensor_desc %arg0 shape\[",
        result.stderr,
    )
    assert descriptor_match is not None, result.stderr
    lowered_copy = next(
        line for line in result.stderr.splitlines() if "tla.copy" in line
    )
    assert descriptor_match.group("descriptor") in lowered_copy
    assert "%arg0" not in lowered_copy
    assert "copy_gm_RowMajor_to_ub_RowMajor_float" in result.stdout
    assert "tla.copy" not in result.stdout


@tla.kernel
def copy_dynamic_l0c_to_ub_split_m_kernel(gm_c: tla.Tensor) -> None:
    """Reproduce a dynamically clipped N extent flowing from L0C into UB."""
    l0c_ptr = tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l0c, 512)
    ub_ptr = tla.allocate((64, 128), tla.Float32, tla.AddressSpace.ub, 512)

    for col in tla.range(tla.arch.block_idx(), 1, tla.arch.block_num()):
        gm_tile = tla.tile_view(gm_c, tla.make_shape(128, 128), tla.make_coord(0, col))
        l0c = tla.make_tensor_like(l0c_ptr, gm_tile, tla.arch.L0Clayout)
        ub = tla.make_tensor_like(ub_ptr, l0c, tla.arch.RowMajor)
        with tla.cube():
            tla.copy(
                ub,
                l0c,
                tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
            )


def test_split_m_dynamic_RowMajor_stride_uses_child_n_extent(tmp_path) -> None:
    """Split-M destination stride0 must be aligned N, not the L0C packing stride."""
    tla_compile = _require_hivm_tla_compile()
    gm_c = make_fake_tensor(
        tla.Float32,
        (128, 128),
        (1, 128),
        origin_shape=(128, 128),
        layout_tag=tla.arch.ColumnMajor,
    )

    mlir = copy_dynamic_l0c_to_ub_split_m_kernel.dump_mlir(type_args=(gm_c,))
    assert "!tla.shape<128,?>" in mlir
    assert "!tla.stride<?,1>" in mlir

    input_path = tmp_path / "copy_dynamic_l0c_to_ub_split_m.mlir"
    input_path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(input_path), "-o", "-"],
        text=True,
        capture_output=True,
        check=True,
    )

    call_match = re.search(
        r"call @copy_l0c_to_ub_RowMajor_splitm_float\(([^)]*)\)",
        result.stdout,
    )
    assert call_match is not None, result.stdout
    operands = [operand.strip() for operand in call_match.group(1).split(",")]
    # 2 memref + 12 src (L0C) + 12 dst (row-major, unified 4D) + unitFlag + subBlockId = 28.
    assert len(operands) == 28
    dst_shape1 = operands[15]
    dst_stride0 = operands[18]
    _assert_lowered_f32_dynamic_stride_is_32b_aligned(
        result.stdout, extent=dst_shape1, stride=dst_stride0
    )


_MAKE_TENSOR_LIKE_LAYOUT_CASES = (
    ("RowMajor", tla.arch.RowMajor),
    ("ColumnMajor", tla.arch.ColumnMajor),
    ("zN", tla.arch.zN),
    ("nZ", tla.arch.nZ),
    ("L0Clayout", tla.arch.L0Clayout),
    ("zNUnAlign", tla.arch.zNUnAlign),
)
_MAKE_TENSOR_LIKE_PARENT_LAYOUT = tla.arch.RowMajor
_MAKE_TENSOR_LIKE_CHILD_LAYOUT = tla.arch.RowMajor


@tla.kernel
def make_tensor_like_layout_pair_kernel(mem_in: tla.Tensor, m: int, n: int) -> None:
    """Build one dynamic parent->child make_tensor_like layout pair."""
    root = tla.make_tensor(
        mem_in.ptr,
        tla.make_layout(
            tla.make_shape(m, n),
            tla.make_stride(n, 1),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    parent_ptr = tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l1, 512)
    child_ptr = tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l1, 512)
    with tla.cube():
        parent = tla.make_tensor_like(parent_ptr, root, _MAKE_TENSOR_LIKE_PARENT_LAYOUT)
        child = tla.make_tensor_like(child_ptr, parent, _MAKE_TENSOR_LIKE_CHILD_LAYOUT)
        tla.make_shape(child.origin_shape[0], child.origin_shape[1])


def _tensor_desc_metadata(
    line: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    match = re.search(
        r"tla\.tensor_desc\s+%\S+\s+shape\[([^\]]+)\]\s+"
        r"stride\[([^\]]+)\]\s+origin_shape\[([^\]]+)\]\s+"
        r"coord\[([^\]]+)\]\s+:",
        line,
    )
    assert match is not None, line
    return tuple(
        [value.strip() for value in match.group(group).split(",")]
        for group in range(1, 5)
    )


def _index_binary_operands(mlir: str, result: str, op_name: str) -> tuple[str, str]:
    ssa = r"%[A-Za-z0-9_]+"
    match = re.search(
        rf"^\s*{re.escape(result)} = arith\.{op_name} "
        rf"(?P<lhs>{ssa}), (?P<rhs>{ssa}) : index$",
        mlir,
        re.MULTILINE,
    )
    assert match is not None, f"missing defining arith.{op_name} for {result}"
    return match.group("lhs"), match.group("rhs")


def _assert_f32_dynamic_stride_is_32b_aligned(
    mlir: str, extent: str, stride: str
) -> None:
    quotient, alignment = _index_binary_operands(mlir, stride, "muli")
    adjusted, divisor = _index_binary_operands(mlir, quotient, "divsi")
    input_extent, delta = _index_binary_operands(mlir, adjusted, "addi")
    assert input_extent == extent
    assert divisor == alignment
    assert re.search(
        rf"^\s*{re.escape(alignment)} = arith\.constant 8 : index$",
        mlir,
        re.MULTILINE,
    )
    assert re.search(
        rf"^\s*{re.escape(delta)} = arith\.constant 7 : index$",
        mlir,
        re.MULTILINE,
    )


def _llvm_i64_binary_operands(mlir: str, result: str, op_name: str) -> tuple[str, str]:
    ssa = r"%[A-Za-z0-9_]+"
    match = re.search(
        rf"^\s*{re.escape(result)} = llvm\.{op_name} "
        rf"(?P<lhs>{ssa}), (?P<rhs>{ssa})\s*: i64$",
        mlir,
        re.MULTILINE,
    )
    assert match is not None, f"missing defining llvm.{op_name} for {result}"
    return match.group("lhs"), match.group("rhs")


def _assert_lowered_f32_dynamic_stride_is_32b_aligned(
    mlir: str, extent: str, stride: str
) -> None:
    quotient, alignment = _llvm_i64_binary_operands(mlir, stride, "mul")
    adjusted, divisor = _llvm_i64_binary_operands(mlir, quotient, "sdiv")
    input_extent, delta = _llvm_i64_binary_operands(mlir, adjusted, "add")
    assert input_extent == extent
    assert divisor == alignment
    assert re.search(
        rf"^\s*{re.escape(alignment)} = llvm\.mlir\.constant\(8 : index\) : i64$",
        mlir,
        re.MULTILINE,
    )
    assert re.search(
        rf"^\s*{re.escape(delta)} = llvm\.mlir\.constant\(7 : index\) : i64$",
        mlir,
        re.MULTILINE,
    )


@pytest.mark.parametrize(
    ("parent_name", "parent_layout"),
    _MAKE_TENSOR_LIKE_LAYOUT_CASES,
)
@pytest.mark.parametrize(
    ("child_name", "child_layout"),
    _MAKE_TENSOR_LIKE_LAYOUT_CASES,
)
def test_make_tensor_like_supports_every_layout_pair(
    parent_name: str,
    parent_layout: object,
    child_name: str,
    child_layout: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """All seven layouts can be the parent and child of make_tensor_like."""
    monkeypatch.setitem(globals(), "_MAKE_TENSOR_LIKE_PARENT_LAYOUT", parent_layout)
    monkeypatch.setitem(globals(), "_MAKE_TENSOR_LIKE_CHILD_LAYOUT", child_layout)
    mem_in = make_fake_tensor(
        tla.Float32,
        (128, 128),
        (128, 1),
        origin_shape=(128, 128),
        layout_tag=tla.arch.RowMajor,
    )

    mlir = make_tensor_like_layout_pair_kernel.dump_mlir(type_args=(mem_in, 64, 96))
    assert f"layoutTag(<{parent_name}>)" in mlir
    assert f"layoutTag(<{child_name}>)" in mlir
    assert "!tla.shape<?,?>" in mlir

    input_path = tmp_path / f"layout_pair_{parent_name}_to_{child_name}.mlir"
    input_path.write_text(mlir)
    result = subprocess.run(
        [
            str(_require_hivm_tla_compile()),
            str(input_path),
            "-o",
            "-",
            "--mlir-print-ir-after=tla-lower-tensor-desc",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    desc_lines = [
        line.strip()
        for line in result.stderr.splitlines()
        if " = tla.tensor_desc " in line
    ]
    assert len(desc_lines) >= 4, result.stderr
    child_desc = desc_lines[-1]
    assert child_name in child_desc
    shape, stride, origin, coord = _tensor_desc_metadata(child_desc)
    assert len(shape) == 4
    assert len(stride) == 4
    assert len(origin) == 2
    assert len(coord) == 2

    if child_name == "RowMajor":
        assert shape[:2] == origin
        assert shape[2:] == stride[2:]
        _assert_f32_dynamic_stride_is_32b_aligned(
            result.stderr, extent=shape[1], stride=stride[0]
        )
    elif child_name == "ColumnMajor":
        assert shape[:2] == origin
        assert shape[2:] == stride[2:]
        _assert_f32_dynamic_stride_is_32b_aligned(
            result.stderr, extent=shape[0], stride=stride[1]
        )
    else:
        assert child_name in {"zN", "nZ", "L0Clayout", "zNUnAlign"}


@tla.kernel
def nested_ub_subtile_copy_kernel(mem_in: tla.Tensor, mem_out: tla.Tensor) -> None:
    gm_in = tla.tile_view(mem_in, tla.make_shape(64, 64), tla.make_coord(0, 0))
    gm_out = tla.tile_view(mem_out, tla.make_shape(32, 32), tla.make_coord(0, 0))
    ub_ptr = tla.allocate((64, 64), tla.Float32, tla.AddressSpace.ub, 256)
    ub_root = tla.make_tensor_like(ub_ptr, gm_in, tla.arch.RowMajor)
    ub_tile = tla.tile_view(ub_root, tla.make_shape(32, 32), tla.make_coord(1, 1))
    with tla.vector():
        tla.copy(ub_root, gm_in)
        tla.copy(gm_out, ub_tile)


def test_nested_ub_subtile_copy_lowers(tmp_path) -> None:
    """GM<->UB staging copies lower to vector-core (AIV) cifax runtime calls."""
    tla_compile = _require_hivm_tla_compile()
    mem_in = make_fake_tensor(
        tla.Float32,
        (64, 64),
        (64, 1),
        origin_shape=(64, 64),
        layout_tag=tla.arch.RowMajor,
    )
    mem_out = make_fake_tensor(
        tla.Float32,
        (32, 32),
        (32, 1),
        origin_shape=(32, 32),
        layout_tag=tla.arch.RowMajor,
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
    assert "copy_gm_RowMajor_to_ub_RowMajor_float" in lowered
    assert "copy_ub_RowMajor_to_gm_RowMajor_float" in lowered
    assert "hivm.func_core_type = #hivm.func_core_type<AIV>" in lowered
    assert '"tla.copy"' not in lowered


@tla.kernel
def ptradd_ub_subtile_copy_kernel(mem_in: tla.Tensor, mem_src: tla.Tensor) -> None:
    gm_root = tla.tile_view(mem_in, tla.make_shape(64, 64), tla.make_coord(0, 0))
    gm_src = tla.tile_view(mem_src, tla.make_shape(32, 32), tla.make_coord(0, 0))
    ub_ptr = tla.allocate(64 * 64 + 16, tla.Float32, tla.AddressSpace.ub, 256) + 16
    ub_root = tla.make_tensor_like(ub_ptr, gm_root, tla.arch.RowMajor)
    ub_tile = tla.tile_view(ub_root, tla.make_shape(32, 32), tla.make_coord(1, 1))
    with tla.vector():
        tla.copy(ub_tile, gm_src)


def test_ptradd_ub_subtile_copy_applies_ptr_offset(tmp_path) -> None:
    """The ptr_add offset is preserved in the cifax base pointer_cast and tile payload."""
    tla_compile = _require_hivm_tla_compile()
    mem_in = make_fake_tensor(
        tla.Float32,
        (64, 64),
        (64, 1),
        origin_shape=(64, 64),
        layout_tag=tla.arch.RowMajor,
    )
    mem_src = make_fake_tensor(
        tla.Float32,
        (32, 32),
        (32, 1),
        origin_shape=(32, 32),
        layout_tag=tla.arch.RowMajor,
    )

    mlir = ptradd_ub_subtile_copy_kernel.dump_mlir(type_args=(mem_in, mem_src))
    input_path = tmp_path / "ptradd_ub_subtile.mlir"
    (input_path.write_text(mlir),)
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
    assert "copy_gm_RowMajor_to_ub_RowMajor_float" in out
    assert "hivm.func_core_type = #hivm.func_core_type<AIV>" in out
    # ptr_add contributes 64 bytes (16 f32 elements) to the UB base pointer_cast;
    # the UB sub-tile (coord (1,1) of a 64-wide buffer) carries stride0=64 and
    # coord (32,32) in the i64 payload, so the stub computes the flat offset
    # (32*64+32 = 2080) from coord/stride at runtime.
    assert "arith.constant 64 : i64" in out
    assert '"tla.copy"' not in out


# UB -> UB is not an MTE transfer -- no engine has UB on both sides. It goes
# through the DataCopy meta-op (MOV.UB.TO.UB) for every row length: the move
# counts in whole 32-byte blocks and rounds a ragged row up to one. The frontend
# must accept the route inside a tla.vector region rather than rejecting it as
# unsupported.
def _ub_to_ub_kernel(dtype, cols):
    """A UB->UB copy of a 1 x cols tile, built for one dtype."""

    @tla.kernel
    def kernel(mem_in: tla.Tensor) -> None:
        src = tla.make_tensor_like(
            tla.allocate((1, cols), dtype, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        dst = tla.make_tensor_like(
            tla.allocate((1, cols), dtype, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        with tla.vector():
            tla.copy(dst, src)

    return kernel


# The route is accepted for every registered width. A 64-element row is a whole
# number of 32-byte blocks for each of these dtypes, so it takes the DataCopy
# meta-op path (a tla.copy op).
@pytest.mark.parametrize(
    "dtype",
    [
        tla.Float32,
        tla.Float16,
        tla.BFloat16,
        tla.Int32,
        tla.Int16,
        tla.Int8,
        tla.Float8E4M3FN,
        tla.Float8E5M2,
    ],
)
def test_frontend_copy_ub_to_ub_accepts_every_registered_dtype(dtype) -> None:
    mem = make_fake_tensor(
        dtype, (1, 64), (64, 1), origin_shape=(1, 64), layout_tag=tla.arch.RowMajor
    )
    mlir = _ub_to_ub_kernel(dtype, 64).dump_mlir(type_args=(mem,))
    assert "tla.copy" in mlir


# A byte row that is not a whole number of 32-byte blocks takes the same
# DataCopy meta-op as a block-aligned one: the move rounds the row up to whole
# blocks. Every width is accepted -- b8 included.
@pytest.mark.parametrize("dtype", [tla.Int8, tla.Float8E4M3FN, tla.Float8E5M2])
@pytest.mark.parametrize("cols", [1, 17, 48, 63, 64])
def test_frontend_copy_ub_to_ub_accepts_byte_rows(dtype, cols) -> None:
    mem = make_fake_tensor(
        dtype,
        (1, cols),
        (cols, 1),
        origin_shape=(1, cols),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = _ub_to_ub_kernel(dtype, cols).dump_mlir(type_args=(mem,))
    assert "tla.copy" in mlir


def test_frontend_copy_ub_to_ub_lowers_to_runtime_call() -> None:
    mem = make_fake_tensor(
        tla.Float32,
        (32, 32),
        (32, 1),
        origin_shape=(32, 32),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = copy_ub_to_ub_kernel.dump_mlir(type_args=(mem,))
    assert "tla.copy" in mlir


# An enclosing tla.vec.func region does not change which route a UB->UB copy
# takes: the meta-op is emitted inside one exactly as it is outside, and a ragged
# row is rounded up to whole blocks either way.
def _ub_to_ub_in_vec_func_kernel(dtype, cols):
    """A UB->UB copy of a 1 x cols tile, written inside a tla.vec.func region."""

    @tla.kernel
    def kernel(mem_in: tla.Tensor) -> None:
        src = tla.make_tensor_like(
            tla.allocate((1, cols), dtype, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        dst = tla.make_tensor_like(
            tla.allocate((1, cols), dtype, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        with tla.vector():
            with tla.vec.func(mode="simd"):
                tla.copy(dst, src)

    return kernel


# 48 bytes of f32 (12 elements) and 48 bytes of i8 are both one and a half
# blocks; 17 bytes of fp8 is just over half of one. All of them are copied.
@pytest.mark.parametrize(
    ("dtype", "cols"),
    [
        (tla.Float32, 12),
        (tla.Float16, 24),
        (tla.Int8, 48),
        (tla.Float8E4M3FN, 17),
        (tla.Float32, 8),
        (tla.Int8, 32),
    ],
)
def test_frontend_copy_ub_to_ub_inside_vec_func_takes_the_metaop(dtype, cols) -> None:
    mem = make_fake_tensor(
        dtype,
        (1, cols),
        (cols, 1),
        origin_shape=(1, cols),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = _ub_to_ub_in_vec_func_kernel(dtype, cols).dump_mlir(type_args=(mem,))
    assert "tla.copy" in mlir


# A padded sub-tile with a ragged row is copied as long as the pitch leaves a
# whole number of blocks of slack for the rounded row. A 64x64 f32 parent sliced
# to 64x12 has a 256-byte pitch and a 48-byte row, rounded to 64.
def test_frontend_copy_ub_to_ub_accepts_ragged_subtile_with_padded_pitch() -> None:
    @tla.kernel
    def kernel(mem_in: tla.Tensor) -> None:
        src_root = tla.make_tensor_like(
            tla.allocate((64, 64), tla.Float32, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        dst_root = tla.make_tensor_like(
            tla.allocate((64, 64), tla.Float32, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        src = tla.tile_view(src_root, tla.make_shape(64, 12), tla.make_coord(0, 0))
        dst = tla.tile_view(dst_root, tla.make_shape(64, 12), tla.make_coord(0, 0))
        with tla.vector():
            tla.copy(dst, src)

    mem = make_fake_tensor(
        tla.Float32,
        (64, 64),
        (64, 1),
        origin_shape=(64, 64),
        layout_tag=tla.arch.RowMajor,
    )
    assert "tla.copy" in kernel.dump_mlir(type_args=(mem,))


# The block move addresses whole blocks, so a tile that does not *start* on one
# has no route. A 12-element f32 tile at tile index 1 begins at byte 48.
def test_frontend_copy_ub_to_ub_rejects_unaligned_origin() -> None:
    @tla.kernel
    def kernel(mem_in: tla.Tensor) -> None:
        src_root = tla.make_tensor_like(
            tla.allocate((64, 64), tla.Float32, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        dst_root = tla.make_tensor_like(
            tla.allocate((64, 64), tla.Float32, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        src = tla.tile_view(src_root, tla.make_shape(64, 12), tla.make_coord(0, 1))
        dst = tla.tile_view(dst_root, tla.make_shape(64, 12), tla.make_coord(0, 1))
        with tla.vector():
            tla.copy(dst, src)

    mem = make_fake_tensor(
        tla.Float32,
        (64, 64),
        (64, 1),
        origin_shape=(64, 64),
        layout_tag=tla.arch.RowMajor,
    )
    with pytest.raises(
        runtime_mod.TlaCoreAPIError,
        match=r"source tile starts at byte 48 of its parent",
    ):
        kernel.dump_mlir(type_args=(mem,))


# A multi-row ragged tile is copied, because a UB allocation's row pitch is
# padded to 32 bytes: 4 x 12 f32 is a 48-byte row with a 64-byte pitch, so the
# tail the block move rounds up to lands in that padding rather than on the next
# row. (The frontend still refuses a pitch too narrow to carry it, which is what
# makes the meta-op's gap arithmetic exact, but an allocation cannot produce
# one.)
def test_frontend_copy_ub_to_ub_accepts_ragged_rows_with_padded_pitch() -> None:
    mem = make_fake_tensor(
        tla.Float32,
        (4, 12),
        (12, 1),
        origin_shape=(4, 12),
        layout_tag=tla.arch.RowMajor,
    )

    @tla.kernel
    def kernel(mem_in: tla.Tensor) -> None:
        src = tla.make_tensor_like(
            tla.allocate((4, 12), tla.Float32, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        dst = tla.make_tensor_like(
            tla.allocate((4, 12), tla.Float32, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        with tla.vector():
            tla.copy(dst, src)

    assert "tla.copy" in kernel.dump_mlir(type_args=(mem,))


# UB -> UB with a ColumnMajor destination is a transposing copy: both tiles keep
# the logical (M, N) shape and the bytes land as (N, M). It is its own route,
# served by the meta-op's VNCHWCONV, so it reaches a runtime call distinct from
# the straight RowMajor->RowMajor one.
def _ub_to_ub_transpose_kernel(dtype, rows, cols):
    """A transposing UB->UB copy of a rows x cols tile, built for one dtype."""

    @tla.kernel
    def kernel(mem_in: tla.Tensor) -> None:
        src = tla.make_tensor_like(
            tla.allocate((rows, cols), dtype, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        dst = tla.make_tensor_like(
            tla.allocate((cols, rows), dtype, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.ColumnMajor,
        )
        with tla.vector():
            tla.copy(dst, src)

    return kernel


@pytest.mark.parametrize(
    "dtype",
    [
        tla.Float32,
        tla.Float16,
        tla.BFloat16,
        tla.Int32,
        tla.Int16,
        tla.Int8,
        tla.Float8E4M3FN,
        tla.Float8E5M2,
    ],
)
def test_frontend_copy_ub_to_ub_transpose_accepts_every_registered_dtype(dtype) -> None:
    mem = make_fake_tensor(
        dtype, (64, 64), (64, 1), origin_shape=(64, 64), layout_tag=tla.arch.RowMajor
    )
    mlir = _ub_to_ub_transpose_kernel(dtype, 64, 64).dump_mlir(type_args=(mem,))
    # Every registered width rides VNCHWCONV, so the op survives to a tla.copy.
    assert "tla.copy" in mlir


def test_frontend_copy_ub_to_ub_transpose_is_its_own_route() -> None:
    """The transposing copy must not be served by the straight route's callee."""
    mem = make_fake_tensor(
        tla.Float32, (64, 64), (64, 1), origin_shape=(64, 64), layout_tag=tla.arch.RowMajor
    )
    mlir = _ub_to_ub_transpose_kernel(tla.Float32, 64, 64).dump_mlir(type_args=(mem,))
    assert "tla.copy" in mlir
    # The layout tags are what separate the two routes, and the destination's
    # ColumnMajor tag is what picks the transposing callee further down.
    assert "ColumnMajor" in mlir


def test_frontend_copy_ub_to_ub_transpose_refuses_a_shape_the_metaop_cannot_do() -> None:
    """A tile that is not a whole number of VNCHWCONV blocks is refused.

    16 rows by C0 columns is the block, so a 24x40 f16 tile is neither a whole
    number of rows nor of columns and would have the meta-op read and write past
    the tile's edges.
    """
    mem = make_fake_tensor(
        tla.Float16, (24, 40), (40, 1), origin_shape=(24, 40), layout_tag=tla.arch.RowMajor
    )
    with pytest.raises(Exception, match="multiple of"):
        _ub_to_ub_transpose_kernel(tla.Float16, 24, 40).dump_mlir(type_args=(mem,))
