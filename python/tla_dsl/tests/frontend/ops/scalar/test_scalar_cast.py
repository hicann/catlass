import re

import pytest

import catlass.tla as tla
from catlass._mlir import ir, passmanager
from catlass._mlir.dialects import arith
from catlass.base_dsl import BaseDSL
from catlass.base_dsl.typing import _emit_low_precision_scalar_cast
from catlass.compiler_bridge import lower_tlair_module_to_mlir
from catlass.core_api import (
    _numeric_ir_value_for_element_type,
    _padding_value_to_i64_bits,
)
from catlass.tla.runtime import make_fake_tensor


BYTE_ENCODED_TYPES = (tla.HiFloat8, tla.Float4E1M2x2, tla.Float4E2M1x2)


@tla.kernel
def _scalar_cast(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        dst[0] = src[0].to(dst.element_type)


@tla.kernel
def _decode_and_copy_packed_value(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        value = src[0]
        dst[0] = value.to(tla.Float32)
        src[1] = value
        dst[1] = value.to(tla.Float32)


@tla.kernel
def _decode_after_if(src: tla.Tensor, flag: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        value = src[0]
        if flag[0] != 0:
            value = src[1]
        src[2] = value
        dst[0] = value.to(tla.Float32)


@tla.kernel
def _decode_after_for(src: tla.Tensor, count: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        value = src[0]
        for i in tla.range(count[0]):
            dst[i + 1] = value.to(tla.Float32)
            value = src[i + 1]
        src[2] = value
        dst[0] = value.to(tla.Float32)


@tla.kernel
def _decode_after_while(src: tla.Tensor, count: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        value = src[0]
        i = tla.Int32(0)
        limit = count[0]
        while i < limit:
            i = i + 1
            value = src[i]
        src[2] = value
        dst[0] = value.to(tla.Float32)


@tla.kernel
def _scalar_cast_in_scalar_only_region(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst[0] = src[0].to(dst.element_type)


@tla.kernel
def _simt_cast(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=32):
            tid, _, _ = tla.arch.thread_idx()
            dst[tid] = src[tid].to(dst.element_type)


def _lower(kernel, *dtypes):
    args = tuple(make_fake_tensor(dtype, (64,), (1,)) for dtype in dtypes)
    return BaseDSL()._lower(
        kernel.fn,
        kind=kernel.kind,
        options=dict(kernel.options),
        type_args=args,
        location=kernel.decorator_location,
    )


def _has_storage_byte_bridge(mlir):
    return any(
        "builtin.unrealized_conversion_cast" in line and " to i8" in line
        for line in mlir.splitlines()
    )


def _assert_carried_i8_dataflow(mlir: str, expected_decodes: int) -> None:
    i8_loads = re.findall(
        r"(%\w+)\s*=\s*memref\.load\b[^\n]*: memref<[^\n]*xi8\b", mlir
    )
    assert len(i8_loads) == 2

    i8_stores = re.findall(r"memref\.store\s+(%\w+),[^\n]*: memref<[^\n]*xi8\b", mlir)
    assert len(i8_stores) == 1
    carried = i8_stores[0]

    block_arg = None
    for block_name, args_text in re.findall(r"(\^bb\d+)\(([^\n]+)\):", mlir):
        for index, arg in enumerate(args_text.split(",")):
            if re.match(rf"^{re.escape(carried)}:\s*i8\b", arg.strip()):
                block_arg = (block_name, index)
                break
    assert block_arg is not None
    target_block, arg_index = block_arg

    incoming = [
        operands_text.split(",")[arg_index].strip()
        for operands_text in re.findall(
            rf"cf\.br\s+{re.escape(target_block)}\(([^:\n]+):", mlir
        )
    ]
    assert len(incoming) == 2 and set(incoming) == set(i8_loads)

    decodes = re.findall(
        r"(?:arith\.extui|llvm\.zext)\s+(%\w+)\s*:\s*i8 to i32\b", mlir
    )
    assert decodes == [carried] * expected_decodes


@pytest.mark.parametrize(
    "kernel",
    [_scalar_cast, _scalar_cast_in_scalar_only_region],
    ids=["kernel-body", "inlined-region"],
)
@pytest.mark.parametrize(
    "src, dst, result_bitcast",
    [
        (tla.Float32, tla.BFloat16, "i16 to bf16"),
        (tla.BFloat16, tla.Float32, "i32 to f32"),
        (tla.HiFloat8, tla.Float32, "i32 to f32"),
        (tla.Float4E1M2x2, tla.Float32, "i32 to f32"),
        (tla.Float4E2M1x2, tla.Float32, "i32 to f32"),
    ],
)
def test_scalar_cast_lowers_in_aiv_kernel(kernel, src, dst, result_bitcast):
    lowered = _lower(kernel, src, dst)
    frontend = lowered.asm(generic=False)
    assert "tla.scalar_cast" not in frontend
    assert "tla.aiv_scalar_cast" not in frontend
    assert any(
        "arith.bitcast" in line and f": {result_bitcast}" in line
        for line in frontend.splitlines()
    )

    result = lower_tlair_module_to_mlir(lowered.module).lowered_mlir
    assert "func.func private @vector_region_" not in result
    assert "call @_mlir_ciface_tla_scalar_cast" not in result
    if src.width == 8:
        assert any(
            "tla.scalar_load" in line and line.endswith(f"-> !tla.{src.dtype}")
            for line in frontend.splitlines()
        )
        assert "tla.scalar_storage_byte" in frontend
        assert "tla.scalar_storage_byte" not in result
        assert any(
            "arith.extui" in line and ": i8 to i32" in line
            for line in frontend.splitlines()
        )
        assert not _has_storage_byte_bridge(frontend)
        assert not _has_storage_byte_bridge(result)


@pytest.mark.parametrize("src", BYTE_ENCODED_TYPES)
def test_encoded_scalar_casts_share_the_storage_load(src):
    lowered = _lower(_decode_and_copy_packed_value, src, tla.Float32)
    result = lower_tlair_module_to_mlir(lowered.module).lowered_mlir
    assert result.count("memref.load") == 1
    assert result.count("memref.store") == 3
    assert not _has_storage_byte_bridge(result)


@pytest.mark.parametrize("src", BYTE_ENCODED_TYPES)
@pytest.mark.parametrize(
    "kernel, operation, expected_decodes",
    [
        (_decode_after_if, "scf.if", 1),
        (_decode_after_for, "scf.for", 2),
        (_decode_after_while, "scf.while", 1),
    ],
)
def test_encoded_scalar_cast_after_control_flow(
    kernel, operation, expected_decodes, src
):
    lowered = _lower(kernel, src, tla.Int32, tla.Float32)
    frontend = lowered.asm(generic=False)
    assert any(
        operation in line and f"!tla.{src.dtype}" in line
        for line in frontend.splitlines()
    )
    assert "tla.scalar_storage_byte" in frontend

    result = lower_tlair_module_to_mlir(lowered.module).lowered_mlir
    assert re.search(r"cf\.br [^\n]* : [^\n]*\bi8\b", result)
    _assert_carried_i8_dataflow(result, expected_decodes)
    assert f"!tla.{src.dtype}" not in result
    assert "tla.scalar_storage_byte" not in result
    assert not _has_storage_byte_bridge(result)


@pytest.mark.parametrize(
    "source, dst, expected",
    [
        (tla.Float32(float("nan")), tla.BFloat16, 0x7FFF),
        (tla.Float32(1.0 + 2.0**-8 + 2.0**-30), tla.BFloat16, 0x3F80),
        (tla.Float32(1.0 + 2.0**-8 - 2.0**-23), tla.BFloat16, 0x3F80),
        (tla.Float32(1.0 + 2.0**-8 + 2.0**-23), tla.BFloat16, 0x3F81),
        (tla.BFloat16(0.1), tla.Float32, 0x3DCD0000),
        (tla.BFloat16(1.0 + 2.0**-8 + 2.0**-30), tla.Float32, 0x3F810000),
        (tla.BFloat16(tla.BFloat16.signaling_nan), tla.Float32, 0x7FA00000),
    ],
)
def test_bfloat16_cast_preserves_folded_literal_bits(source, dst, expected):
    for folded in (source.to(dst), dst(source), tla.cast(source, dst)):
        assert isinstance(folded, dst)
        assert isinstance(folded.to(float), float)
        assert folded.__c_pointers__() == [expected]
        assert _padding_value_to_i64_bits(folded, dst.dtype, dst.width // 8) == expected
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                literal = dst(folded).ir_value()
                attr = ir.Attribute.parse(f"0x{expected:X} : {literal.type}")
                assert literal.owner.attributes["value"] == attr
                vec_scalar = _numeric_ir_value_for_element_type(
                    "add", folded, literal.type
                )
                assert vec_scalar.owner.attributes["value"] == attr


@pytest.mark.parametrize(
    "src, source_bits, dst, expected",
    [
        (tla.Float32, 0x3F807FFF, tla.BFloat16, 0x3F80),
        (tla.Float32, 0x3F808000, tla.BFloat16, 0x3F80),
        (tla.Float32, 0x3F808001, tla.BFloat16, 0x3F81),
        (tla.Float32, 0x3F818000, tla.BFloat16, 0x3F82),
        (tla.Float32, 0xFFC00000, tla.BFloat16, 0x7FFF),
        (tla.BFloat16, 0xFF81, tla.Float32, 0xFF810000),
        (tla.HiFloat8, 0x01, tla.Float32, 0x34800000),
        (tla.HiFloat8, 0x80, tla.Float32, 0x7FFFFFFF),
        (tla.HiFloat8, 0xEF, tla.Float32, 0xFF800000),
        (tla.Float4E1M2x2, 0xF7, tla.Float32, 0x3FE00000),
        (tla.Float4E2M1x2, 0xF7, tla.Float32, 0x40C00000),
        (tla.Float4E2M1x2, 0xF8, tla.Float32, 0x80000000),
    ],
)
def test_scalar_cast_constant_encodings(src, source_bits, dst, expected):
    with ir.Context(), ir.Location.unknown() as loc:
        module = ir.Module.create()
        raw_type = ir.IntegerType.get_signless(src.width)
        result_type = ir.IntegerType.get_signless(dst.width)
        with ir.InsertionPoint(module.body):
            function = ir.Operation.create(
                "func.func",
                attributes={
                    "sym_name": ir.StringAttr.get("constant_cast"),
                    "function_type": ir.TypeAttr.get(
                        ir.FunctionType.get([], [result_type])
                    ),
                },
                regions=1,
            )
        with ir.InsertionPoint(function.regions[0].blocks.append()):
            source = arith.constant(raw_type, source_bits)
            if src.width != 8:
                source = arith.bitcast(src.mlir_type(), source)
            result = _emit_low_precision_scalar_cast(source, src, dst.mlir_type(), loc)
            bits = arith.bitcast(result_type, result)
            ir.Operation.create("func.return", operands=[bits])

        lowered = ir.Module.parse(lower_tlair_module_to_mlir(module).lowered_mlir)
        passmanager.PassManager.parse("builtin.module(canonicalize)").run(
            lowered.operation
        )
        kernel = next(
            op
            for op in lowered.body.operations
            if ir.StringAttr(op.attributes["sym_name"]).value == "constant_cast"
        )
        returned = list(kernel.regions[0].blocks[0].operations)[-1].operands[0]
        constant = ir.IntegerAttr(returned.owner.attributes["value"])
        assert constant.value & ((1 << dst.width) - 1) == expected


@pytest.mark.parametrize(
    "src, dst", [(tla.BFloat16, tla.Float32), (tla.Float32, tla.BFloat16)]
)
def test_bfloat16_cast_in_simt_keeps_simt_lowering(src, dst):
    lowered = _lower(_simt_cast, src, dst)
    assert "tla.simt_cast" in lowered.asm(generic=True)
