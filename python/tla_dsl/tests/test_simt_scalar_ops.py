"""Frontend dispatch for the SIMT scalar arithmetic ops.

Inside a ``tla.vec.func`` with ``mode="simt"`` the Python operators emit
``tla.simt_*`` ops; outside one they keep emitting plain ``arith``. These tests
pin that split, which is easy to break by touching ``_binary_op`` in
``base_dsl/typing.py`` -- and hard to notice, because both forms lower to the
same arith op in the end and every kernel still passes.

The lowering itself (``tla.simt_* -> arith.*``) is covered by the lit test
``lit/tla-compile/simt-scalar-ops-lowering.mlir``.
"""

from __future__ import annotations

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor
from catlass._mlir_bindings import tla_ops_gen


@tla.kernel
def _simt_arith_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=64):
            tid, _, _ = tla.arch.thread_idx()
            nthreads, _, _ = tla.arch.thread_block_dim()
            for i in tla.range(tid, 64, nthreads):
                s = src[i] + src[i]
                t = s - src[i]
                u = t * src[i]
                v = u / src[i]
                m = tla.sqrt(abs(v)) + tla.exp(v) + tla.log(v)
                if src[i] > v:
                    m = m + v
                m = m.to(tla.Float16).to(tla.Float32)
                m = tla.where(m > v, m, v)
                m = tla.max(m, v)
                m = tla.min(m, src[i])
                dst[i] = m + v ** tla.Float32(2.0)


@tla.kernel
def _simt_int_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    """Integer per-thread ops: `//` must reach tla.simt_div like the rest."""
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=64):
            tid, _, _ = tla.arch.thread_idx()
            nthreads, _, _ = tla.arch.thread_block_dim()
            for i in tla.range(tid, 64, nthreads):
                v = (src[i] + src[i]) * src[i]
                v = v // src[i]
                v = tla.max(v, tla.Int32(0))
                dst[i] = tla.where(v > src[i], v, src[i])


@tla.kernel
def _simd_arith_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    """Same arithmetic, but on the scalar path outside any SIMT region."""
    with tla.vector():
        s = src[0] + src[0]
        t = s - src[0]
        u = t * src[0]
        v = u / src[0]
        dst[0] = v + abs(v) + v ** tla.Float32(2.0)


def _tensors():
    def _fake():
        return make_fake_tensor(
            tla.Float32,
            (64,),
            (1,),
            addrspace=tla.AddressSpace.gm,
            origin_shape=(64,),
        )

    return _fake(), _fake()


def test_generated_bindings_exist_for_simt_scalar_ops() -> None:
    for symbol in (
        "simt_add", "simt_sub", "simt_mul", "simt_div", "simt_pow",
        "simt_sqrt", "simt_exp", "simt_abs", "simt_log",
        "simt_max", "simt_min", "simt_cmp", "simt_where", "simt_cast",
    ):
        assert hasattr(tla_ops_gen, symbol), f"missing generated binding: {symbol}"


def test_simt_region_emits_simt_arithmetic_ops() -> None:
    src, dst = _tensors()
    mlir = _simt_arith_kernel.dump_mlir(type_args=(src, dst))
    for op_name in ("tla.simt_add", "tla.simt_sub", "tla.simt_mul", "tla.simt_div"):
        assert op_name in mlir, f"{op_name} not emitted inside a SIMT region"


def test_simt_region_emits_simt_math_ops() -> None:
    src, dst = _tensors()
    mlir = _simt_arith_kernel.dump_mlir(type_args=(src, dst))
    for op_name in (
        "tla.simt_sqrt",
        "tla.simt_exp",
        "tla.simt_abs",
        "tla.simt_log",
        "tla.simt_cmp",
        "tla.simt_where",
        "tla.simt_cast",
        "tla.simt_max",
        "tla.simt_min",
        "tla.simt_pow",
    ):
        assert op_name in mlir, f"{op_name} not emitted inside a SIMT region"


def test_simt_region_does_not_emit_raw_math() -> None:
    src, dst = _tensors()
    mlir = _simt_arith_kernel.dump_mlir(type_args=(src, dst))
    for op_name in ("math.sqrt", "math.exp", "math.absf", "math.powf", "math.log"):
        assert op_name not in mlir, f"{op_name} leaked into a SIMT region"


def test_simt_region_does_not_emit_raw_arith() -> None:
    """The point of the ops: a SIMT body is TLA IR, not raw arith."""
    src, dst = _tensors()
    mlir = _simt_arith_kernel.dump_mlir(type_args=(src, dst))
    for op_name in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf",
                    "arith.cmpf"):
        assert op_name not in mlir, f"{op_name} leaked into a SIMT region"


def test_simt_region_emits_simt_element_access() -> None:
    src, dst = _tensors()
    mlir = _simt_arith_kernel.dump_mlir(type_args=(src, dst))
    assert "tla.simt_load" in mlir
    assert "tla.simt_store" in mlir
    assert "tla.scalar_load" not in mlir
    assert "tla.scalar_store" not in mlir


def test_outside_a_simt_region_the_operators_stay_arith() -> None:
    """The dispatch is region-scoped: the general scalar path is unchanged."""
    src, dst = _tensors()
    mlir = _simd_arith_kernel.dump_mlir(type_args=(src, dst))
    for op_name in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf",
                    "math.absf", "math.powf"):
        assert op_name in mlir, f"{op_name} should still be emitted outside a SIMT region"
    for op_name in ("tla.simt_add", "tla.simt_sub", "tla.simt_mul", "tla.simt_div",
                    "tla.simt_abs", "tla.simt_pow"):
        assert op_name not in mlir, f"{op_name} emitted outside a SIMT region"
    assert "tla.scalar_load" in mlir
    assert "tla.scalar_store" in mlir


def _int_tensors():
    def _fake():
        return make_fake_tensor(
            tla.Int32, (64,), (1,),
            addrspace=tla.AddressSpace.gm, origin_shape=(64,),
        )

    return _fake(), _fake()


def test_simt_region_emits_simt_ops_for_integers() -> None:
    """Integer `//`, max and where go through the TLA ops, not raw arith."""
    src, dst = _int_tensors()
    mlir = _simt_int_kernel.dump_mlir(type_args=(src, dst))
    for op_name in ("tla.simt_add", "tla.simt_mul", "tla.simt_div",
                    "tla.simt_max", "tla.simt_cmp", "tla.simt_where"):
        assert op_name in mlir, f"{op_name} not emitted for an integer SIMT region"
    for op_name in ("arith.divsi", "arith.addi", "arith.muli"):
        assert op_name not in mlir, f"{op_name} leaked into an integer SIMT region"
