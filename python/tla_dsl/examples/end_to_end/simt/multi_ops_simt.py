"""Every SIMT scalar op, over every element type.

Two kernels, because the per-thread ops split by type:

    multi_ops_float_simt   f32 / f16 / bf16
        t = clamp(((a + b) - c) * d / e, -8, 8)   simt_add/sub/mul/div
                                                  simt_max/simt_min
        out = t if a > b else t + 1               simt_cmp/simt_where
        round-tripped through the other width     simt_cast
        m = sqrt(|t|) + exp(g) + log(h)           simt_sqrt/simt_abs/simt_exp
            + a ** 2                              simt_log/simt_pow

    multi_ops_int_simt     i32 / i16 / i8
        the same arithmetic with ``//`` for the divide, and no math half:
        sqrt/exp/log/abs are float-only, and ``**`` has no integer
        lowering.

``--dtype`` picks the element type and therefore the kernel; ``--dtype all``
runs all six.

The arithmetic is exact for every type (the divisor is a power of two and the
operands are small integers), so it is compared exactly. The float math half
gets a tolerance, since transcendentals are not bit-identical to the host.

Two lowering notes. ``//`` maps to arith.divsi, which truncates toward zero
rather than flooring like Python, so the reference uses
``rounding_mode="trunc"``. And bf16 has no transcendental unit, so
tla-vector-region evaluates those five ops in f32 and rounds back -- see
Tla.td. Of the math ops, only the ones that survive this pipeline are used:
sin/cos, floor/ceil/round and log2/exp2 have no instruction on this target,
and log goes through llvm.intr.log because convert-hivm-to-std would turn
math.log into a vln_1d_float call the c310 bitcode does not export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass.tla as tla

N_ELE = 1024
THREADS = 128

# name -> (tla type, torch dtype, kind)
DTYPES = {
    "f32": ("Float32", "float32", "float"),
    "f16": ("Float16", "float16", "float"),
    "bf16": ("BFloat16", "bfloat16", "float"),
    "i32": ("Int32", "int32", "int"),
    "i16": ("Int16", "int16", "int"),
    "i8": ("Int8", "int8", "int"),
}
DEFAULT_DTYPE = "f32"

# Both set from --dtype before the kernel is traced. _CAST_VIA is always the
# other width, so the round-trip narrows once and widens once whatever the
# kernel type is.
_KERNEL_DTYPE = tla.Float32
_CAST_VIA = tla.Float16


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@tla.kernel
def multi_ops_float_simt(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    gm_d: tla.Tensor,
    gm_e: tla.Tensor,
    gm_g: tla.Tensor,
    gm_h: tla.Tensor,
    gm_arith: tla.Tensor,
    gm_math: tla.Tensor,
) -> None:
    two = _KERNEL_DTYPE(2.0)
    lo = _KERNEL_DTYPE(-8.0)
    hi = _KERNEL_DTYPE(8.0)
    one = _KERNEL_DTYPE(1.0)

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            nthreads, _, _ = tla.arch.thread_block_dim()

            for i in tla.range(tid, N_ELE, nthreads):
                s = gm_a[i] + gm_b[i]  # tla.simt_add
                t = s - gm_c[i]  # tla.simt_sub
                u = t * gm_d[i]  # tla.simt_mul
                v = u / gm_e[i]  # tla.simt_div
                v = tla.max(v, lo)  # tla.simt_max
                v = tla.min(v, hi)  # tla.simt_min
                v = tla.where(gm_a[i] > gm_b[i], v, v + one)  # simt_cmp/where
                v = v.to(_CAST_VIA).to(_KERNEL_DTYPE)  # tla.simt_cast
                gm_arith[i] = v

                m = tla.sqrt(tla.abs(v))  # tla.simt_sqrt, tla.simt_abs
                m = m + tla.exp(gm_g[i])  # tla.simt_exp
                m = m + tla.log(gm_h[i])  # tla.simt_log
                gm_math[i] = m + gm_a[i] ** two  # tla.simt_pow

        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def multi_ops_int_simt(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    gm_d: tla.Tensor,
    gm_e: tla.Tensor,
    gm_arith: tla.Tensor,
) -> None:
    lo = _KERNEL_DTYPE(-8)
    hi = _KERNEL_DTYPE(8)
    one = _KERNEL_DTYPE(1)

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            nthreads, _, _ = tla.arch.thread_block_dim()

            for i in tla.range(tid, N_ELE, nthreads):
                s = gm_a[i] + gm_b[i]  # tla.simt_add
                t = s - gm_c[i]  # tla.simt_sub
                u = t * gm_d[i]  # tla.simt_mul
                v = u // gm_e[i]  # tla.simt_div (truncating)
                v = tla.max(v, lo)  # tla.simt_max
                v = tla.min(v, hi)  # tla.simt_min
                v = tla.where(gm_a[i] > gm_b[i], v, v + one)  # simt_cmp/where
                v = v.to(_CAST_VIA).to(_KERNEL_DTYPE)  # tla.simt_cast
                gm_arith[i] = v

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = EXAMPLE_DIR / "artifacts" / "runtime-cache"
_SENTINEL = -99


def _set_kernel_types(dtype_key: str) -> str:
    """The kernel bodies read these when traced, so set them before compiling."""
    global _KERNEL_DTYPE, _CAST_VIA
    tla_name, _, kind = DTYPES[dtype_key]
    _KERNEL_DTYPE = getattr(tla, tla_name)
    if kind == "float":
        _CAST_VIA = tla.Float16 if _KERNEL_DTYPE is tla.Float32 else tla.Float32
    else:
        _CAST_VIA = tla.Int8 if _KERNEL_DTYPE is tla.Int32 else tla.Int32
    return kind


def _compile_and_launch(args, kernel, host_tensors, from_dlpack, torch):
    tensors = [
        from_dlpack(t.contiguous(), layout_tag=tla.arch.RowMajor) for t in host_tensors
    ]
    artifact = tla.compile(
        kernel,
        *tensors,
        options=NPU_ARCH,
    )
    artifact(*tensors, block_num=1)
    torch.npu.synchronize()


def _run_float(args, dtype_key, torch, from_dlpack) -> bool:
    dt = getattr(torch, DTYPES[dtype_key][1])
    via = torch.float16 if dt is torch.float32 else torch.float32

    idx = torch.arange(N_ELE, dtype=torch.float32, device="npu")
    af, bf, cf = idx % 17.0, idx % 5.0, idx % 3.0
    df = (idx % 7.0) - 3.0
    # a power of two and never zero: the divide is exact
    ef = torch.full((N_ELE,), 4.0, dtype=torch.float32, device="npu")
    # exp() input small; log() input strictly positive
    gf, hf = (idx % 5.0) - 2.0, (idx % 11.0) + 1.0

    a, b, c, d, e, g, h = (t.to(dt) for t in (af, bf, cf, df, ef, gf, hf))
    arith_out = torch.full((N_ELE,), float(_SENTINEL), dtype=dt, device="npu")
    math_out = torch.full((N_ELE,), float(_SENTINEL), dtype=dt, device="npu")

    clamped = (((af + bf) - cf) * df / ef).clamp(-8.0, 8.0)
    selected = torch.where(af > bf, clamped, clamped + 1.0)
    expected_arith = selected.to(dt).to(via).to(dt)  # the cast round-trip
    expected_math = (
        expected_arith.float().abs().sqrt() + gf.exp() + hf.log() + af**2.0
    ).to(dt)

    _compile_and_launch(
        args,
        multi_ops_float_simt,
        (a, b, c, d, e, g, h, arith_out, math_out),
        from_dlpack,
        torch,
    )

    untouched = int((arith_out == _SENTINEL).sum()) + int((math_out == _SENTINEL).sum())
    arith_ok = bool(torch.equal(arith_out, expected_arith))
    rtol, atol = (1e-5, 1e-4) if dt is torch.float32 else (5e-2, 5e-2)
    math_ok = bool(
        torch.isclose(
            math_out.float(), expected_math.float(), rtol=rtol, atol=atol
        ).all()
    )
    if not arith_ok:
        bad = (arith_out != expected_arith).nonzero().flatten()[:3].tolist()
        print(
            f"        arith mismatch at {bad}: got={arith_out[bad].tolist()} "
            f"want={expected_arith[bad].tolist()}"
        )
    if not math_ok:
        diff = (math_out.float() - expected_math.float()).abs()
        k = int(diff.argmax())
        print(
            f"        math worst at {k}: got={float(math_out[k])} "
            f"want={float(expected_math[k])} diff={float(diff[k])}"
        )
    ok = arith_ok and math_ok
    print(
        f"  {dtype_key:>4}: arith_ok={arith_ok} math_ok={math_ok} "
        f"untouched={untouched}/{2 * N_ELE} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def _run_int(args, dtype_key, torch, from_dlpack) -> bool:
    dt = getattr(torch, DTYPES[dtype_key][1])
    via = torch.int8 if dt is torch.int32 else torch.int32

    # Ranges kept small so every intermediate fits in int8 too.
    idx = torch.arange(N_ELE, dtype=torch.int32, device="npu")
    ai, bi, ci = idx % 5, idx % 3, idx % 2
    di = (idx % 3) - 1
    ei = torch.full((N_ELE,), 2, dtype=torch.int32, device="npu")

    a, b, c, d, e = (t.to(dt) for t in (ai, bi, ci, di, ei))
    arith_out = torch.full((N_ELE,), _SENTINEL, dtype=dt, device="npu")

    u = ((ai + bi) - ci) * di
    # arith.divsi truncates toward zero; Python // floors
    v = torch.div(u, ei, rounding_mode="trunc").clamp(-8, 8)
    selected = torch.where(ai > bi, v, v + 1)
    expected_arith = selected.to(dt).to(via).to(dt)  # the cast round-trip

    _compile_and_launch(
        args,
        multi_ops_int_simt,
        (a, b, c, d, e, arith_out),
        from_dlpack,
        torch,
    )

    untouched = int((arith_out == _SENTINEL).sum())
    ok = bool(torch.equal(arith_out, expected_arith))
    if not ok:
        bad = (arith_out != expected_arith).nonzero().flatten()[:3].tolist()
        print(
            f"        mismatch at {bad}: got={arith_out[bad].tolist()} "
            f"want={expected_arith[bad].tolist()}"
        )
    print(
        f"  {dtype_key:>4}: arith_ok={ok} untouched={untouched}/{N_ELE} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def run_one(args: argparse.Namespace, dtype_key: str) -> bool:
    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    kind = _set_kernel_types(dtype_key)
    if kind == "float":
        return _run_float(args, dtype_key, torch, from_dlpack)
    return _run_int(args, dtype_key, torch, from_dlpack)


# Arch selection is the only Host compile knob now; caching moved to env vars
# (dsl e745bf10 converged the Host surface). --force-recompile / --no-cache are
# kept as flags and translated here so the runner scripts keep working.
NPU_ARCH = "--npu-arch 3510"


def _apply_cache_env(args) -> None:
    import os

    if getattr(args, "force_recompile", False):
        os.environ["CATLASS_DSL_FORCE_RECOMPILE"] = "1"
    if getattr(args, "no_cache", False):
        os.environ["CATLASS_DSL_CACHE"] = "0"
    if getattr(args, "cache_dir", None):
        os.environ["CATLASS_DSL_CACHE_DIR"] = str(args.cache_dir)


def run(args: argparse.Namespace) -> int:
    import torch

    keys = list(DTYPES) if args.dtype == "all" else [args.dtype]
    _apply_cache_env(args)

    torch.npu.set_device(args.device)
    print(f"--- multi_ops_simt n={N_ELE} block={THREADS} dtypes={','.join(keys)} ---")
    results = {key: run_one(args, key) for key in keys}
    passed = all(results.values())
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"failing dtypes: {','.join(failed)}")
    print(f"passed={passed}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile and run the per-thread scalar op kernels."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=[*DTYPES, "all"],
        help=f"element type to run (default: {DEFAULT_DTYPE}); 'all' runs every type",
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
