"""MX MMAD: microscaling matmul on the cube, fp8 and fp4 through one kernel.

Tiled over M, N and K and spread across the cube cores, in the shape
basic_matmul uses: the tile extents are fixed Constexpr constants, the grid is
derived from the operands at runtime, and each core walks a strided slice of the
M x N tiles. The point is to exercise the scale path -- e8m0 blocks reaching L1,
riding the L1->L0 load, and steering the cube onto mad_mx -- on a kernel shaped
like the other matmul examples rather than a one-tile special case.

fp8 and fp4 share `mx_mmad_kernel`. They can, because the fp4-ness is not spelled
at the matmul: the operand element type carries it. The two paths therefore
differ only in which element type the tiles are allocated with -- everything
else, the scale tiles, the flags, the pipeline, the `tla.mmad_mx`, is identical.

What genuinely differs is host-side: fp8 quantises with torch's float8 dtypes,
while fp4 needs a 16-entry LUT and nibble packing. Hence two `quantize_mx_*`
functions over one shared scale packing.

Scale blocks go over as the plain (M, K/32) and (K/32, N) matrices. The GM->L1
DMA does the zZ / nN fractal reorder, steered by the GM-side layout tag, so the
host never pre-swizzles them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str((Path(__file__).resolve().parent / "..").resolve())

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.insert(0, _DSL_EXAMPLE_PATH)

import argparse
from dataclasses import dataclass

import catlass.tla as tla
import torch
import torch_npu  # noqa: F401
from catlass.tla.runtime import from_dlpack

# One e8m0 shared exponent per this many elements along K (OCP microscaling).
MX_SCALE_GROUP_NUM = 32
# L1 tile extents. The problem is chunked at these widths and the kernel derives
# the grid and the K-tile count from the operands themselves.
MX_L1_TM = 128
MX_L1_TN = 128
MX_L1_TK = 128

C0_NUM_PER_FRACTAL = 16
MX_SCALE_ELE_NUM_PER_C0 = 2
MX_SCALE_ELE_NUM_PER_FRACTAL = 32

# Which fp4 element type a case wants, selected by index. A plain int rather
# than a Constexpr[str]: a str-typed Constexpr does not deliver its value at
# trace time (it arrives as None), which silently made every fp4 kernel decode
# as e2m1.
FP8_SEL = -1
FP4_SEL = {"f4e2m1": 0, "f4e1m2": 1}
FP4_DTYPES = (tla.Float4E2M1, tla.Float4E1M2)


@dataclass(frozen=True)
class MxTilingParams:
    """Tile extents for the MX matmul, in the shape the other examples use.

    Fixed constants, independent of the problem shape: they size the on-chip
    buffers, which `tla.allocate` reserves at compile time. The kernel derives
    the grid and the K-tile count from the operands, so one compile serves any
    M, N and K that divide by these.
    """

    tm: tla.Constexpr[int] = MX_L1_TM
    tn: tla.Constexpr[int] = MX_L1_TN
    tk: tla.Constexpr[int] = MX_L1_TK


@tla.kernel
def mx_mmad_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_sa: tla.Tensor,
    gm_sb: tla.Tensor,
    gm_c: tla.Tensor,
    _tiling: MxTilingParams,
    fp4_sel: tla.Constexpr[int],
) -> None:
    # Resolved here, at trace time, so the loop body below stays free of
    # compile-time branching.
    is_fp4 = fp4_sel >= 0

    # The element type is the whole story for fp4: it states the 4-bit width --
    # which is what puts 64 elements in a 32-byte C0 -- and the encoding, and
    # the i8 buffer underneath is a lowering detail the kernel never mentions.
    # fp8 tiles take their type from the operand, so either pair may mix formats.
    dtype_a = FP4_DTYPES[fp4_sel] if is_fp4 else gm_a.ptr.dtype
    dtype_b = FP4_DTYPES[fp4_sel] if is_fp4 else gm_b.ptr.dtype

    c0 = 0
    c1 = 1
    # Constexpr, because tla.allocate reserves on-chip bytes at compile time and
    # so needs a literal count. They cannot be read off the tensors: the operands
    # are layout-dynamic, so origin_shape is a runtime extent -- which is exactly
    # what lets the grid and k_tiles below be runtime loop bounds, and equally
    # what stops any of them sizing a buffer.
    tm = _tiling.tm
    tn = _tiling.tn
    tk = _tiling.tk
    # The scale tile that accompanies a tk-wide operand tile: one e8m0 exponent
    # per group of 32 along K. Constexpr for the same reason -- it sizes l1sa/l1sb.
    sk = tk // MX_SCALE_GROUP_NUM

    # Derived, not passed: the extents come off the operands and the tile widths
    # off the tiling, so the two cannot disagree. All runtime values -- same
    # shape as basic_matmul's grid_m / grid_n / k_l1_count.
    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    grid_m = (m + tm - 1) // tm
    grid_n = (n + tn - 1) // tn
    total_blocks = grid_m * grid_n
    k_tiles = (gm_a.origin_shape[1] + tk - 1) // tk

    l1a = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l1, 512)
    l1b = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    # The scale tiles are e8m0 whatever the operands are: one shared exponent
    # per 32 elements along K.
    l1sa = tla.allocate(tm * sk, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(tm * tn, tla.Float32, tla.AddressSpace.l0c, 512)

    ab_ready = tla.flag("ab_ready", tla.arch.MTE2, tla.arch.MTE1)
    l0_ready = tla.flag("l0_ready", tla.arch.MTE1, tla.arch.CUBE)
    # Single-buffered L1/L0, so each stage must be told when its buffer is free
    # again: MTE1 releases L1 back to MTE2, and the cube releases L0 back to MTE1.
    l1_avail = tla.flag("l1_avail", tla.arch.MTE1, tla.arch.MTE2)
    l0_avail = tla.flag("l0_avail", tla.arch.CUBE, tla.arch.MTE1)

    with tla.cube():
        # One linear pass over the M x N tiles, strided by the core count, so a
        # core picks up every block_num-th tile. Same distribution basic_matmul
        # uses; the MX path is indifferent to it.
        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for block_linear in block_range:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n

            gm_c_t = tla.tile_view(
                gm_c, tla.make_shape(tm, tn), tla.make_coord(block_row, block_col)
            )
            t_l0c = tla.make_tensor_like(l0c, gm_c_t)

            # Set here rather than once outside: each tile's K loop consumes both
            # flags on the way out, so the pair has to be re-armed per tile for
            # the counts to balance.
            tla.set_flag(l1_avail)
            tla.set_flag(l0_avail)
            for kk in tla.range(c0, k_tiles, c1):
                # A walks K along its columns; B (column-major) and both scale
                # blocks walk K along their rows. The M/N coordinate is the tile
                # this core is on.
                gm_a_t = tla.tile_view(
                    gm_a, tla.make_shape(tm, tk), tla.make_coord(block_row, kk)
                )
                gm_b_t = tla.tile_view(
                    gm_b, tla.make_shape(tk, tn), tla.make_coord(kk, block_col)
                )
                # A's scale block is the real (M, K/32) matrix, so chunk kk is a
                # column range of it; B's is (K/32, N), so chunk kk is a row
                # range. The M/N coordinate is shared with the operand: a scale
                # block describes the tile it scales, row for row.
                gm_sa_t = tla.tile_view(
                    gm_sa, tla.make_shape(tm, sk), tla.make_coord(block_row, kk)
                )
                gm_sb_t = tla.tile_view(
                    gm_sb, tla.make_shape(sk, tn), tla.make_coord(kk, block_col)
                )

                # No layout tag: the L1 tile takes the orientation of the GM tile it
                # is loaded from, so --layout-a / --layout-b reach the kernel without
                # it having to know about them. The L0 tiles below are always zN / nZ
                # -- that is what the cube reads -- and the transposing L1 -> L0 load
                # bridges the two when the operand is the other way round.
                t_l1a = tla.make_tensor_like(l1a, gm_a_t)
                t_l1b = tla.make_tensor_like(l1b, gm_b_t)
                t_l1sa = tla.make_tensor_like(l1sa, gm_sa_t, tla.arch.zZMxScale)
                t_l1sb = tla.make_tensor_like(l1sb, gm_sb_t, tla.arch.nNMxScale)

                tla.wait_flag(l1_avail)
                tla.copy(t_l1a, gm_a_t)
                tla.copy(t_l1b, gm_b_t)
                tla.copy(t_l1sa, gm_sa_t)
                tla.copy(t_l1sb, gm_sb_t)
                tla.set_flag(ab_ready)
                tla.wait_flag(ab_ready)

                t_l0a = tla.make_tensor_like(l0a, t_l1a, tla.arch.zN)
                t_l0b = tla.make_tensor_like(l0b, t_l1b, tla.arch.nZ)

                tla.wait_flag(l0_avail)
                # The scale rides the L1->L0 load, not the matmul: this is what
                # attaches the e8m0 block and selects mad_mx.
                tla.copy(t_l0a, t_l1a, scale=t_l1sa)
                tla.copy(t_l0b, t_l1b, scale=t_l1sb)
                tla.set_flag(l1_avail)
                tla.set_flag(l0_ready)
                tla.wait_flag(l0_ready)

                init_c = True if kk == 0 else False
                unit_flag = 0b11 if kk == k_tiles - 1 else 0b10
                # tla.mmad_mx, not tla.mmad: both operands were loaded with a scale,
                # and the two ops are not interchangeable -- passing scaled operands
                # to tla.mmad is an error, and vice versa.
                tla.mmad_mx(t_l0c, t_l0a, t_l0b, init_c=init_c, unit_flag=unit_flag)
                tla.set_flag(l0_avail)

            tla.wait_flag(l1_avail)
            tla.wait_flag(l0_avail)
            tla.copy(gm_c_t, t_l0c, tla.params.CopyL0C2DstParams(unit_flag=0b11))


# ---------------------------------------------------------------------------
# Host-side quantisation. fp8 and fp4 diverge here and nowhere else.
# ---------------------------------------------------------------------------

# fp8 format constants, mirroring examples/53_ascend950_fp8_mx_matmul/gen_data.py.
_FP8_FORMATS = {
    torch.float8_e4m3fn: {"emax": 8, "max_value": 448.0},
    torch.float8_e5m2: {"emax": 15, "max_value": 57344.0},
}

# fp4 format constants, mirroring examples/54_ascend950_fp4_mx_matmul/gen_data.py.
_FP4_FORMATS = {
    "f4e2m1": {
        "exp_bits": 2,
        "mantissa_bits": 1,
        "bias": 1,
        "emax": 2,
        "max_value": 6.0,
    },
    "f4e1m2": {
        "exp_bits": 1,
        "mantissa_bits": 2,
        "bias": 1,
        "emax": 0,
        "max_value": 1.75,
    },
}

_FP8 = {
    "f8e4m3fn": (torch.float8_e4m3fn, tla.Float8E4M3FN),
    "f8e5m2": (torch.float8_e5m2, tla.Float8E5M2),
}


def _e8m0_exp(max_abs: torch.Tensor, emax: int) -> torch.Tensor:
    """Per-block e8m0 exponent, bit-exact with gen_data.py's ``_e8m0_exp``.

    ``floor(log2(x))`` for positive finite fp32 ``x`` is exactly
    ``biased_exp(x) - 127``, so it is read off the exponent field rather than
    computed in floating point.
    """
    zero_mask = max_abs < 1e-30
    safe = torch.where(zero_mask, torch.ones_like(max_abs), max_abs).float()
    bits = safe.contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127 - emax
    exp = exp.clamp(-128, 127)
    return torch.where(zero_mask, torch.zeros_like(exp), exp)


def _scale_bytes(scale: torch.Tensor, rows: int, nb: int) -> torch.Tensor:
    """e8m0 storage bytes: torch encodes exp2(e) as the biased exponent 127+e."""
    return torch.tensor(
        scale.to(torch.float8_e8m0fnu).flatten().untyped_storage(), dtype=torch.uint8
    ).reshape(rows, nb)


def quantize_mx_fp8(matrix: torch.Tensor, axis: int, fp8_dtype):
    """Split ``matrix`` into fp8 mantissas plus e8m0 block scales along ``axis``.

    Returns ``(fp8_values, e8m0_scale_bytes, dequantized_fp32)``. The
    dequantized tensor is what the device actually multiplies, so the reference
    is built from it -- only accumulation order then differs.
    """
    fmt = _FP8_FORMATS[fp8_dtype]
    work = matrix if axis == 1 else matrix.transpose(0, 1)
    work = work.float().contiguous()
    rows, cols = work.shape
    assert cols % MX_SCALE_GROUP_NUM == 0, "K must be a multiple of 32 for this slice"
    nb = cols // MX_SCALE_GROUP_NUM

    blocks = work.view(rows, nb, MX_SCALE_GROUP_NUM)
    exp = _e8m0_exp(blocks.abs().amax(dim=-1), fmt["emax"])
    scale = torch.exp2(exp.float())
    scaled = (blocks / scale.unsqueeze(-1)).clamp(-fmt["max_value"], fmt["max_value"])
    q = scaled.to(fp8_dtype)
    dequant = (q.float() * scale.unsqueeze(-1)).reshape(rows, cols)
    q = q.reshape(rows, cols)
    scale_bytes = _scale_bytes(scale, rows, nb)

    if axis == 1:
        return q, scale_bytes, dequant
    return (
        q.transpose(0, 1).contiguous(),
        scale_bytes.transpose(0, 1).contiguous(),
        dequant.transpose(0, 1).contiguous(),
    )


def _fp4_lut(fmt: str) -> torch.Tensor:
    """The 16 representable fp4 values, in encoding order."""
    cfg = _FP4_FORMATS[fmt]
    exp_bits, mbits, bias = cfg["exp_bits"], cfg["mantissa_bits"], float(cfg["bias"])
    values = []
    for i in range(16):
        sign = (i >> 3) & 0x01
        exp = (i >> mbits) & ((1 << exp_bits) - 1)
        mant = i & ((1 << mbits) - 1)
        if exp == 0:
            v = 0.0 if mant == 0 else (mant / float(1 << mbits)) * (2.0 ** (1.0 - bias))
        else:
            v = (1.0 + mant / float(1 << mbits)) * (2.0 ** (float(exp) - bias))
        values.append(-v if sign else v)
    return torch.tensor(values, dtype=torch.float32)


def quantize_mx_fp4(matrix: torch.Tensor, axis: int, fmt: str):
    """Split ``matrix`` into fp4 nibble indices plus e8m0 block scales.

    Returns ``(packed_uint8, e8m0_scale_bytes, dequantized_fp32)``. Packing is
    two elements per byte, element 2i in the low nibble (as gen_data.py does).
    """
    cfg = _FP4_FORMATS[fmt]
    lut = _fp4_lut(fmt)
    work = (matrix if axis == 1 else matrix.transpose(0, 1)).float().contiguous()
    rows, cols = work.shape
    assert cols % MX_SCALE_GROUP_NUM == 0, "K must be a multiple of 32 for this slice"
    nb = cols // MX_SCALE_GROUP_NUM

    blocks = work.view(rows, nb, MX_SCALE_GROUP_NUM)
    max_abs = blocks.abs().amax(dim=-1)
    exp = torch.floor(torch.log2(max_abs.clamp(min=1e-30))) - cfg["emax"]
    exp = torch.where(max_abs > 0, exp, torch.zeros_like(exp)).clamp(-127, 127)
    scale = torch.exp2(exp)

    scaled = (blocks / scale.unsqueeze(-1)).clamp(-cfg["max_value"], cfg["max_value"])
    # Nearest representable value; ties go to the lower index, matching argmin.
    idx = (scaled.unsqueeze(-1) - lut).abs().argmin(dim=-1)
    dequant = (lut[idx] * scale.unsqueeze(-1)).reshape(rows, cols)
    idx = idx.reshape(rows, cols).to(torch.uint8)

    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).to(torch.uint8)
    scale_bytes = _scale_bytes(scale, rows, nb)

    if axis == 1:
        return packed, scale_bytes, dequant
    # For the B side the caller wants (K, N) orientation back.
    return (
        packed,  # already (N, K/2): B is column-major
        scale_bytes.transpose(0, 1).contiguous(),  # (K/32, N)
        dequant.transpose(0, 1).contiguous(),  # (K, N)
    )


# ---------------------------------------------------------------------------
# Scale packing. Format-independent: an e8m0 block is a byte per 32 elements
# whatever the operands are, so both paths share this.
# ---------------------------------------------------------------------------
def _random_operands(m: int, n: int, k: int, spread: float):
    return (
        torch.randn(m, k, device="cpu") * spread,
        torch.randn(k, n, device="cpu") * spread,
    )


def run(args: argparse.Namespace) -> int:
    from common import compare, create_tla_tensor, get_block_num

    # One layout per side, driving both the operand and its scale block: a
    # scale block describes the operand it scales, so they are never independent.
    # Row-major A and column-major B reach L1 through a transposing DN2NZ, the
    # other orientations through a plain ND2NZ.
    layout_a = getattr(args, "layout_a", "row")
    layout_b = getattr(args, "layout_b", "col")

    torch.npu.set_device(args.device)
    m, n, k = args.m, args.n, args.k
    # Fixed tile extents, as in basic_matmul: the kernel derives the grid and the
    # K-tile count from the operands, so the host states no shape-dependent
    # tiling. Whole tiles only -- the kernel has no partial-tile predication and
    # a scale group must not straddle a K tile.
    tm, tn = min(m, MX_L1_TM), min(n, MX_L1_TN)
    tk = min(k, MX_L1_TK)
    assert m % tm == 0, "M must divide evenly into the L1 M-tile height"
    assert n % tn == 0, "N must divide evenly into the L1 N-tile width"
    assert k % tk == 0, "K must divide evenly into the L1 K-tile width"
    is_fp4 = args.fp4_case is not None

    # Packed fp4 pairs two elements per byte along K, and the fractal layout
    # requires that pairing to be the contiguous one, so an fp4 operand can only
    # be handed over with K contiguous: a row-major A (M, K/2) or a column-major
    # B (N, K/2). The other two orientations would put M or N contiguous and
    # split a nibble pair across the stride. fp8 has no such constraint -- every
    # element is a whole byte -- so it takes all four combinations.
    if is_fp4 and (
        getattr(args, "layout_a", "row") != "row"
        or getattr(args, "layout_b", "col") != "col"
    ):
        raise SystemExit(
            "packed fp4 requires --layout-a row --layout-b col: the two-per-byte "
            "packing runs along K and must be the contiguous axis"
        )

    if is_fp4:
        fmt = args.fp4_case
        print(
            f"--- mx fp4 mnk=({m},{n},{k}) fmt={fmt} "
            f"grid={m // tm}x{n // tn} k_tiles={k // tk} ---"
        )
    else:
        print(
            f"--- mx fp8 mnk=({m},{n},{k}) "
            f"dtype={args.dtype_a}/{args.dtype_b} "
            f"grid={m // tm}x{n // tn} k_tiles={k // tk} ---"
        )
    torch.manual_seed(0)

    # fp4's tighter dynamic range wants a smaller spread than fp8's.
    a_f, b_f = _random_operands(m, n, k, 4.0 if is_fp4 else 10.0)
    if not is_fp4:
        a_f, b_f = a_f - 5.0, b_f - 5.0

    # Quantise the whole operand at once. An MX scale group is 32 elements along
    # K and never straddles a K tile, so this is identical to quantising per tile
    # and concatenating -- which is what this used to do, back when the host also
    # pre-swizzled each tile's scale block into zZ / nN. The DMA does that
    # reorder now, so the blocks go over as the plain (M, K/32) and (K/32, N)
    # matrices and there is nothing left for the host to chunk.
    if is_fp4:
        qa, sa_packed, adq = quantize_mx_fp4(a_f, 1, fmt)
        qb, sb_packed, bdq = quantize_mx_fp4(b_f, 0, fmt)
    else:
        qa, sa_packed, adq = quantize_mx_fp8(a_f, 1, _FP8[args.dtype_a][0])
        qb, sb_packed, bdq = quantize_mx_fp8(b_f, 0, _FP8[args.dtype_b][0])

    ref = adq @ bdq

    # Each operand is physically transposed to match its --layout-*, the same
    # convention create_tla_tensor expects and basic_matmul uses: a column-major
    # operand is handed over as the transposed contiguous buffer.
    if layout_a == "col":
        qa = qa.permute(1, 0).contiguous()
    # fp4's B side comes back already packed as (N, K/2) -- the column-major
    # form -- because the nibble pairing runs along K. fp8's is (K, N).
    # fp4's B side comes back already packed as (N, K/2) -- the column-major
    # form, because the nibble pairing runs along K. fp8's is (K, N).
    if (layout_b == "row") if is_fp4 else (layout_b == "col"):
        qb = qb.permute(1, 0).contiguous()
    # Packed blocks are stacked per chunk along rows so chunk kk is at (kk, 0).
    # Natural blocks are the real matrices, so A's chunks join along columns.

    a_npu = qa.contiguous().view(torch.int8).npu()
    b_npu = qb.contiguous().view(torch.int8).npu()
    c_npu = torch.zeros(m, n, dtype=torch.float32).npu()

    # Each GM tag says how its logical scale matrix is laid out, and the two
    # copies really do differ: the row-major A / column-major B pair reach L1
    # through a transposing DN2NZ, the other pair through a plain ND2NZ. Because
    # the DMA moves e8m0 reinterpreted as half, every 16-bit unit has to hold two
    # groups of the *same* row (A) or column (B) -- so the ND2NZ pair takes the
    # block with its groups interleaved in C0-sized pairs, and a plain transpose
    # would pair two different rows and could never be right.
    C0 = MX_SCALE_ELE_NUM_PER_C0

    def _pair_interleave_a(t):  # (M, G) -> pairs of groups, M fastest between
        rows, groups = t.shape
        inter = t.view(rows, groups // C0, C0).permute(1, 0, 2).contiguous().reshape(-1)
        return torch.as_strided(inter, (rows, groups), (1, rows))

    def _pair_interleave_b(t):  # (G, N) -> pairs of groups, N fastest between
        groups, cols = t.shape
        inter = t.view(groups // C0, C0, cols).permute(0, 2, 1).contiguous().reshape(-1)
        return inter.view(groups, cols)

    sa_npu = sa_packed.view(torch.int8).contiguous().npu()
    sb_npu = sb_packed.view(torch.int8).contiguous().npu()
    # colMajorMxScaleA and rowMajorMxScaleB reach L1 through ND2NZ and need the
    # groups interleaved in C0-sized pairs; their DN2NZ counterparts take the
    # block as it comes.
    if layout_a == "col":
        sa_npu = _pair_interleave_a(sa_npu)
    if layout_b == "row":
        sb_npu = _pair_interleave_b(sb_npu)
    else:
        sb_npu = sb_npu.transpose(0, 1).contiguous().transpose(0, 1)

    if is_fp4:
        # The GM tile carries the fp4 element type, exactly as the fp8 branch
        # below carries its own -- the host has no fp4 dtype, so the buffer is
        # exported as bytes and element_type says how to read them. It cannot use
        # create_tla_tensor only because origin_shape has to state the count in
        # fp4 ELEMENTS while the buffer holds half that many, and the helper
        # takes no origin_shape.
        fp4_t = FP4_DTYPES[FP4_SEL[fmt]]
        a_t = from_dlpack(
            a_npu,
            layout_tag=tla.arch.RowMajor,
            origin_shape=(m, k),
            element_type=fp4_t,
        ).mark_layout_dynamic()
        b_t = from_dlpack(
            b_npu,
            layout_tag=tla.arch.ColumnMajor,
            origin_shape=(k, n),
            element_type=fp4_t,
        ).mark_layout_dynamic()
        fp4_sel = FP4_SEL[fmt]
    else:
        a_t = create_tla_tensor(a_npu, layout_a, _FP8[args.dtype_a][1])
        b_t = create_tla_tensor(b_npu, layout_b, _FP8[args.dtype_b][1])
        fp4_sel = FP8_SEL
    c_t = create_tla_tensor(c_npu, "row")
    sk_total = k // MX_SCALE_GROUP_NUM
    tag_a = (
        tla.arch.rowMajorMxScaleA if layout_a == "row" else tla.arch.colMajorMxScaleA
    )
    tag_b = (
        tla.arch.rowMajorMxScaleB if layout_b == "row" else tla.arch.colMajorMxScaleB
    )
    sa_t = from_dlpack(
        sa_npu,
        layout_tag=tag_a,
        origin_shape=(m, sk_total),
        element_type=tla.Float8E8M0,
    ).mark_layout_dynamic()
    sb_t = from_dlpack(
        sb_npu,
        layout_tag=tag_b,
        origin_shape=(sk_total, n),
        element_type=tla.Float8E8M0,
    ).mark_layout_dynamic()

    artifact = tla.compile(
        mx_mmad_kernel,
        a_t,
        b_t,
        sa_t,
        sb_t,
        c_t,
        MxTilingParams(tm=tm, tn=tn, tk=tk),
        fp4_sel,
        options="--npu-arch 3510",
    )
    block_num = get_block_num(args.block_num, args.device, kind="cube")
    artifact(a_t, b_t, sa_t, sb_t, c_t, block_num=block_num)
    torch.npu.synchronize()

    result = c_npu.detach().cpu().float()
    # The shared accumulative-precision check, as every other matmul example
    # uses: the tolerance scales with K, which is what a dot product's error
    # does. A misplaced scale block is off by a factor, not by rounding, so it
    # lands far outside this and is caught.
    passed = compare(result, ref, k, rtol=1.0 / 128.0 if is_fp4 else None)
    print(f"  passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    if not passed:
        d = (result - ref).abs()
        print(
            f"  max_abs_diff={d.max().item():.4f} ref_absmax={ref.abs().max().item():.4f}"
        )
        print(f"  result[0,:4]={result[0, :4].tolist()}")
        print(f"  ref[0,:4]={ref[0, :4].tolist()}")
    return 0 if passed else 1


def _apply_case(args: argparse.Namespace, case: str, parser) -> None:
    """Point `args` at one case: an fp8 operand pair, or a single fp4 encoding."""
    case = case.strip()
    if case in _FP4_FORMATS:
        args.fp4_case, args.dtype_a, args.dtype_b = case, None, None
        return
    parts = [part.strip() for part in case.split(",")]
    if len(parts) == 2 and all(part in _FP8 for part in parts):
        args.fp4_case = None
        args.dtype_a, args.dtype_b = parts
        return
    parser.error(
        f"--case expects an fp4 encoding {{{', '.join(sorted(_FP4_FORMATS))}}} "
        f"or an fp8 pair DTYPE_A,DTYPE_B with each in "
        f"{{{', '.join(sorted(_FP8))}}}, got {case!r}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--block-num", type=int, default=-1)
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--k", type=int, default=128)
    # One per side: the operand and its scale block share an orientation.
    p.add_argument("--layout-a", choices=("row", "col"), default="row")
    p.add_argument("--layout-b", choices=("row", "col"), default="col")
    p.add_argument(
        "--case",
        action="append",
        metavar="FP4_FORMAT|DTYPE_A,DTYPE_B",
        help="case to run; repeat to run several in one process",
    )
    args = p.parse_args()

    cases = args.case or ["f8e4m3fn,f8e4m3fn"]
    failures = 0
    for case in cases:
        _apply_case(args, case, p)
        rc = run(args)
        failures += rc
    print(f"SUMMARY passed={len(cases) - failures}/{len(cases)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
