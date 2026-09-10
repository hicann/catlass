"""tla.full: broadcast a one-lane vector fragment OR a scalar, with/without a mask.

Exercises the two ``tla.full`` value forms and the optional lane mask:

* ``tla.full(reduced)``          — vector<1x> (reduction result) -> vector_broadcast (vdup.z)
* ``tla.full(scalar)``           — plain scalar -> broadcast (vdups.z)
* ``tla.full(..., mask=m)``      — the AVE broadcast is predicated by ``m``; only
                                   lane i < MASK_LANES is written.

The fragment form is the one that matters: a reduction result lives in a vector
register lane, and the SIMD backend has no instruction to read a lane into a
scalar register. Broadcasting the fragment directly never leaves the vector
register file, so a data-dependent value can be splatted without a scalar at all.

Four broadcast results are stored directly, so each output *is* the broadcast:
  out_vec_nomask : all lanes  = sum(x)
  out_scl_nomask : all lanes  = SCALAR_FILL
  out_vec_mask   : lanes<K    = sum(x)      (lanes>=K left unwritten)
  out_scl_mask   : lanes<K    = SCALAR_FILL  (lanes>=K left unwritten)
"""

from __future__ import annotations

import argparse

import catlass.tla as tla
import torch
import torch_npu
from catlass.tla.runtime import from_dlpack

VECTOR_ELE = 64
SCALAR_FILL = 3.0
MASK_LANES = 40  # masked full writes lanes [0, MASK_LANES)
_DT = tla.Float32


@tla.kernel
def full_op(
    mem_x: tla.Tensor,
    mem_vn: tla.Tensor,
    mem_sn: tla.Tensor,
    mem_vm: tla.Tensor,
    mem_sm: tla.Tensor,
) -> None:
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    vn_gm = tla.tile_view(mem_vn, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    sn_gm = tla.tile_view(mem_sn, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    vm_gm = tla.tile_view(mem_vm, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    sm_gm = tla.tile_view(mem_sm, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, _DT, tla.AddressSpace.ub, 256), x_gm, tla.arch.RowMajor
    )
    vn_ub = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, _DT, tla.AddressSpace.ub, 256),
        vn_gm,
        tla.arch.RowMajor,
    )
    sn_ub = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, _DT, tla.AddressSpace.ub, 256),
        sn_gm,
        tla.arch.RowMajor,
    )
    vm_ub = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, _DT, tla.AddressSpace.ub, 256),
        vm_gm,
        tla.arch.RowMajor,
    )
    sm_ub = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, _DT, tla.AddressSpace.ub, 256),
        sm_gm,
        tla.arch.RowMajor,
    )

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            shp = tla.make_shape(VECTOR_ELE)
            c0 = tla.make_coord(0)
            all_lanes, _ = tla.update_mask(VECTOR_ELE, _DT)
            m, _ = tla.update_mask(MASK_LANES, _DT)  # lanes < MASK_LANES

            xv = tla.tile_view(x_ub, shp, c0).load()
            reduced = xv.reduce(tla.ReductionOp.ADD, mask=all_lanes)  # vector<1x>

            tla.tile_view(vn_ub, shp, c0).store(tla.full(reduced, _DT))
            tla.tile_view(sn_ub, shp, c0).store(tla.full(SCALAR_FILL, _DT))
            tla.tile_view(vm_ub, shp, c0).store(tla.full(reduced, _DT, mask=m))
            tla.tile_view(sm_ub, shp, c0).store(tla.full(SCALAR_FILL, _DT, mask=m))

        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(vn_gm, vn_ub)
        tla.copy(sn_gm, sn_ub)
        tla.copy(vm_gm, vm_ub)
        tla.copy(sm_gm, sm_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    x = torch.linspace(-17.0, 46.0, VECTOR_ELE, dtype=torch.float32, device="npu")
    outs = {
        name: torch.full((VECTOR_ELE,), -999.0, dtype=torch.float32, device="npu")
        for name in ("vn", "sn", "vm", "sm")
    }
    tensors = [x, *outs.values()]
    tla_args = [from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in tensors]

    artifact = tla.compile(full_op, *tla_args, options="--npu-arch 3510")
    artifact(*tla_args, block_num=1)
    torch.npu.synchronize()

    s = float(x.sum())
    k = MASK_LANES

    def eq(a, b):
        return bool(torch.isclose(a, b, rtol=0.0, atol=1e-3).all())

    ok_vn = eq(outs["vn"], torch.full_like(outs["vn"], s))
    ok_sn = eq(outs["sn"], torch.full_like(outs["sn"], SCALAR_FILL))
    ok_vm = eq(outs["vm"][:k], torch.full((k,), s, device="npu"))
    ok_sm = eq(outs["sm"][:k], torch.full((k,), SCALAR_FILL, device="npu"))
    ok = ok_vn and ok_sn and ok_vm and ok_sm

    print(f"vector<1x> full, no mask (all == sum(x))      correct? {ok_vn}")
    print(f"scalar full,     no mask (all == {SCALAR_FILL})       correct? {ok_sn}")
    print(f"vector<1x> full, mask<{k} (active == sum(x))  correct? {ok_vm}")
    print(f"scalar full,     mask<{k} (active == {SCALAR_FILL})   correct? {ok_sm}")
    print(f"  masked-out lanes vm[{k}:] = {outs['vm'][k : k + 4].tolist()} (undefined)")
    print(f"  masked-out lanes sm[{k}:] = {outs['sm'][k : k + 4].tolist()} (undefined)")
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
