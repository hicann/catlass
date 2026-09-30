//===----------------------------------------------------------------------===//
// Execution regions: which unit a body runs on, and under which model.
//
// tla.cube and tla.vector name the unit -- AIC and AIV. tla.vec.func nests
// inside tla.vector and names the model: its mode attribute selects simd or
// simt, so it belongs to neither TlaSimdOps.td nor TlaSimtOps.td.
//
// The ops that run *in* each region live with their unit: TlaCubeOps.td,
// TlaSimdOps.td, TlaSimtOps.td.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

def Tla_CubeOp : Tla_Op<"cube", [NoTerminator, SingleBlock]> {
  let summary = "Cube region";
  let description = [{
    Launch a cube region for matrix multiply and related operations.
  }];
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $body attr-dict
  }];
}

def Tla_VectorOp : Tla_Op<"vector", [NoTerminator, SingleBlock]> {
  let summary = "Vector region";
  let description = [{
    Launch a vector region for post-processing and epilogue ops.
  }];
  let regions = (region AnyRegion:$body);
}

def Tla_VecFuncOp : Tla_Op<"vec.func", [NoTerminator, SingleBlock]> {
  let summary = "Vector compute function region";
  let description = [{
    Group register-resident vector compute ops inside a vector region. Must be
    nested inside a ``tla.vector`` region.

    ``mode`` selects the execution model: ``"simd"`` (the default when the
    attribute is absent) runs the body as whole-vector ops, ``"simt"`` runs it
    once per thread.

    ``thread_block_dim`` gives the ``(x, y, z)`` thread extents of one block and
    is valid only with ``mode = "simt"``, where it is required. Each extent must
    be positive and the product may not exceed 2048 threads per block, which is
    both the supported maximum and what fits the ``simt_entry`` attribute the
    lowering builds from it.
  }];
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}
