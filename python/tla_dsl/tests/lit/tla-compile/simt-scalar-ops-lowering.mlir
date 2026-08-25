// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

// Every per-thread scalar arithmetic op in one SIMT region. tla-vector-region
// outlines the region into a vector function and rewrites each tla.simt_* op to
// the arith op for its element type, so none of them may survive the pass.

!fscalar = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>

module {
  func.func @simt_scalar_ops(
      %src_memref: memref<1xf32, #hivm.address_space<gm>>,
      %dst_memref: memref<1xf32, #hivm.address_space<gm>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c1, %src_c1, %src_c1] stride [%src_c1, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c1] coord [%src_c0, %src_c0] : memref<1xf32, #hivm.address_space<gm>> -> !fscalar
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c1, %dst_c1, %dst_c1] stride [%dst_c1, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c1] coord [%dst_c0, %dst_c0] : memref<1xf32, #hivm.address_space<gm>> -> !fscalar
    "tla.vec.func"() ({
      %i = arith.constant 0 : index
      %x = tla.simt_load %src[%i] : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32
      %sum = tla.simt_add %x, %x : f32
      %dif = tla.simt_sub %sum, %x : f32
      %prd = tla.simt_mul %dif, %x : f32
      %quo = tla.simt_div %prd, %x : f32
      %pow = tla.simt_pow %quo, %x : f32
      %mx  = tla.simt_max %pow, %x : f32
      %mn  = tla.simt_min %mx, %x : f32
      %cmp = tla.simt_cmp "gt" %mn, %x : f32
      %sel = tla.simt_where %cmp, %mn, %x : f32
      %narrow = tla.simt_cast "truncf" %sel : f32 to f16
      %wide = tla.simt_cast "extf" %narrow : f16 to f32
      %sqr = tla.simt_sqrt %wide : f32
      %ex  = tla.simt_exp %sqr : f32
      %lg  = tla.simt_log %ex : f32
      %ab  = tla.simt_abs %lg : f32
      tla.simt_store %dst[%i], %ab : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
    }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    return
  }
}

// The region becomes its own function with the SIMT calling convention, invoked
// through launch_func with one raw pointer per captured buffer.
// CHECK: hivm_regbaseintrins.intrins.launch_func @simt_scalar_ops_vf_simt
// CHECK-SAME: !llvm.ptr<1>, !llvm.ptr<1>
// CHECK: func.func @simt_scalar_ops_vf_simt
// CHECK-SAME: memref<1xf32, 1>
// CHECK-SAME: hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_entry<1>

// Element access becomes plain memref traffic on the outlined parameters, and
// each arithmetic op becomes its float arith counterpart, in program order.
// CHECK: memref.load
// CHECK: arith.addf
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: arith.divf
// CHECK: math.powf
// CHECK: llvm.intr.maxnum
// CHECK: llvm.intr.minnum
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: llvm.fptrunc
// CHECK: llvm.fpext
// CHECK: math.sqrt
// CHECK: math.exp
// CHECK: llvm.intr.log
// CHECK: math.absf
// CHECK: memref.store

// Nothing SIMT-specific may be left behind for the later passes.
// CHECK-NOT: tla.simt_add
// CHECK-NOT: tla.simt_sub
// CHECK-NOT: tla.simt_mul
// CHECK-NOT: tla.simt_div
// CHECK-NOT: tla.simt_pow
// CHECK-NOT: tla.simt_max
// CHECK-NOT: tla.simt_min
// CHECK-NOT: tla.simt_cmp
// CHECK-NOT: tla.simt_where
// CHECK-NOT: tla.simt_cast
// CHECK-NOT: tla.simt_sqrt
// CHECK-NOT: tla.simt_exp
// CHECK-NOT: tla.simt_log
// CHECK-NOT: tla.simt_abs
// CHECK-NOT: tla.simt_load
// CHECK-NOT: tla.simt_store
// CHECK-NOT: tla.vec.func
