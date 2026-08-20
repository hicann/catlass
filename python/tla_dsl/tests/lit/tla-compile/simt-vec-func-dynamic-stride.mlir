// RUN: %tla_compile %s -o - | %filecheck %s

// A rank-1 view whose stride is only known at runtime (here a function
// argument; in the DSL it is typically tla.arch.block_num() + 1).
//
// The stride cannot be read from the tensor type, so the folding takes it from
// the descriptor, where it is an SSA value. That multiply lands inside the
// region, which makes the stride a captured scalar -- and since an index-typed
// launch argument does not survive hivmc, it is narrowed to i32 outside the
// region and widened back inside.
//
// Before this, a dynamic stride was neither provably 1 nor provably anything
// else, so the index was emitted unscaled and t[1] silently addressed t[0]+1.

!ub_dyn = !tla.tensor<!tla.layout<!tla.shape<4>, !tla.stride<?>, !tla.shape<16>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @simt_dynamic_stride(%ub_memref: memref<16xf32, #hivm.address_space<ub>>,
                                 %stride: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %v = tla.tensor_desc %ub_memref shape [%c1, %c4, %c1, %c1] stride [%c16, %stride, %c1, %c1] origin_shape [%c1, %c16] coord [%c0, %c0] : memref<16xf32, #hivm.address_space<ub>> -> !ub_dyn
    "tla.vector"() ({
      "tla.vec.func"() ({
        %x = tla.simt_load %v[%c2] : <!tla.layout<!tla.shape<4>, !tla.stride<?>, !tla.shape<16>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>> -> f32
        tla.simt_store %v[%c2], %x : <!tla.layout<!tla.shape<4>, !tla.stride<?>, !tla.shape<16>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 4, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// The stride is narrowed once, outside the region, and forwarded as an i32
// launch argument after the buffer pointer.
// CHECK-LABEL: func.func @simt_dynamic_stride(
// CHECK:         hivm_regbaseintrins.intrins.launch_func @simt_dynamic_stride_vf_simt
// CHECK-SAME:      !llvm.ptr<6>, i32

// Inside, it is widened back and multiplied into the index -- both accesses
// share the one captured scalar. (The index is 2 rather than 1 on purpose: at
// index 1 the multiply folds to the stride itself and there is nothing to see.)
// CHECK-LABEL: func.func @simt_dynamic_stride_vf_simt
// CHECK-SAME:    i32
// CHECK:         llvm.sext %{{[a-z0-9_]+}} : i32 to i64
// CHECK:         llvm.mul
