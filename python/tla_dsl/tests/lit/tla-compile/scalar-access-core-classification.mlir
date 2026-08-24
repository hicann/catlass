// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-split-mixed-func 2>&1 | %filecheck %s

!ub = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!gm = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>

module {
  // UB scalar accesses live in a vector scope, so combining them with cube work
  // produces a mixed function and moves the whole scope to the AIV split.
  tla.func @ub_scalar_with_cube() {
    %raw = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<i8, ub, 256>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, ub, 256> -> !tla.ptr<f32, ub, 256>
    %shape = tla.make_shape -> !tla.shape<1>
    %stride = tla.make_stride -> !tla.stride<1>
    %layout = tla.make_layout %shape, %stride
      : !tla.shape<1>, !tla.stride<1>
        -> !tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>
    %coord = tla.make_coord -> !tla.coord<0>
    %tensor = tla.make_tensor %ptr, %layout, %coord
      : !tla.ptr<f32, ub, 256>,
        !tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>,
        !tla.coord<0> -> !ub
    %c0 = arith.constant 0 : index
    "tla.vector"() ({
      %value = tla.scalar_load %tensor[%c0] : !ub -> f32
      tla.scalar_store %tensor[%c0], %value : !ub, f32
    }) : () -> ()
    "tla.cube"() ({
    }) : () -> ()
    tla.return
  }

  // A GM scalar access does not by itself require AIV execution. With cube
  // work, the function therefore remains a single AIC function.
  tla.func @gm_scalar_with_cube(%arg0: !gm) {
    %c0 = arith.constant 0 : index
    %value = tla.scalar_load %arg0[%c0] : !gm -> f32
    "tla.cube"() ({
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @ub_scalar_with_cube_mix_aic(
// CHECK: tla.cube {
// CHECK-NOT: tla.scalar_load
// CHECK-NOT: tla.scalar_store

// CHECK-LABEL: func.func @ub_scalar_with_cube_mix_aiv(
// CHECK: tla.scalar_load
// CHECK: tla.scalar_store
// CHECK-NOT: tla.cube {

// CHECK-LABEL: func.func @gm_scalar_with_cube(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK: tla.scalar_load
// CHECK: tla.cube {
