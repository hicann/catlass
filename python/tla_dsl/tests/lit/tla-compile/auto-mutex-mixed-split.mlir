// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-split-mixed-func 2>&1 | %filecheck %s

!gm = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!l1 = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
!ub = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>

module {
  tla.func @auto_mutex_mixed_split(%gm: !gm) attributes {tla.auto_sync = "v0"} {
    %ready = tla.cross_flag "ready" -> !tla.cross_flag<2>
    %l1_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %ub_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, ub, 256>
    %l1 = "tla.make_tensor_like"(%l1_ptr, %gm) {layoutTag = "zN"} : (!tla.ptr<f32, l1, 512>, !gm) -> !l1
    %ub = "tla.make_tensor_like"(%ub_ptr, %gm) {layoutTag = "RowMajor"} : (!tla.ptr<f32, ub, 256>, !gm) -> !ub
    "tla.cube"() ({
      tla.copy %l1, %gm : !l1, !gm
      tla.cross_core_set_flag %ready {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
    }) : () -> ()
    "tla.vector"() ({
      tla.cross_core_wait_flag %ready {pipe = #tla.pipe<vector>} : !tla.cross_flag<2>
      tla.copy %ub, %gm : !ub, !gm
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @auto_mutex_mixed_split_mix_aic
// CHECK: %[[L1:.*]] = tla.mutex "auto_l1_0_4096" {id = 0 : i64} -> !tla.mutex
// CHECK: tla.cube {
// CHECK-NEXT: tla.mutex_lock %[[L1]][<mte2>] : !tla.mutex
// CHECK-NEXT: tla.copy
// CHECK-NEXT: tla.mutex_unlock %[[L1]][<mte2>] : !tla.mutex

// CHECK-LABEL: func.func @auto_mutex_mixed_split_mix_aiv
// Cube and Vector own independent mutex ID spaces, so each side starts at 0.
// CHECK: %[[UB:.*]] = tla.mutex "auto_ub_0_4096" {id = 0 : i64} -> !tla.mutex
// CHECK: "tla.vector"() ({
// CHECK-NEXT: tla.cross_core_wait_flag
// CHECK-NEXT: tla.mutex_lock %[[UB]][<mte2>] : !tla.mutex
// CHECK-NEXT: tla.copy
// CHECK-NEXT: tla.mutex_unlock %[[UB]][<mte2>] : !tla.mutex
