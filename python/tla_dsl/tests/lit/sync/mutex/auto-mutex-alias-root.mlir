// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-insert-auto-mutex 2>&1 | %filecheck %s

!layout = !tla.layout<!tla.shape<?>, !tla.stride<1>, !tla.shape<?>, RowMajor>
!gm_layout = !tla.layout<!tla.shape<?>, !tla.stride<1>, !tla.shape<256>, RowMajor>
!coord = !tla.coord<0>
!ub = !tla.tensor<!layout, !coord, !tla.ptr<f32, ub, 256>>
!gm = !tla.tensor<!gm_layout, !coord, !tla.ptr<f32, gm, 4>>

module {
  tla.func @auto_mutex_alias_root(%extent: index, %gm: !gm) attributes {tla.auto_sync = "v0"} {
    %shape = "tla.make_shape"(%extent) : (index) -> !tla.shape<?>
    %stride = tla.make_stride -> !tla.stride<1>
    %layout = tla.make_layout %shape, %stride : !tla.shape<?>, !tla.stride<1> -> !layout
    %coord = tla.make_coord -> !coord
    %zero = arith.constant 0 : index
    %raw = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<i8, ub, 256>
    %f32 = tla.recast_ptr %raw : !tla.ptr<i8, ub, 256> -> !tla.ptr<f32, ub, 256>
    %offset = tla.ptr_add %f32, %zero : !tla.ptr<f32, ub, 256>, index -> !tla.ptr<f32, ub, 256>
    %i32 = tla.recast_ptr %offset : !tla.ptr<f32, ub, 256> -> !tla.ptr<i32, ub, 256>
    %alias = tla.recast_ptr %i32 : !tla.ptr<i32, ub, 256> -> !tla.ptr<f32, ub, 256>
    %tensor0 = tla.make_tensor %f32, %layout, %coord : !tla.ptr<f32, ub, 256>, !layout, !coord -> !ub
    %tensor1 = tla.make_tensor %alias, %layout, %coord : !tla.ptr<f32, ub, 256>, !layout, !coord -> !ub
    %tile = tla.tile_view %tensor1, %shape, %coord : !ub, !tla.shape<?>, !coord -> !ub
    "tla.vector"() ({
      tla.copy %tensor0, %gm : !ub, !gm
      tla.copy %tile, %gm : !ub, !gm
    }) : () -> ()
    tla.return
  }
}

// Dynamic tensor shape and every view-producing pointer transform retain the
// alloc_ptr root, so both instructions share one ID.
// CHECK-LABEL: func.func @auto_mutex_alias_root
// CHECK-COUNT-1: tla.mutex "auto_ub_0_1024" {id = 0 : i64} -> !tla.mutex
// CHECK: tla.mutex_lock %{{.*}}[<mte2>] : !tla.mutex
// CHECK-NEXT: tla.copy
// CHECK-NEXT: tla.mutex_unlock %{{.*}}[<mte2>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %{{.*}}[<mte2>] : !tla.mutex
// CHECK-NEXT: tla.copy
// CHECK-NEXT: tla.mutex_unlock %{{.*}}[<mte2>] : !tla.mutex
