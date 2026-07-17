// RUN: %tla_compile %s -o - | %filecheck %s
//
// Rank-1 linear make_tensor with a dynamic extent is promoted to rank-2 as
// shape=(1,E), stride=(E*S,S). The leading stride leaf used to be rejected as an
// unbacked dynamic leaf; it must now lower successfully (dead make_tensor may
// be erased, so we only assert the pipeline no longer errors).

module {
  tla.func @make_tensor_rank1_dynamic_extent(%extent: index) {
    %raw = "tla.alloc_ptr"() {size_bytes = 1024 : i64} : () -> !tla.ptr<i8, ub, 256>
    %ptr = "tla.recast_ptr"(%raw) : (!tla.ptr<i8, ub, 256>) -> !tla.ptr<f32, ub, 256>
    %shape = "tla.make_shape"(%extent) : (index) -> !tla.shape<?>
    %stride = "tla.make_stride"() : () -> !tla.stride<1>
    %layout = "tla.make_layout"(%shape, %stride) : (!tla.shape<?>, !tla.stride<1>) -> !tla.layout<!tla.shape<?>, !tla.stride<1>, !tla.shape<?>, row_major>
    %coord = "tla.make_coord"() : () -> !tla.coord<0>
    %tensor = "tla.make_tensor"(%ptr, %layout, %coord) : (!tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<?>, !tla.stride<1>, !tla.shape<?>, row_major>, !tla.coord<0>) -> !tla.tensor<!tla.layout<!tla.shape<?>, !tla.stride<1>, !tla.shape<?>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
    "tla.return"() : () -> ()
  }
}

// CHECK-LABEL: func.func @make_tensor_rank1_dynamic_extent
// CHECK-NOT: derived dynamic leaf
// CHECK-NOT: error:
// CHECK-NOT: tla.make_tensor
