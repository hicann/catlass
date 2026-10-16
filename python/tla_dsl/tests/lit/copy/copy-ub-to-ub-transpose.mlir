// RUN: %tla_compile %s -o - | %filecheck %s
// RUN: sed -e 's/f32/i8/g' -e 's/copy_ub_to_ub_t/copy_ub_to_ub_t_i8/' %s | %tla_compile - -o - | %filecheck %s --check-prefix=CHECK-I8

// tla.copy with UB on both sides and a ColumnMajor destination: a transposing
// copy. It is served by AscendC TransDataTo5HD, which lowers to the vector
// unit's native S.VNCHWCONV transpose. Checked at the IR level: the route must
// reach its own runtime call, distinct from the straight RowMajor->RowMajor one.
// CHECK: copy_ub_RowMajor_to_ub_ColumnMajor_float
//
// VNCHWCONV has a b8 form, so 1-byte element types take the same route.
// CHECK-I8: copy_ub_RowMajor_to_ub_ColumnMajor_int8_t

!src_t = !tla.tensor<!tla.layout<!tla.shape<16, 16>, !tla.stride<16, 1>, !tla.shape<16, 16>, RowMajor>, !tla.coord<0, 0>, !tla.ptr<f32, ub, 4>>
!dst_t = !tla.tensor<!tla.layout<!tla.shape<16, 16>, !tla.stride<1, 16>, !tla.shape<16, 16>, ColumnMajor>, !tla.coord<0, 0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @copy_ub_to_ub_t(
      %src_memref: memref<256xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<256xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %src = tla.tensor_desc %src_memref shape [%c16, %c16, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c16, %c16] coord [%c0, %c0] : memref<256xf32, #hivm.address_space<ub>> -> !src_t
    %dst = tla.tensor_desc %dst_memref shape [%c16, %c16, %c1, %c1] stride [%c1, %c16, %c1, %c1] origin_shape [%c16, %c16] coord [%c0, %c0] : memref<256xf32, #hivm.address_space<ub>> -> !dst_t
    "tla.vector"() ({
      "tla.copy"(%dst, %src) : (!dst_t, !src_t) -> ()
    }) : () -> ()
    return
  }
}
