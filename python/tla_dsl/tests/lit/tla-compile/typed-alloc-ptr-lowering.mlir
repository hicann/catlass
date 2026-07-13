// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-alloc-ptr-to-hivm-pointer-cast 2>&1 | %filecheck %s

module {
  func.func @typed_alloc_ptr() {
    %0 = "tla.alloc_ptr"() {size_bytes = 512 : i64} : () -> !tla.ptr<f16, l1, 512>
    %1 = "tla.alloc_ptr"() {size_bytes = 512 : i64} : () -> !tla.ptr<f32, ub, 256>
    %2 = "tla.alloc_ptr"() {size_bytes = 128 : i64} : () -> !tla.ptr<i16, l0c, 128>
    func.call @sink(%0, %1, %2) : (!tla.ptr<f16, l1, 512>, !tla.ptr<f32, ub, 256>, !tla.ptr<i16, l0c, 128>) -> ()
    func.return
  }
  func.func private @sink(!tla.ptr<f16, l1, 512>, !tla.ptr<f32, ub, 256>, !tla.ptr<i16, l0c, 128>)
}

// CHECK-LABEL: func.func @typed_alloc_ptr
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<256xf16, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<64xi16, #hivm.address_space<cc>>
// CHECK-NOT: tla.recast_ptr
