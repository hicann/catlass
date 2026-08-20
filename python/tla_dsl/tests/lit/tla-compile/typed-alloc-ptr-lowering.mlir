// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-lower-ptr 2>&1 | %filecheck %s

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

// Each address space owns an independent static offset range, so all three
// first allocations lower to byte address zero. Function ABI conversion must
// happen atomically with the pointer producers and call site.
// CHECK-LABEL: func.func @typed_alloc_ptr
// CHECK-COUNT-3: arith.constant {{.*}}0 : i64
// CHECK: call @sink({{.*}}) : (i64, i64, i64) -> ()
// CHECK-NOT: !tla.ptr
// CHECK-NOT: tla.alloc_ptr
// CHECK: func.func private @sink(i64, i64, i64)
