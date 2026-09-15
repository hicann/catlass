// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s < %t

module {
  tla.func @main(%arg0: i32) {
    tla.return
  }
}

// CHECK-LABEL: func.func @main(%arg0: i32)
// CHECK: return
// CHECK-NOT: tla.return
// CHECK-NOT: tla.func
