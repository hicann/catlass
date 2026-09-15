// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s < %t

module {
  tla.func @main() {
    tla.return
  }
}

// CHECK-LABEL: func.func @main()
// CHECK: return
// CHECK-NOT: tla.return
// CHECK-NOT: tla.func
