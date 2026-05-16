// RUN: %tla_compile %s --mlir-print-ir-after=tla-lower-to-std -o %t 2>&1 | %filecheck %s

module {
  func.func @main() {
    %v0 = "tla.splat"() {value = 1.0 : f32} : () -> !tla.value<f32>
    %v1 = "tla.splat"() {value = 7 : i32} : () -> !tla.value<i32>
    func.return
  }
}

// CHECK-LABEL: func.func @main()
// CHECK: arith.constant 1.000000e+00 : f32
// CHECK: arith.constant 7 : i32
// CHECK-NOT: tla.splat
