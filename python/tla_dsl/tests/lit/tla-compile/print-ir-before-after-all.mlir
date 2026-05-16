// RUN: %tla_compile %s -o %t --mlir-print-ir-before-all --mlir-print-ir-after-all 2>&1 | %filecheck %s

module {
  func.func @print_ir_smoke() {
    %v = "tla.splat"() {value = 7 : i32} : () -> !tla.value<i32>
    func.return
  }
}

// CHECK: IR Dump Before
// CHECK: IR Dump After
