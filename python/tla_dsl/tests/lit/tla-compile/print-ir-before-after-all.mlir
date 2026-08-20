// RUN: %tla_compile %s -o %t --mlir-print-ir-before-all --mlir-print-ir-after-all 2>&1 | %filecheck %s

module {
  func.func @print_ir_smoke() {
    func.return
  }
}

// CHECK: IR Dump Before
// CHECK: IR Dump After
