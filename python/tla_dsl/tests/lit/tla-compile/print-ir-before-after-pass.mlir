// RUN: %tla_compile %s -o %t --mlir-print-ir-before=tla-func-to-hacc --mlir-print-ir-after=tla-func-to-hacc 2>&1 | %filecheck %s --check-prefix=HACC
// RUN: %tla_compile %s -o %t --mlir-print-ir-before=tla-lower-to-std --mlir-print-ir-after=tla-lower-to-std 2>&1 | %filecheck %s --check-prefix=LOWER

module {
  func.func @print_ir_pass_selection_smoke() {
    func.return
  }
}

// HACC: IR Dump Before
// HACC-SAME: tla-func-to-hacc
// HACC: IR Dump After
// HACC-SAME: tla-func-to-hacc

// LOWER: IR Dump Before
// LOWER-SAME: tla-lower-to-std
// LOWER: IR Dump After
// LOWER-SAME: tla-lower-to-std
