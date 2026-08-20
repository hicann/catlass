// RUN: %tla_compile %s -o %t --mlir-print-ir-before=tla-lower-func --mlir-print-ir-after=tla-lower-func 2>&1 | %filecheck %s --check-prefix=HACC
// RUN: %tla_compile %s -o %t --mlir-print-ir-before=tla-finalize-memref --mlir-print-ir-after=tla-finalize-memref 2>&1 | %filecheck %s --check-prefix=LOWER

module {
  func.func @print_ir_pass_selection_smoke() {
    func.return
  }
}

// HACC: IR Dump Before
// HACC-SAME: tla-lower-func
// HACC: IR Dump After
// HACC-SAME: tla-lower-func

// LOWER: IR Dump Before
// LOWER-SAME: tla-finalize-memref
// LOWER: IR Dump After
// LOWER-SAME: tla-finalize-memref
