// Template: copy this file and rename to regression-<bug-id>.mlir
// Example: regression-1234.mlir
//
// Goal:
// - Keep the smallest reproducer for a real bug.
// - Add one clear assertion that proves the fix.
//
// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s < %t
//
// If bug is an expected diagnostic instead:
// RUN: not %tla_compile %s -o %t 2>&1 | %filecheck %s --check-prefix=ERR

module {
  func.func @regression_template() {
    // TODO: Replace with minimal reproducer IR.
    func.return
  }
}

// CHECK-LABEL: func.func @regression_template()
// CHECK: return
//
// For diagnostic style tests:
// ERR: <expected diagnostic substring>
