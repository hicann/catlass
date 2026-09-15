// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s < %t

module {
  func.func @pipeline_smoke() {
    func.return
  }
}

// CHECK: module
// CHECK-SAME: {
// CHECK-LABEL: func.func @pipeline_smoke()
// CHECK-SAME: attributes
// CHECK: return
