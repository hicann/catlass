// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s < %t

module {
  func.func @pipeline_smoke() {
    %v = "tla.splat"() {value = 42 : i32} : () -> !tla.value<i32>
    func.return
  }
}

// CHECK: module
// CHECK-SAME: {
// CHECK-LABEL: func.func @pipeline_smoke()
// CHECK-SAME: attributes
// CHECK-NOT: tla.splat
// CHECK: return
