// RUN: not %tla_compile %s -o %t 2>&1 | %filecheck %s

module {
  tla.func @mixed() {
    "tla.cube"() ({
    }) : () -> ()
    "tla.vector"() ({
    }) : () -> ()
    tla.return
  }
  func.func @mixed_mix_aic() {
    func.return
  }
}

// CHECK: error: 'func.func' op cannot split mixed function because symbol @mixed_mix_aic already exists
// CHECK: Failed to run Tla pipeline.
