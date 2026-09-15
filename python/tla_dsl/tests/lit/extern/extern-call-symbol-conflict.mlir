// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s

module {
  tla.func @conflicting_symbol() {
    %count = arith.constant 1 : i32
    "tla.vector"() ({
      tla.call_extern @conflicting_symbol(%count) : (i32) -> ()
    }) : () -> ()
    tla.return
  }
}

// CHECK: error: 'tla.call_extern' op external symbol @conflicting_symbol conflicts with a defined function
