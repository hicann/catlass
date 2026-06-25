// RUN: %tla_compile %s -o - | %filecheck %s

module {
  func.func @mmad_runtime_call_f32(
      %lhs: !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>,
      %rhs: !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>,
      %acc: !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>) {
      %init_c = arith.constant true
      %unit_flag = arith.constant 3 : i64
      "tla.mmad"(%acc, %lhs, %rhs, %init_c, %unit_flag)
        : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>,
           !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>,
           !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>, i1, i64) -> ()
    func.return
  }
}

// CHECK-COUNT-1: func.func private @mmad_float_float_float(
// CHECK-LABEL: func.func @mmad_runtime_call_f32(
// CHECK-DAG: [[INIT:%.*]] = llvm.mlir.constant(true) : i1
// CHECK-DAG: [[UNIT:%.*]] = llvm.mlir.constant(3 : i8) : i8
// CHECK-DAG: [[MN:%.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK-COUNT-1: hivm.hir.set_ctrl false at ctrl[60]
// CHECK-COUNT-1: hivm.hir.set_ctrl true at ctrl[48]
// CHECK: call @mmad_float_float_float({{%.*}}, {{%.*}}, {{%.*}}, [[MN]], [[MN]], [[MN]], [[INIT]], [[UNIT]])
// CHECK: return
// CHECK-NOT: tla.mmad
