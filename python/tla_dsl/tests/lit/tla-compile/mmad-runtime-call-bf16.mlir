// RUN: %tla_compile %s -o - | %filecheck %s

module {
  func.func @mmad_runtime_call_bf16(
      %lhs: !tla.tensor<!tla.layout<!tla.shape<(16,8),(16,4)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,64>, zN>, !tla.coord<0,0>, !tla.ptr<bf16, l0a, 512>>,
      %rhs: !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,8)>, !tla.stride<(1,2048),(16,256)>, !tla.shape<64,128>, nZ>, !tla.coord<0,0>, !tla.ptr<bf16, l0b, 512>>,
      %acc: !tla.tensor<!tla.layout<!tla.shape<(16,8),(16,8)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,128>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>) {
      %init_c = arith.constant true
      %unit_flag = arith.constant 3 : i64
      "tla.mmad"(%acc, %lhs, %rhs, %init_c, %unit_flag)
        : (!tla.tensor<!tla.layout<!tla.shape<(16,8),(16,8)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,128>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>,
           !tla.tensor<!tla.layout<!tla.shape<(16,8),(16,4)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,64>, zN>, !tla.coord<0,0>, !tla.ptr<bf16, l0a, 512>>,
           !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,8)>, !tla.stride<(1,2048),(16,256)>, !tla.shape<64,128>, nZ>, !tla.coord<0,0>, !tla.ptr<bf16, l0b, 512>>, i1, i64) -> ()
    func.return
  }
}

// CHECK-COUNT-1: func.func private @mmad_bf16_bf16_float(
// CHECK-SAME: hacc.always_inline
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SAME: llvm.emit_c_interface
// CHECK-LABEL: func.func @mmad_runtime_call_bf16(
// CHECK-DAG: [[INIT:%.*]] = llvm.mlir.constant(true) : i1
// CHECK-DAG: [[UNIT:%.*]] = llvm.mlir.constant(3 : i8) : i8
// CHECK-DAG: [[MN:%.*]] = llvm.mlir.constant(128 : i64) : i64
// CHECK-DAG: [[K:%.*]] = llvm.mlir.constant(64 : i64) : i64
// CHECK: call @mmad_bf16_bf16_float({{%.*}}, {{%.*}}, {{%.*}}, [[MN]], [[MN]], [[K]], [[INIT]], [[UNIT]])
// CHECK: return
// CHECK-NOT: tla.mmad
