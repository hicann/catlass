// RUN: %tla_compile %s -o - | %filecheck %s

module {
  func.func @mmad_runtime_call_positive(
      %lhs: !tla.tensor<!tla.layout<!tla.shape<(16,8),(16,4)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>,
      %rhs: !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,8)>, !tla.stride<(1,2048),(16,256)>, !tla.shape<64,128>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>,
      %acc: !tla.tensor<!tla.layout<!tla.shape<(16,8),(16,8)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,128>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>) {
    "tla.mmad"(%acc, %lhs, %rhs) {init_c = true}
        : (!tla.tensor<!tla.layout<!tla.shape<(16,8),(16,8)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,128>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>,
           !tla.tensor<!tla.layout<!tla.shape<(16,8),(16,4)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>,
           !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,8)>, !tla.stride<(1,2048),(16,256)>, !tla.shape<64,128>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>) -> ()
    func.return
  }
}

// CHECK-COUNT-1: func.func private @mmad_half_half_float(
// CHECK-SAME: hacc.always_inline
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SAME: llvm.emit_c_interface
// CHECK-LABEL: func.func @mmad_runtime_call_positive(
// CHECK-DAG: [[INIT:%.*]] = arith.constant true
// CHECK-DAG: [[MN_IDX:%.*]] = arith.constant 128 : index
// CHECK-DAG: [[K_IDX:%.*]] = arith.constant 64 : index
// CHECK-DAG: [[MN:%.*]] = arith.index_cast [[MN_IDX]] : index to i64
// CHECK-DAG: [[K:%.*]] = arith.index_cast [[K_IDX]] : index to i64
// CHECK: call @mmad_half_half_float({{%.*}}, {{%.*}}, {{%.*}}, [[MN]], [[MN]], [[K]], [[INIT]], {{%.*}})
// CHECK: return
// CHECK-NOT: tla.mmad
