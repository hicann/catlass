// RUN: %tla_compile %s -o - | %filecheck %s

// CHECK: module
// CHECK: func.func private @mmad_half_half_float(
// CHECK-SAME: hacc.always_inline
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SAME: llvm.emit_c_interface
// CHECK: func.func @mmad_lit(

module {
  func.func @mmad_lit(
      %lhs: !tla.tensor<!tla.layout<!tla.shape<(16,8),(16,4)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>,
      %rhs: !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,8)>, !tla.stride<(1,2048),(16,256)>, !tla.shape<64,128>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>,
      %acc: !tla.tensor<!tla.layout<!tla.shape<(16,8),(16,8)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,128>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>) {
    %init_c = arith.constant true
    %unit_flag = arith.constant 3 : i64
    "tla.mmad"(%acc, %lhs, %rhs, %init_c, %unit_flag)
        : (!tla.tensor<!tla.layout<!tla.shape<(16,8),(16,8)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,128>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>,
           !tla.tensor<!tla.layout<!tla.shape<(16,8),(16,4)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>,
           !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,8)>, !tla.stride<(1,2048),(16,256)>, !tla.shape<64,128>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>, i1, i64) -> ()
    func.return
  }
}
