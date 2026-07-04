// RUN: not %tla_compile %s -o %t 2>&1 | %filecheck %s

module {
  func.func @mmad_runtime_call_type_gate(
      %lhs: !tla.tensor<!tla.layout<!tla.shape<(16,8),(16,4)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>,
      %rhs: !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<64,64>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>,
      %acc: !tla.tensor<!tla.layout<!tla.shape<(16,8),(16,8)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,128>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>) {
      %init_c = arith.constant true
      %unit_flag = arith.constant 3 : i64
      "tla.cube"() ({
        "tla.mmad"(%acc, %lhs, %rhs, %init_c, %unit_flag)
          : (!tla.tensor<!tla.layout<!tla.shape<(16,8),(16,8)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,128>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>,
             !tla.tensor<!tla.layout<!tla.shape<(16,8),(16,4)>, !tla.stride<(16,256),(1,2048)>, !tla.shape<128,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>,
             !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<64,64>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>, i1, i64) -> ()
      }) : () -> ()
    func.return
  }
}

// CHECK: error: unsupported tla.mmad tile shape contract; expected lhs(MxK), rhs(KxN), acc(MxN)
