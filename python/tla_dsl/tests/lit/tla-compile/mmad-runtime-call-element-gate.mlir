// RUN: not %tla_compile %s -o %t 2>&1 | %filecheck %s

module {
  func.func @mmad_runtime_call_element_gate(
      %lhs: !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>,
      %rhs: !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>,
      %acc: !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>) {
      %init_c = arith.constant true
      %unit_flag = arith.constant 3 : i64
      "tla.mmad"(%acc, %lhs, %rhs, %init_c, %unit_flag)
        : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>,
           !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>,
           !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>, i1, i64) -> ()
    func.return
  }
}

// CHECK: error: unsupported tla.mmad element types; expected f16,f16 -> f32, bf16,bf16 -> f32, or f32,f32 -> f32 (L0C accumulator is fp32)
