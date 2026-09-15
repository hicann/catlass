// RUN: %tla_compile %s -o - | %filecheck %s

// A SIMT tla.vec.func outlined out of a *mixed* AIC+AIV function must be an AIV
// function. A pure AIV module gets away without this attribute because
// hivm.module_core_type covers the whole module; a mixed module has no single
// core type, and hivmc-a5 rejects the outlined function with "Unknown core
// type". SIMT is inherently vector-core work, so it must be AIV regardless of
// the enclosing function or pass ordering.

!gm_f32 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>

module {
  tla.func @mixed_cube_simt(%arg0: !gm_f32) {
    %c0 = arith.constant 0 : index
    "tla.cube"() ({
    }) : () -> ()
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %arg0[%c0] : <!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32
        %sum = tla.simt_add %v, %v : f32
        tla.simt_store %arg0[%c0], %sum : <!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
    }) : () -> ()
    tla.return
  }
}

// The vector half of the split keeps the SIMT launch...
// CHECK-LABEL: func.func @mixed_cube_simt_mix_aiv(
// CHECK-SAME:    hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK:         hivm_regbaseintrins.intrins.launch_func @mixed_cube_simt_mix_aiv_vf_simt

// ...and the outlined SIMT function is explicitly AIV, never MIX.
// CHECK-LABEL: func.func @mixed_cube_simt_mix_aiv_vf_simt
// CHECK-SAME:    hivm.func_core_type = #hivm.func_core_type<AIV>
