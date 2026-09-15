// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s

// CHECK: error: 'tla.vec.func' op cannot outline scalar access because its base memref does not dominate the vector helper call site; materialize dynamic pointer-backed storage outside tla.vec.func
// CHECK-NOT: operand #0 does not dominate this use

!vec64 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>

module {
  "tla.func"() ({
    %address = arith.constant 0 : i64
    %ptr = tla.inttoptr %address : i64 -> !tla.ptr<f32, ub, 256>
    %shape = tla.make_shape -> !tla.shape<64>
    %stride = tla.make_stride -> !tla.stride<1>
    %layout = tla.make_layout %shape, %stride
      : !tla.shape<64>, !tla.stride<1>
        -> !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>
    %coord = tla.make_coord -> !tla.coord<0>
    %tensor = tla.make_tensor %ptr, %layout, %coord
      : !tla.ptr<f32, ub, 256>,
        !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>,
        !tla.coord<0> -> !vec64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    "tla.vector"() ({
      "tla.vec.func"() ({
        // Keep the scalar access before the vector access so its consumer-local
        // base memref is the first helper operand considered.
        %scalar = tla.scalar_load %tensor[%c0] : !vec64 -> f32
        tla.scalar_store %tensor[%c1], %scalar : !vec64, f32
        %vector = tla.load %tensor : !vec64 -> !tla.vector<64xf32>
        tla.store %tensor, %vector : !vec64, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "vector", function_type = () -> (), sym_name = "scalar_access_helper_dominance"} : () -> ()
}
