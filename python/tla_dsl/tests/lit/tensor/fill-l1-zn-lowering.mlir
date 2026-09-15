// RUN: %tla_compile %s -o - | %filecheck %s
//
// tla.fill on an L1 zN tile lowers to the inlinable AIC runtime template
// fill_l1_zN_<elem>: the call carries the destination's 12-field descriptor,
// the flat rank-2 fill region (coord0, coord1, ext0, ext1, each i64) and the
// i32 bit pattern, then the tla.fill itself disappears. Mirrors the MX mmad's
// K-64 pad zeroing (InitZeroInL1A/B in the C++ block mmad).

module {
  tla.func @fill_l1_zn_fp8() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f8E4M3FN, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(32,2)>
    %stride = tla.make_stride -> !tla.stride<(32,512),(1,1024)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<f8E4M3FN, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l1, 512>>
    %zero = arith.constant 0 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }
}

// The runtime template: f8E4M3FN maps to the fp8_e4m3fn_t suffix, L1 lands on the AIC
// core, and the template is inlinable with a C interface.
// CHECK: func.func private @fill_l1_zN_fp8_e4m3fn_t(memref<?xf8E4M3FN, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32)
// CHECK-SAME: hacc.always_inline
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SAME: llvm.emit_c_interface
// CHECK-LABEL: func.func @fill_l1_zn_fp8()
// The descriptor (4-leaf shape/stride + coord + origin), then the fill region
// (crd0, crd1, ext0, ext1) and the bit pattern, as i64/i32 constants.
// CHECK-DAG: %[[T32:.+]] = llvm.mlir.constant(32 : i64) : i64
// CHECK-DAG: %[[VAL0:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: call @fill_l1_zN_fp8_e4m3fn_t
// CHECK-SAME: %[[T32]], %[[T32]], %[[T32]], %[[VAL0]])
// CHECK-NOT: tla.fill
