// RUN: %tla_compile %s -o - | %filecheck %s

module attributes {tla.module_exec_units = "cube"} {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, %arg1: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, %arg2: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>):
    %0 = "tla.flag"() {dst_pipe = #tla.pipe<mte1>, name = "l1_loaded", src_pipe = #tla.pipe<mte2>} : () -> !tla.flag
    %1 = "tla.flag"() {dst_pipe = #tla.pipe<cube>, name = "l0_loaded", src_pipe = #tla.pipe<mte1>} : () -> !tla.flag
    %2 = "tla.flag"() {dst_pipe = #tla.pipe<fix>, name = "mmad_done", src_pipe = #tla.pipe<cube>} : () -> !tla.flag
    %3 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %4 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %5 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %6 = "tla.tile_view"(%arg0, %3, %5) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<32,32>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %7 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %8 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %9 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %10 = "tla.tile_view"(%arg1, %7, %9) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<32,32>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %11 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %12 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %13 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %14 = "tla.tile_view"(%arg2, %11, %13) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<32,32>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %15 = "tla.alloc_ptr"() {size_bytes = 262144 : i64} : () -> !tla.ptr<i8, l1, 512>
    %16 = "tla.recast_ptr"(%15) : (!tla.ptr<i8, l1, 512>) -> !tla.ptr<f32, l1, 512>
    %17 = "tla.alloc_ptr"() {size_bytes = 262144 : i64} : () -> !tla.ptr<i8, l1, 512>
    %18 = "tla.recast_ptr"(%17) : (!tla.ptr<i8, l1, 512>) -> !tla.ptr<f32, l1, 512>
    %19 = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l0a, 512>
    %20 = "tla.recast_ptr"(%19) : (!tla.ptr<i8, l0a, 512>) -> !tla.ptr<f32, l0a, 512>
    %21 = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l0b, 512>
    %22 = "tla.recast_ptr"(%21) : (!tla.ptr<i8, l0b, 512>) -> !tla.ptr<f32, l0b, 512>
    %23 = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l0c, 512>
    %24 = "tla.recast_ptr"(%23) : (!tla.ptr<i8, l0c, 512>) -> !tla.ptr<f32, l0c, 512>
    %25 = "tla.make_tensor_like"(%16, %6) {layoutTag = "zN"} : (!tla.ptr<f32, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    %26 = "tla.make_tensor_like"(%18, %10) {layoutTag = "zN"} : (!tla.ptr<f32, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    "tla.copy"(%25, %6) : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> ()
    "tla.copy"(%26, %10) : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> ()
    "tla.set_flag"(%0) : (!tla.flag) -> ()
    "tla.wait_flag"(%0) : (!tla.flag) -> ()
    %27 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %28 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %29 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %30 = "tla.tile_view"(%25, %27, %29) : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>, !tla.shape<32,32>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    %31 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %32 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %33 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %34 = "tla.tile_view"(%26, %31, %33) : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>, !tla.shape<32,32>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    %35 = "tla.make_tensor_like"(%20, %30) {layoutTag = "zN"} : (!tla.ptr<f32, l0a, 512>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>
    %36 = "tla.make_tensor_like"(%22, %34) {layoutTag = "nZ"} : (!tla.ptr<f32, l0b, 512>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>) -> !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>
    %37 = "tla.make_tensor_like"(%24, %14) {layoutTag = "L0Clayout"} : (!tla.ptr<f32, l0c, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>
    "tla.copy"(%35, %30) : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>) -> ()
    "tla.copy"(%36, %34) : (!tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>) -> ()
    "tla.set_flag"(%1) : (!tla.flag) -> ()
    "tla.wait_flag"(%1) : (!tla.flag) -> ()
    "tla.mmad"(%37, %35, %36) {init_c = true} : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>) -> ()
    "tla.set_flag"(%2) : (!tla.flag) -> ()
    "tla.wait_flag"(%2) : (!tla.flag) -> ()
    "tla.copy"(%14, %37) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>) -> ()
    "tla.pipe_barrier"() {pipe = #tla.pipe<all>} : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "basic_mmad"} : () -> ()
}

// CHECK: func.func private @mmad_float_float_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ca>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cb>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i1, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK: func.func private @copy_cc_to_gm_row_major_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK: func.func private @copy_cbuf_zN_to_cb_nZ_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cb>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK: func.func private @copy_cbuf_zN_to_ca_zN_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ca>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK: func.func private @copy_gm_row_major_to_cbuf_zN_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-LABEL: func.func @basic_mmad(%arg0: memref<32x32xf32, #hivm.address_space<gm>>, %arg1: memref<32x32xf32, #hivm.address_space<gm>>, %arg2: memref<32x32xf32, #hivm.address_space<gm>>)
// CHECK-COUNT-1: hivm.hir.set_ctrl false at ctrl[60]
// CHECK-COUNT-1: hivm.hir.set_ctrl true at ctrl[48]
// CHECK-DAG: hivm.hir.pointer_cast{{.*}}memref<65536xf32, #hivm.address_space<cbuf>>
// CHECK-DAG: hivm.hir.pointer_cast{{.*}}memref<1024xf32, #hivm.address_space<ca>>
// CHECK-DAG: hivm.hir.pointer_cast{{.*}}memref<1024xf32, #hivm.address_space<cb>>
// CHECK-DAG: hivm.hir.pointer_cast{{.*}}memref<1024xf32, #hivm.address_space<cc>>

// CHECK: memref.cast{{.*}}memref<65536xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
// CHECK: memref.cast{{.*}}memref<32x32xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
// CHECK: call @copy_gm_row_major_to_cbuf_zN_float
// CHECK: memref.cast{{.*}}memref<65536xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
// CHECK: memref.cast{{.*}}memref<32x32xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
// CHECK: call @copy_gm_row_major_to_cbuf_zN_float

// CHECK: memref.cast{{.*}}memref<1024xf32, #hivm.address_space<ca>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ca>>
// CHECK: call @copy_cbuf_zN_to_ca_zN_float
// CHECK: memref.cast{{.*}}memref<1024xf32, #hivm.address_space<cb>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cb>>
// CHECK: call @copy_cbuf_zN_to_cb_nZ_float
// CHECK: call @mmad_float_float_float
// CHECK: call @copy_cc_to_gm_row_major_float
// CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
// CHECK-NOT: "tla.mmad"
// CHECK-NOT: "tla.copy"
// CHECK-NOT: "tla.alloc_ptr"
// CHECK-NOT: "tla.recast_ptr"
