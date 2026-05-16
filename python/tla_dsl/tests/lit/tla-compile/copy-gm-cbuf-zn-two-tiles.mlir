// RUN: %tla_compile %s -o - | %filecheck %s
//
// 与 ``tests/test_framework_overview_interfaces.py`` 中
// ``_kernel_copy_gm_row_major_to_cbuf_zn``（两路 ``tile_view``、单块 L1、两次 ``tla.copy``）
// 对应的 TLA MLIR（``tla`` 方言）及 TlaCompile 管线期望一致；Python 侧仅做 ``dump_mlir`` 断言，golden 以本文件为准。

module {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<200,260>, !tla.stride<88,1>, !tla.shape<72,88>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, %arg1: !tla.tensor<!tla.layout<!tla.shape<100,140>, !tla.stride<92,1>, !tla.shape<56,92>, row_major>, !tla.coord<0,0>, !tla.ptr<i8, gm, 1>>):
    %0 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %1 = "tla.make_coord"() : () -> !tla.coord<1,1>
    %2 = "tla.make_coord"() : () -> !tla.coord<32,32>
    %3 = "tla.tile_view"(%arg0, %0, %2) : (!tla.tensor<!tla.layout<!tla.shape<200,260>, !tla.stride<88,1>, !tla.shape<72,88>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<32,32>, !tla.coord<32,32>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<88,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>
    %4 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %5 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %6 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %7 = "tla.tile_view"(%arg0, %4, %6) : (!tla.tensor<!tla.layout<!tla.shape<200,260>, !tla.stride<88,1>, !tla.shape<72,88>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<32,32>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<88,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %8 = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l1, 512>
    %9 = "tla.recast_ptr"(%8) : (!tla.ptr<i8, l1, 512>) -> !tla.ptr<f32, l1, 512>
    %10 = "tla.make_tensor_like"(%9, %3) {layoutTag = "zN"} : (!tla.ptr<f32, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<88,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    "tla.copy"(%10, %3) : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<88,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>) -> ()
    "tla.copy"(%10, %7) : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<88,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!tla.tensor<!tla.layout<!tla.shape<200,260>, !tla.stride<88,1>, !tla.shape<72,88>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<100,140>, !tla.stride<92,1>, !tla.shape<56,92>, row_major>, !tla.coord<0,0>, !tla.ptr<i8, gm, 1>>) -> (), sym_name = "_kernel_copy_gm_row_major_to_cbuf_zn"} : () -> ()
}

// CHECK: func.func private @copy_gm_row_major_to_cbuf_zN_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK: func.func @_kernel_copy_gm_row_major_to_cbuf_zn(%arg0: memref<200x260xf32, strided<[88, 1], offset: ?>, #hivm.address_space<gm>>, %arg1: memref<100x140xi8, strided<[92, 1], offset: ?>, #hivm.address_space<gm>>)
// CHECK-DAG: hivm.hir.pointer_cast{{.*}}memref<1024xf32, #hivm.address_space<cbuf>>
// CHECK-DAG: memref.cast{{.*}}memref<1024xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
// CHECK-DAG: memref.cast %arg0 : memref<200x260xf32, strided<[88, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
// CHECK-COUNT-2: call @copy_gm_row_major_to_cbuf_zN_float
// CHECK-NOT: "tla.copy"
// CHECK-NOT: "tla.alloc_ptr"
// CHECK-NOT: "tla.recast_ptr"
