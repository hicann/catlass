// RUN: %tla_compile %s -o - | %filecheck %s

// Tensor.ptr() + tla.ptr_add (element-count offset) feeding make_tensor /
// make_tensor_like, lowered end-to-end through the cube (TlaLowerToStd) path.
//   gm_off = tile.ptr() + 4   -> make_tensor (row_major GM)
//   l1_off = alloc_ptr + 8    -> make_tensor_like (zN L1)
//   tla.copy(l1_dst, gm_src)  -> copy_gm_row_major_to_cbuf_zN_float
// After lowering, tla.tensor_ptr / tla.ptr_add / tla.make_tensor[_like] are gone and
// the element offsets materialize as memref.reinterpret_cast over the
// hivm_memref_as_ptr-backed base memrefs.

"builtin.module"() ({
  "tla.func"() <{function_type = (!tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "ptr_extract_kernel"}> ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>):
    %0 = "tla.make_shape"() : () -> !tla.shape<16,16>
    %1 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %2 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %3 = "tla.tile_view"(%arg0, %0, %2) : (!tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<16,16>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %4 = "tla.alloc_ptr"() <{size_bytes = 1024 : i64}> : () -> !tla.ptr<i8, l1, 256>
    %5 = "tla.recast_ptr"(%4) : (!tla.ptr<i8, l1, 256>) -> !tla.ptr<f32, l1, 256>
    %6 = "tla.tensor_ptr"(%3) : (!tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> !tla.ptr<f32, gm, 4>
    %7 = "arith.constant"() <{value = 4 : index}> : () -> index
    %8 = "tla.ptr_add"(%6, %7) : (!tla.ptr<f32, gm, 4>, index) -> !tla.ptr<f32, gm, 4>
    %9 = "tla.make_shape"() : () -> !tla.shape<8,8>
    %10 = "tla.make_stride"() : () -> !tla.stride<8,1>
    %11 = "tla.make_layout"(%9, %10) : (!tla.shape<8,8>, !tla.stride<8,1>) -> !tla.layout<!tla.shape<8,8>, !tla.stride<8,1>, !tla.shape<8,8>, row_major>
    %12 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %13 = "tla.make_tensor"(%8, %11, %12) : (!tla.ptr<f32, gm, 4>, !tla.layout<!tla.shape<8,8>, !tla.stride<8,1>, !tla.shape<8,8>, row_major>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<8,8>, !tla.stride<8,1>, !tla.shape<8,8>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %14 = "arith.constant"() <{value = 8 : index}> : () -> index
    %15 = "tla.ptr_add"(%5, %14) : (!tla.ptr<f32, l1, 256>, index) -> !tla.ptr<f32, l1, 256>
    %16 = "tla.make_tensor_like"(%15, %13) <{layoutTag = "zN"}> : (!tla.ptr<f32, l1, 256>, !tla.tensor<!tla.layout<!tla.shape<8,8>, !tla.stride<8,1>, !tla.shape<8,8>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(8,1)>, !tla.stride<(8,128),(1,128)>, !tla.shape<8,8>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 256>>
    "tla.cube"() ({
      "tla.copy"(%16, %13) : (!tla.tensor<!tla.layout<!tla.shape<(16,1),(8,1)>, !tla.stride<(8,128),(1,128)>, !tla.shape<8,8>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 256>>, !tla.tensor<!tla.layout<!tla.shape<8,8>, !tla.stride<8,1>, !tla.shape<8,8>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
// The +4 (GM) and +8 (L1) element offsets are carried as index constants.
// CHECK-LABEL: func.func @ptr_extract_kernel
// CHECK-DAG: llvm.mlir.constant(4 : index)
// CHECK-DAG: llvm.mlir.constant(8 : index)
// L1 alloc-backed pointer (1024 B / 4 = 256 f32) advanced by 8 elements; non-GM keeps
// the flat 1D view (bridged base memref type, 256 f32 = the physical alloc).
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<256xf32, #hivm.address_space<cbuf>>
// CHECK: memref.reinterpret_cast{{.*}} : memref<256xf32, #hivm.address_space<cbuf>> to memref<256xf32, #hivm.address_space<cbuf>>
// GM kernel-arg tile.ptr() + 4: the offset view matches the consuming tensor's
// origin_shape rank (8x8, row-major contiguous strides [8,1]).
// CHECK: memref.reinterpret_cast %arg0 to offset:{{.*}} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>
// The tla-level pointer ops must be fully lowered away.
// CHECK-NOT: "tla.tensor_ptr"
// CHECK-NOT: "tla.ptr_add"
// CHECK-NOT: "tla.make_tensor"
// CHECK-NOT: "tla.make_tensor_like"
// CHECK: call @copy_gm_row_major_to_cbuf_zN_float
