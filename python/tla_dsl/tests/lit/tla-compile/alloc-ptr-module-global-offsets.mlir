// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-alloc-ptr-to-hivm-pointer-cast 2>&1 | %filecheck %s

// Also asserts ``tla.hivm_memref_as_ptr`` appears after the alloc pass (same bridge as smaller copy
// fixtures; folded here with the module-wide offset checks to avoid a second near-duplicate lit).

// Two ``func.func`` bodies each allocate the same on-chip scratch size. ``TlaAllocPtrToHivmPointerCastPass``
// keeps one module-wide ``nextOffsetByAddrspace`` so the second kernel's ``pointer_cast`` must use a
// non-zero byte offset (4096 here) after the first 4096-byte reservation.

module {
  func.func @copy_a(%arg0: !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    %0 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %1 = "tla.make_coord"() : () -> !tla.coord<32,32>
    %2 = "tla.tile_view"(%arg0, %0, %1)
        : (!tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<32,32>, !tla.coord<32,32>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>
    %4 = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l1, 512>
    %5 = "tla.recast_ptr"(%4) : (!tla.ptr<i8, l1, 512>) -> !tla.ptr<f32, l1, 512>
    %6 = "tla.make_tensor_like"(%5, %2) {layoutTag = "zN"}
        : (!tla.ptr<f32, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,999)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    "tla.copy"(%6, %2)
        : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,999)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>) -> ()
    func.return
  }
  func.func @copy_b(%arg0: !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    %0 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %1 = "tla.make_coord"() : () -> !tla.coord<32,32>
    %2 = "tla.tile_view"(%arg0, %0, %1)
        : (!tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<32,32>, !tla.coord<32,32>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>
    %4 = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l1, 512>
    %5 = "tla.recast_ptr"(%4) : (!tla.ptr<i8, l1, 512>) -> !tla.ptr<f32, l1, 512>
    %6 = "tla.make_tensor_like"(%5, %2) {layoutTag = "zN"}
        : (!tla.ptr<f32, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,999)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    "tla.copy"(%6, %2)
        : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,999)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>) -> ()
    func.return
  }
}

// CHECK-DAG: arith.constant 4096 : i64
// CHECK-DAG: tla.hivm_memref_as_ptr
// CHECK-DAG: tla.hivm_memref_as_ptr
