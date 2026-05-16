// RUN: not %tla_compile %s -o %t 2>&1 | %filecheck %s --check-prefix=ERR

module {
  func.func @copy_explicit_root_layout_unsupported(
      %src: !tla.memref<64x64xf16, gm>,
      %dst: !tla.memref<64x64xf16, l1>) {
    %sh64 = "tla.make_shape"() : () -> !tla.shape<64,64>
    %cd00 = "tla.make_coord"() : () -> !tla.coord<0,0>

    %src_tile = "tla.tile_view"(%src, %sh64, %cd00) {layouttag = "zN"}
        : (!tla.memref<64x64xf16, gm>, !tla.shape<64,64>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,4)>, !tla.stride<(16,256),(1,1024)>, !tla.shape<64,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>
    %dst_tile = "tla.tile_view"(%dst, %sh64, %cd00)
        : (!tla.memref<64x64xf16, l1>, !tla.shape<64,64>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,4)>, !tla.stride<(16,256),(1,1024)>, !tla.shape<64,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>

    "tla.copy"(%dst_tile, %src_tile)
      : (!tla.tensor<!tla.layout<!tla.shape<(16,4),(16,4)>, !tla.stride<(16,256),(1,1024)>, !tla.shape<64,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,4)>, !tla.stride<(16,256),(1,1024)>, !tla.shape<64,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>) -> ()
    func.return
  }
}

// ERR: error: tla.copy descriptor/layout combination is unsupported: gm(zN) -> l1(zN)
