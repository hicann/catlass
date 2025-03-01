#ifndef ACOT_GEMM_TILE_COPY_L1_TO_L0_HPP
#define ACOT_GEMM_TILE_COPY_L1_TO_L0_HPP

#include "acot/acot.hpp"
#include "acot/layout/layout.hpp"
#include "acot/gemm/gemm_type.hpp"

namespace acot::gemm::tile{
template<
    class ArchTag,
    class GmType
>
struct CopyL1ToL0A{};

template<class Element>
struct CopyL1ToL0A<acot::arch::AscendC910B3, acot::gemm::GemmType<Element, layout::RowMajor>>{
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    ACOT_DEVICE
    CopyL1ToL0A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t MRound = layoutDst.shape(0) * layoutDst.shape(1);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t ML0Alignment = layoutDst.shape(0);
        uint32_t KL0Alignment = layoutDst.shape(2);
        uint32_t MLoops = CeilDiv(MRound, ML0Alignment);
        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(KRound / KL0Alignment);
        params.srcStride = MRound / ML0Alignment;
        params.sid = 0;
        params.dstGap = 0;
        params.ifTranspose = false;
        params.addrMode = 0;
        for(uint32_t i = 0; i < MLoops; i++){
            AscendC::LoadData(dstTensor[i * ML0Alignment * KRound], srcTensor[i * KL0Alignment * ML0Alignment], params);
        }
    }
};

template<class Element>
struct CopyL1ToL0A<acot::arch::AscendC910B3, acot::gemm::GemmType<Element, layout::ColumnMajor>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);

    ACOT_DEVICE
    CopyL1ToL0A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t MActual = layoutDst.orgShape(1);
        uint32_t MRound = RoundUp(MActual, ELE_NUM_PER_C0);
        uint32_t KActual = layoutDst.orgShape(0);
        uint32_t KRound = RoundUp(KActual, C0_NUM_PER_FRACTAL);
        uint32_t KLoops = CeilDiv(KRound, C0_NUM_PER_FRACTAL);
        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(MRound / ELE_NUM_PER_C0); // 单位为512B
        params.srcStride = 1;
        params.sid = 0;
        params.dstGap = 0;
        params.ifTranspose = true;
        params.addrMode = 0;
        for(uint32_t i = 0; i < KLoops; i++){
            AscendC::LoadData(dstTensor[i * C0_NUM_PER_FRACTAL * MRound], srcTensor[i * C0_NUM_PER_FRACTAL * MRound], params);
        }
    }
};

template<>
struct CopyL1ToL0A<acot::arch::AscendC910B3, acot::gemm::GemmType<float, layout::ColumnMajor>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(float);

    ACOT_DEVICE
    CopyL1ToL0A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<float> dstTensor,
        AscendC::LocalTensor<float> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t MActual = layoutDst.orgShape(1);
        uint32_t MRound = RoundUp(MActual, C0_NUM_PER_FRACTAL);
        uint32_t KActual = layoutDst.orgShape(0);
        uint32_t KRound = RoundUp(KActual, C0_NUM_PER_FRACTAL);
        uint32_t ML0Alignment = ELE_NUM_PER_C0 * 2;
        uint32_t KLoops = CeilDiv(KRound, C0_NUM_PER_FRACTAL);
        AscendC::LoadData2dTransposeParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(MRound / ML0Alignment); // 16 * 16 * 4B 为单位
        params.srcStride = 1;
        params.dstGap = 0;
        params.dstFracGap = static_cast<uint16_t>(MRound / ML0Alignment) - 1;
        for(uint32_t i = 0; i < KLoops; i++){ // k方向切割
            AscendC::LoadDataWithTranspose(dstTensor[i * MRound * C0_NUM_PER_FRACTAL], srcTensor[i * C0_NUM_PER_FRACTAL * MRound], params);
        }
    }
};

template<>
struct CopyL1ToL0A<acot::arch::AscendC910B3, acot::gemm::GemmType<int8_t, layout::ColumnMajor>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(int8_t);

    ACOT_DEVICE
    CopyL1ToL0A(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> dstTensor,
        AscendC::LocalTensor<int8_t> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t KAlignment = C0_NUM_PER_FRACTAL * 2;
        uint32_t MActual = layoutDst.orgShape(1);
        uint32_t MRound = RoundUp(MActual, ELE_NUM_PER_C0);
        uint32_t KActual = layoutSrc.orgShape(0); // 这个重要点
        uint32_t KRound = RoundUp(KActual, KAlignment);
        uint32_t KL0Alignment = C0_NUM_PER_FRACTAL * 2; // 32个元素对齐
        uint32_t KLoops = CeilDiv(KRound, KL0Alignment);
        AscendC::LoadData2dTransposeParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(MRound / ELE_NUM_PER_C0); // 单位为32 * 32 * 1B
        params.srcStride = static_cast<uint16_t>(KRound / KL0Alignment); // 对齐单位 32 * 32 * 1B
        params.dstGap = 1; // 单位为512B
        params.dstFracGap = 0;
        for(uint32_t i = 0; i < KLoops; i++){
            AscendC::LoadDataWithTranspose(dstTensor[i * MRound * KL0Alignment],srcTensor[i * KL0Alignment * ELE_NUM_PER_C0],params);
        }
    }
};

// 各种数据类型进行特例化处理
template<
    class ArchTag,
    class GmType
>
struct CopyL1ToL0B{};

template<class Element>
struct CopyL1ToL0B<acot::arch::AscendC910B3, acot::gemm::GemmType<Element, layout::RowMajor>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    ACOT_DEVICE
    CopyL1ToL0B(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t NRound = layoutDst.shape(0) * layoutDst.shape(1);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        // 16 字节的
        uint32_t NL0Alignment = layoutDst.shape(0);
        uint32_t KL0Alignment = layoutDst.shape(2); // 保证了方阵
        uint32_t KLoops = CeilDiv(KRound, KL0Alignment);
        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(NRound / NL0Alignment);
        params.srcStride = 1;
        params.sid = 0;
        params.dstGap = 0;
        params.ifTranspose = true;
        params.addrMode = 0;
        for(uint32_t i = 0; i < KLoops; i++){ // 还是k方向进行切割
            AscendC::LoadData(dstTensor[i * NRound * KL0Alignment], srcTensor[i * NRound * KL0Alignment], params);
        }
    }
};

template<>
struct CopyL1ToL0B<acot::arch::AscendC910B3, acot::gemm::GemmType<float, layout::RowMajor>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(float);

    ACOT_DEVICE
    CopyL1ToL0B(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<float> dstTensor,
        AscendC::LocalTensor<float> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t KAlignment = layoutDst.shape(2);
        uint32_t NRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t KActual = layoutDst.orgShape(0);
        uint32_t KRound = RoundUp(KActual, KAlignment);
        uint32_t NL0Alignment = ELE_NUM_PER_C0 * 2; // 16 原来的NRound就已经向16对齐了
        uint32_t KL0Alignment = layoutDst.shape(2); // 保证了方阵
        uint32_t KLoops = CeilDiv(KRound, KL0Alignment);  // 这里还是向K方向进行切割处理
        AscendC::LoadData2dTransposeParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(NRound / NL0Alignment); // 16 * 16 * 4B 为单位
        params.srcStride = 1;
        params.dstGap = 0;
        params.dstFracGap = static_cast<uint16_t>(NRound / NL0Alignment) - 1;
        for(uint32_t i = 0; i < KLoops; i++){ // k方向切割
            AscendC::LoadDataWithTranspose(dstTensor[i * NRound * KL0Alignment],srcTensor[i * KL0Alignment * NRound],params);
        }
    }
};

template<>
struct CopyL1ToL0B<acot::arch::AscendC910B3, acot::gemm::GemmType<int8_t, layout::RowMajor>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(int8_t);

    ACOT_DEVICE
    CopyL1ToL0B(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> dstTensor,
        AscendC::LocalTensor<int8_t> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t KAlignment = C0_NUM_PER_FRACTAL * 2;
        uint32_t NActual = layoutDst.orgShape(1);
        uint32_t NRound = RoundUp(NActual, ELE_NUM_PER_C0);
        uint32_t KActual = layoutSrc.orgShape(0);
        uint32_t KRound = RoundUp(KActual, KAlignment);
        uint32_t KL0Alignment = C0_NUM_PER_FRACTAL * 2; // 32个元素对齐
        uint32_t KLoops = CeilDiv(KRound, KL0Alignment);
        AscendC::LoadData2dTransposeParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(NRound / ELE_NUM_PER_C0); // 单位为32 * 32 * 1B
        params.srcStride = static_cast<uint16_t>(KRound / KL0Alignment); // 对齐单位 32 * 32 * 1B
        params.dstGap = 1; // 单位为512B
        params.dstFracGap = 0;
        for(uint32_t i = 0; i < KLoops; i++){
            AscendC::LoadDataWithTranspose(dstTensor[i * NRound * KL0Alignment],srcTensor[i * KL0Alignment * ELE_NUM_PER_C0],params);
        }
    }
};

template<class Element>
struct CopyL1ToL0B<acot::arch::AscendC910B3, acot::gemm::GemmType<Element, layout::ColumnMajor>>{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    ACOT_DEVICE
    CopyL1ToL0B(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        uint32_t NRound = layoutDst.shape(0) * layoutDst.shape(1);
        uint32_t KRound = layoutDst.shape(2) * layoutDst.shape(3);
        uint32_t NL0Alignment = layoutDst.shape(0);
        uint32_t KL0Alignment = layoutDst.shape(2);
        uint32_t NLoops = CeilDiv(NRound, NL0Alignment);
        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(KRound / KL0Alignment);
        params.srcStride = NRound / NL0Alignment;
        params.sid = 0;
        params.dstGap = 0;
        params.ifTranspose = false;
        params.addrMode = 0;
        for(uint32_t i = 0; i < NLoops; i++){
            AscendC::LoadData(dstTensor[i * NL0Alignment * KRound], srcTensor[i * NL0Alignment * KL0Alignment], params);
        }
    }
};
}

#endif // ACOT_GEMM_TILE_COPY_L1_TO_L0_HPP