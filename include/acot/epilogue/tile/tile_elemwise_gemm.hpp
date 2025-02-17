#ifndef ACOT_EPILOGUE_TILE_TILE_ELEMWISE_ADD_GEMM_HPP
#define ACOT_EPILOGUE_TILE_TILE_ELEMWISE_ADD_GEMM_HPP

namespace acot::epilogue::tile{
// 包括乘法和加法
template<
    class ArchTag_,
    class ComputeType_,  // 只有有一个数据类型，那就是计算类型
    uint32_t COMPUTE_LENGTH_ // 单块计算个数  是连续计算的
>
struct TileElemWiseMulGemm{
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    ACOT_DEVICE
    TileElemWiseMulGemm(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementCompute> dstLocal,
        AscendC::LocalTensor<ElementCompute> srcTensor,
        ElementCompute scalar // 计算的标量
    ){// Muls 连续模式
        AscendC::Muls(dstLocal, srcTensor, scalar, COMPUTE_LENGTH);
    }
};

template<
    class ArchTag_,
    class ComputeType_,
    uint32_t COMPUTE_LENGTH_
>
struct TileElemWiseAddGemm{
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    ACOT_DEVICE
    TileElemWiseAddGemm(){}

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementCompute> dstTensor,
        AscendC::LocalTensor<ElementCompute> src0Tensor,
        AscendC::LocalTensor<ElementCompute> src1Tensor
    ){// Add 连续模式
        AscendC::Add(dstTensor, src0Tensor, src1Tensor, COMPUTE_LENGTH);
    }
};
}

#endif // ACOT_EPILOGUE_TILE_TILE_ELEMWISE_ADD_GEMM_HPP