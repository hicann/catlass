#ifndef ACOT_EPILOGUE_TILE_TILE_ELEMWISE_ADD_GEMV_HPP
#define ACOT_EPILOGUE_TILE_TILE_ELEMWISE_ADD_GEMV_HPP

namespace acot::epilogue::tile
{
    // 包括乘法和加法
    template <
        class ArchTag_,
        class ComputeType_,      // 只有有一个数据类型，那就是计算类型
        uint32_t COMPUTE_LENGTH_ // 单块计算个数  是连续计算的
        >
    struct TileElemWiseMulGemv
    {
        using ArchTag = ArchTag_;
        using ElementCompute = typename ComputeType_::Element;

        static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

        ACOT_DEVICE
        TileElemWiseMulGemv() {}

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<ElementCompute> const &ubOut,
            AscendC::LocalTensor<ElementCompute> const &ubIn,
            ElementCompute scalar // 计算的标量
        )
        { // Muls 连续模式
            AscendC::Muls(ubOut, ubIn, scalar, COMPUTE_LENGTH);
        }
    };

    template <
        /// Tag indicating architecture
        class ArchTag_,
        /// Compute data type
        class ComputeType_,
        /// Length of the compute buffer
        uint32_t COMPUTE_LENGTH_>
    struct TileElemWiseAddGemv
    {
        using ArchTag = ArchTag_;
        using ElmentCompute = typename ComputeType_::Element;

        static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

        ACOT_DEVICE
        TileElemWiseAddGemv() {}

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<ElmentCompute> const &ubOut,
            AscendC::LocalTensor<ElmentCompute> const &ubIn0,
            AscendC::LocalTensor<ElmentCompute> const &ubIn1)
        {
            // Do the calculation
            AscendC::Add(ubOut, ubIn0, ubIn1, COMPUTE_LENGTH);
        }
    };
}

#endif // ACOT_EPILOGUE_TILE_TILE_ELEMWISE_ADD_GEMV_HPP