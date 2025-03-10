#ifndef ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMV_HPP
#define ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMV_HPP

#include "acot/acot.hpp"
#include "acot/arch/resource.hpp"
#include "acot/epilogue/dispatch_policy.hpp"
#include "acot/gemv_coord.hpp"
#include "acot/matrix_coord.hpp"
#include "acot/layout/layout.hpp"

namespace acot::epilogue::block
{
    template <
        class tempType_,                // y 向量(矩阵表示)
        class YType_,                   // A * x向量(矩阵表示)
        class ZType_,                   // z向量(矩阵表示)
        class TileElemWiseEpilogueAdd_, // 单块元素的处理，包括加法和乘法
        class TileElemWiseEpilogueMul_, // 乘法的后处理
        class TileCopy_>
    class BlockEpilogue<
        EpilogueAtlasA2Gemv,
        tempType_,
        YType_,
        ZType_,
        TileElemWiseEpilogueAdd_,
        TileElemWiseEpilogueMul_,
        TileCopy_>
    {
    public:
        // Type aliases
        // temp = A*x
        // Z = α * temp + β * Y
        using DispatchPolicy = EpilogueAtlasA2Gemv;
        using ArchTag = typename DispatchPolicy::ArchTag;

        using ElementTemp = typename tempType_::Element;
        using LayoutTemp = typename tempType_::Layout;
        using ElementY = typename YType_::Element;
        using LayoutY = typename YType_::Layout;
        using ElementZ = typename ZType_::Element;
        using LayoutZ = typename ZType_::Layout;
        using ElementScalar = typename YType_::Element;

        using TileElemWiseEpilogueAdd = TileElemWiseEpilogueAdd_;
        using TileElemWiseEpilogueMul = TileElemWiseEpilogueMul_;

        using CopyGmToUbY = typename TileCopy_::CopyGmToUbY;       // 传入y  gm->ub
        using CopyGmToUbTemp = typename TileCopy_::CopyGmToUbTemp; // 传入temp  gm->ub
        using CopyUbToGmZ = typename TileCopy_::CopyUbToGmZ;       // 传出z  ub->gm

        static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogueMul::COMPUTE_LENGTH;
        static constexpr uint32_t OPERANDS_NUM = DispatchPolicy::OPERANDS_NUM;

        // Check the element type of Temp, Y and Z
        static_assert(std::is_same_v<ElementY, ElementTemp> && std::is_same_v<ElementY, ElementZ>,
                      "Element type of Y, Temp and Z must be the same");
        using ElementCompute = ElementZ;

        // Check the layout type of Y, Temp and Z
        static_assert(std::is_same_v<LayoutY, layout::RowMajor> && std::is_same_v<LayoutTemp, layout::RowMajor> &&
                          std::is_same_v<LayoutZ, layout::RowMajor>,
                      "Layout type of Y, Temp and Z must be RowMajor");
        using LayoutComputeInUb = layout::RowMajor;

        // Check if ArchTag is matched
        static_assert(std::is_same_v<typename TileElemWiseEpilogueMul::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
        // Check if compute length is valid
        static_assert(COMPUTE_LENGTH * OPERANDS_NUM * sizeof(ElementCompute) <= ArchTag::UB_SIZE, "UB out of bounds");

        struct Params
        {
            ElementScalar alpha;
            ElementScalar beta;
            GM_ADDR ptrY;
            LayoutY layoutY;

            GM_ADDR ptrZ;
            LayoutZ layoutZ;

            // Methods
            ACOT_DEVICE
            Params() {}

            ACOT_DEVICE
            Params(ElementScalar alpha_, ElementScalar beta_, GM_ADDR ptrY_, LayoutTemp layoutY_, GM_ADDR ptrZ_, LayoutZ layoutZ_) : alpha(alpha_), beta(beta_), ptrY(ptrY_), layoutY(layoutY_), ptrZ(ptrZ_), layoutZ(layoutZ_) {}
        };

        ACOT_DEVICE
        BlockEpilogue(arch::Resource<ArchTag> &resource, Params const &params) : params(params) // 对应temp = Ax，不分配空间, 作为参数在operator()中作为参数输入
        {
            ubTemp = resource.ubBuf.template GetBufferByByte<ElementTemp>(0);
            ubY = resource.ubBuf.template GetBufferByByte<ElementY>(COMPUTE_LENGTH * sizeof(ElementTemp));
            ubZ = resource.ubBuf.template GetBufferByByte<ElementZ>(
                COMPUTE_LENGTH * sizeof(ElementY) + COMPUTE_LENGTH * sizeof(ElementTemp));

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        ACOT_DEVICE
        ~BlockEpilogue()
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        ACOT_DEVICE
        void operator()(
            GemvCoord const &blockShapeMN,
            GemvCoord const &blockCoordMN,
            GemvCoord const &actualBlockShapeMN,
            AscendC::GlobalTensor<ElementCompute> const &gmBlockTemp,
            LayoutY const &layoutBlockTemp) // temp通过外部传入
        {                                   // 进行操作，先实现行优先

            // Calculate the offset of the current block
            MatrixCoord blockShape = blockShapeMN.GetCoordMN();
            MatrixCoord blockCoord = blockCoordMN.GetCoordMN();
            MatrixCoord actualBlockShape = actualBlockShapeMN.GetCoordMN();
            MatrixCoord blockOffset = blockCoord * blockShape; // coord级别的乘法，算出行偏移和列偏移

            // 算出当前子块的offset和shape
            // 对行方向，根据aiv核数划分
            MatrixCoord subblockShape{
                actualBlockShape.row(),
                CeilDiv(actualBlockShape.column(), static_cast<uint32_t>(AscendC::GetSubBlockNum()))};

            // MatrixCoord subblockShape{
            //     actualBlockShape.column(),
            //     CeilDiv(actualBlockShape.row(), static_cast<uint32_t>(AscendC::GetSubBlockNum()))};

            // MatrixCoord subblockCoord{AscendC::GetSubBlockIdx(), 0}; // 子块起始地址坐标就是(aiv核序号，0)
            MatrixCoord subblockCoord{0, AscendC::GetSubBlockIdx()};

            MatrixCoord actualSubblockShape = MatrixCoord::Min(subblockShape, actualBlockShape - subblockCoord * subblockShape); // 这里实际上就是获取每个subBlockIdx对应的blockshape
            MatrixCoord subblockOffset = subblockCoord * subblockShape;                                                          // coord级别的乘法，算出行偏移和列偏移

            // Get the data and layout of Temp 外部输入 子偏移
            auto gmSubblockTemp = gmBlockTemp[layoutBlockTemp.GetOffset(subblockOffset)];
            auto layoutSubblockTemp = layoutBlockTemp.GetTileLayout(actualSubblockShape);

            // Get the data and layout of y 全局的原始数据  全局偏移
            AscendC::GlobalTensor<ElementY> gmY;
            gmY.SetGlobalBuffer(reinterpret_cast<__gm__ ElementY *>(params.ptrY));
            auto gmSubblockY = gmY[params.layoutY.GetOffset(blockOffset + subblockOffset)];
            auto layoutSubblockY = params.layoutY.GetTileLayout(actualSubblockShape);

            // Get the data and layout of Z 全局的原始数据
            AscendC::GlobalTensor<ElementZ> gmZ;
            gmZ.SetGlobalBuffer(reinterpret_cast<__gm__ ElementZ *>(params.ptrZ));
            auto gmSubblockZ = gmZ[params.layoutZ.GetOffset(blockOffset + subblockOffset)];
            auto layoutSubblockZ = params.layoutZ.GetTileLayout(actualSubblockShape);

            // get the layout on UB
            auto layoutComputeInUb = LayoutComputeInUb::template MakeLayoutInUb<ElementCompute>(actualSubblockShape); // 对列方向进行了32B对齐

            // copy the data of Y and Temp
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::DataCopyExtParams dataCopyParams0(
                layoutSubblockY.shape(0),
                layoutSubblockY.shape(1) * sizeof(ElementCompute),
                0,
                0,
                0);
            AscendC::DataCopyPadExtParams<ElementCompute> padParams(false, 0, 0, 0);
            AscendC::DataCopyPad(ubY, gmSubblockY, dataCopyParams0, padParams);

            AscendC::DataCopyExtParams dataCopyParams1(
                layoutSubblockTemp.shape(0),
                layoutSubblockTemp.shape(1) * sizeof(ElementCompute),
                0,
                0,
                0);
            // AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
            AscendC::DataCopyPad(ubTemp, gmSubblockTemp, dataCopyParams1, padParams);
            // copyGmToUbY(ubY, gmSubblockY, layoutComputeInUb, layoutSubblockY);
            // AscendC::PipeBarrier<PIPE_MTE2>();
            // copyGmToUbTemp(ubTemp, gmSubblockTemp, layoutComputeInUb, layoutSubblockTemp);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            // 同时算β * Y 和 α * Temp
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubY, ubY, params.beta, actualBlockShape.column());
            // tileEpilogueMul(ubY, ubY, params.beta);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubTemp, ubTemp, params.alpha, actualBlockShape.column());
            // tileEpilogueMul(ubTemp, ubTemp, params.alpha);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(ubZ, ubTemp, ubY, actualBlockShape.column());
            // tileEpilogueAdd(ubZ, ubTemp, ubY);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // copy the data of Z
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyExtParams dataCopyParams2(
                layoutSubblockZ.shape(0),
                layoutSubblockZ.shape(1) * sizeof(ElementCompute),
                0,
                0,
                0);
            AscendC::DataCopyPad(gmSubblockZ, ubZ, dataCopyParams2);
            // copyUbToGmZ(gmSubblockZ, ubZ, layoutSubblockZ, layoutComputeInUb);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        };

    private:
        Params params;

        AscendC::LocalTensor<ElementY> ubY;
        AscendC::LocalTensor<ElementTemp> ubTemp;
        AscendC::LocalTensor<ElementZ> ubZ;

        // TileElemWiseEpilogue tileEpilogue;

        TileElemWiseEpilogueAdd tileEpilogueAdd;
        TileElemWiseEpilogueMul tileEpilogueMul;

        CopyGmToUbY copyGmToUbY;
        CopyGmToUbTemp copyGmToUbTemp;
        CopyUbToGmZ copyUbToGmZ;
    };

} // namespace acot::epilogue::block

#endif // ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMV_HPP