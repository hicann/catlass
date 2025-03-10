#ifndef ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMV_HPP
#define ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMV_HPP

#include "acot/acot.hpp"
#include "acot/arch/resource.hpp"
#include "acot/epilogue/dispatch_policy.hpp"
#include "acot/gemv_coord.hpp"
#include "acot/matrix_coord.hpp"
#include "acot/layout/layout.hpp"
#include "acot/epilogue/helper.hpp"

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
        tempType_, // fp32
        YType_,    // bf16
        ZType_,    // fp32
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

        using TileElemWiseEpilogueAdd = TileElemWiseEpilogueAdd_;
        using TileElemWiseEpilogueMul = TileElemWiseEpilogueMul_;

        using CopyGmToUbY = typename TileCopy_::CopyGmToUbY;       // 传入y  gm->ub
        using CopyGmToUbTemp = typename TileCopy_::CopyGmToUbTemp; // 传入temp  gm->ub
        using CopyUbToGmZ = typename TileCopy_::CopyUbToGmZ;       // 传出z  ub->gm

        static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogueMul::COMPUTE_LENGTH;
        static constexpr uint32_t OPERANDS_NUM = DispatchPolicy::OPERANDS_NUM;

        static constexpr bool noNeedCast = std::is_same<ElementTemp, ElementY>::value; // 如果是1，说明不需要转换。如果是0，说明需要转换

        // Check the element type of Temp, Y and Z
        // static_assert(std::is_same_v<ElementY, ElementTemp> && std::is_same_v<ElementY, ElementZ>,
        //               "Element type of Y, Temp and Z must be the same");

        // debug,确认了ElementY是bf16类型
        // static_assert(std::is_same_v<ElementY, bfloat16_t>,
        //               "Element type of Y 不是 bf16类型");

        // static_assert(std::is_same_v<ElementY, bfloat16_t>,
        //               "Element type of Y 不是 bf16类型");

        using ElementCompute = typename acot::epilogue::helper::ElementAccumulatorSelector<ElementY, ElementZ>::ElementAccumulator;
        using ElementScalar = ElementCompute;

        // // debug,确认了ElementCompute是float类型
        // static_assert(std::is_same_v<ElementCompute, float>,
        //               "Element type of compute 不是 float");

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
            ubYCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(COMPUTE_LENGTH * sizeof(ElementTemp));

            ubZ = resource.ubBuf.template GetBufferByByte<ElementZ>(
                COMPUTE_LENGTH * sizeof(ElementY) + COMPUTE_LENGTH * sizeof(ElementTemp));
            ubZCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(
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
            MatrixCoord const &blockOffsetMN,
            GemvCoord const &actualBlockShapeMN,
            AscendC::GlobalTensor<ElementCompute> const &gmBlockTemp,
            LayoutTemp const &layoutBlockTemp) // temp通过外部传入
        {                                      // 进行操作，先实现行优先

            MatrixCoord actualBlockShape = actualBlockShapeMN.GetCoordMN();
            MatrixCoord blockOffset = blockOffsetMN;

            // 算出当前子块的offset和shape
            // 对行方向，根据aiv核数划分
            MatrixCoord subblockShape{
                actualBlockShape.row(),
                CeilDiv(actualBlockShape.column(), static_cast<uint32_t>(AscendC::GetSubBlockNum()))};
            MatrixCoord subblockCoord{0, AscendC::GetSubBlockIdx()}; // 子块起始地址坐标就是(aiv核序号，0)

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

            // load Temp(A*x) from gm to ub
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            copyGmToUbTemp(ubTemp, gmSubblockTemp, layoutComputeInUb, layoutSubblockTemp);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            // compute Temp * alpha
            tileEpilogueMul(ubTemp, ubTemp, params.alpha);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

            // load Y from gm to ub
            copyGmToUbY(ubY, gmSubblockY, layoutComputeInUb, layoutSubblockY);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            // compute Y * beta
            if constexpr (!noNeedCast)
            {
                // AscendC::Cast<float, ElementTemp>(ubTemp, ubTemp, AscendC::RoundMode::CAST_NONE, COMPUTE_LENGTH);
                AscendC::Cast<ElementCompute, ElementY>(ubYCast, ubY, AscendC::RoundMode::CAST_NONE, COMPUTE_LENGTH);
                AscendC::PipeBarrier<PIPE_V>();
                tileEpilogueMul(ubYCast, ubYCast, params.beta);
                AscendC::PipeBarrier<PIPE_V>();
            }
            else
            {
                tileEpilogueMul(ubY, ubY, params.beta);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // 加法操作
            if constexpr (!noNeedCast)
            {
                tileEpilogueAdd(ubZCast, ubTemp, ubYCast);
            }
            else
            {
                tileEpilogueAdd(ubZ, ubTemp, ubY);
            }

            // 这里需要判断加法存到位置是ubZ还是ubZCast, 如果在ubZCast, 需要先转类型存回ubZ
            if constexpr (!noNeedCast)
            {
                AscendC::PipeBarrier<PIPE_V>(); // 个人理解
                AscendC::Cast<ElementZ, ElementCompute>(ubZ, ubZCast, AscendC::RoundMode::CAST_RINT, COMPUTE_LENGTH);
                AscendC::PipeBarrier<PIPE_V>(); // 个人理解
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            copyUbToGmZ(gmSubblockZ, ubZ, layoutSubblockZ, layoutComputeInUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        };

    private:
        Params params;

        AscendC::LocalTensor<ElementY> ubY;
        AscendC::LocalTensor<ElementCompute> ubYCast;
        AscendC::LocalTensor<ElementTemp> ubTemp;
        AscendC::LocalTensor<ElementZ> ubZ;
        AscendC::LocalTensor<ElementCompute> ubZCast;

        TileElemWiseEpilogueAdd tileEpilogueAdd;
        TileElemWiseEpilogueMul tileEpilogueMul;

        CopyGmToUbY copyGmToUbY;
        CopyGmToUbTemp copyGmToUbTemp;
        CopyUbToGmZ copyUbToGmZ;
    };

} // namespace acot::epilogue::block

#endif // ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMV_HPP