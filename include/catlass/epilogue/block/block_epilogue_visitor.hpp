#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_VISITOR_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_VISITOR_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Epilogue::Block {

template <
    class CType_,
    class ComputeLength_,
    class ElementCompute_,
    class FusionCallbacks_
>
class BlockEpilogue<
    EpilogueWithVisitorCallbacks,
    CType_,
    ComputeLength_,
    ElementCompute_,
    FusionCallbacks_
> {
public:
    using DispatchPolicy = EpilogueWithVisitorCallbacks;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementD = ElementC;
    using LayoutD = LayoutC;

    static constexpr uint32_t COMPUTE_LENGTH = ComputeLength_::value;
    using ElementCompute = ElementCompute_;
    using FusionCallbacks = FusionCallbacks_;

    struct Params {
        typename FusionCallbacks::Params fusion_params;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(typename FusionCallbacks::Params const& fusion_params_)
            : fusion_params(fusion_params_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params)
        : params(params), fusion_callbacks(params.fusion_params)
    {
        // 初始化事件标志
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        // 等待所有事件完成
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void operator()(
        Arch::Resource<ArchTag>& resource,
        GemmCoord const& blockShapeMNK,
        GemmCoord const& blockCoordMNK,
        GemmCoord const& actualBlockShapeMNK,
        AscendC::GlobalTensor<ElementC> const& gmBlockC,
        layout::RowMajor const& layoutBlockC
    )
    {
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;

        // 子块划分
        MatrixCoord subblockShape{
            CeilDiv(actualBlockShape.row(), static_cast<uint32_t>(AscendC::GetSubBlockNum())),
            actualBlockShape.column()
        };
        MatrixCoord subblockCoord{AscendC::GetSubBlockIdx(), 0};
        MatrixCoord actualSubblockShape = MatrixCoord::Min(
            subblockShape, actualBlockShape - subblockCoord * subblockShape);
        MatrixCoord subblockOffset = subblockCoord * subblockShape;

        // 获取 gmSubblockC 和 layoutSubblockC
        auto gmSubblockC = gmBlockC[layoutBlockC.GetOffset(subblockOffset)];
        auto layoutSubblockC = layoutBlockC.GetTileLayout(actualSubblockShape);

        // 分配 UB 空间并获取 callbacks
        uint32_t ub_offset = 0;
        auto callbacks = fusion_callbacks.get_callbacks(
            resource, ub_offset, COMPUTE_LENGTH,
            blockShapeMNK, blockCoordMNK,
            actualSubblockShape, subblockCoord,
            gmSubblockC, layoutSubblockC
        );

        callbacks.begin_epilogue();

        uint32_t rows = actualSubblockShape.row();
        uint32_t cols = actualSubblockShape.column();

        // 遍历所有 tile（tile 间复用 UB）
        for (uint32_t r = 0; r < rows; ) {
            callbacks.begin_row(r);

            for (uint32_t c = 0; c < cols; ) {
                uint32_t remainCols = cols - c;
                uint32_t tileCols = (remainCols < COMPUTE_LENGTH) ? remainCols : COMPUTE_LENGTH;
                uint32_t rowsCap = (tileCols == 0) ? 0 : (COMPUTE_LENGTH / tileCols);
                if (rowsCap == 0) rowsCap = 1;

                uint32_t remainRows = rows - r;
                uint32_t tileRows = (remainRows < rowsCap) ? remainRows : rowsCap;

                MatrixCoord tileShape{tileRows, tileCols};
                MatrixCoord localTileOffset{r, c};
                // 计算全局绝对坐标
                MatrixCoord globalTileOffset = blockOffset + subblockOffset + localTileOffset;
                uint32_t calCount = tileRows * tileCols;

                // 访问当前 tile
                callbacks.visit(globalTileOffset, tileShape, calCount);

                c += tileCols;
            }

            callbacks.end_row(r);

            // 推进行索引
            uint32_t rowsCapAllCols = (cols == 0) ? 0 : (COMPUTE_LENGTH / ((cols < COMPUTE_LENGTH) ? cols : COMPUTE_LENGTH));
            if (rowsCapAllCols == 0) rowsCapAllCols = 1;
            uint32_t advR = ((rows - r) < rowsCapAllCols) ? (rows - r) : rowsCapAllCols;
            r += advR;
        }

        callbacks.end_epilogue();
    }

private:
    Params params;
    FusionCallbacks fusion_callbacks;
};

} // namespace Catlass::Epilogue::Block

#endif
