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
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
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

            // 检查是否需要列分块
            if (cols <= COMPUTE_LENGTH) {
                // 列宽 <= COMPUTE_LENGTH，可以处理完整行宽
                uint32_t maxRowsPerTile = COMPUTE_LENGTH / cols;
                if (maxRowsPerTile == 0) maxRowsPerTile = 1;  // 防止除零
                
                uint32_t remainRows = rows - r;
                uint32_t tileRows = (remainRows < maxRowsPerTile) ? remainRows : maxRowsPerTile;
                
                MatrixCoord tileShape{tileRows, cols};
                MatrixCoord localTileOffset{r, 0};
                // 计算全局绝对坐标
                MatrixCoord globalTileOffset = blockOffset + subblockOffset + localTileOffset;
                uint32_t calCount = tileRows * cols;

                // 访问当前 tile
                callbacks.visit(globalTileOffset, localTileOffset, tileShape, calCount);
                // AscendC::PipeBarrier<PIPE_ALL>();

                r += tileRows;
            } else { //应该暂时都用不到
                // 列宽 > COMPUTE_LENGTH，需要列分块，每次处理1行
                for (uint32_t c = 0; c < cols; ) {
                    uint32_t remainCols = cols - c;
                    uint32_t tileCols = (remainCols < COMPUTE_LENGTH) ? remainCols : COMPUTE_LENGTH;
                    
                    MatrixCoord tileShape{1, tileCols};
                    MatrixCoord localTileOffset{r, c};
                    // 计算全局绝对坐标
                    MatrixCoord globalTileOffset = blockOffset + subblockOffset + localTileOffset;
                    uint32_t calCount = tileCols;  // 1行 * tileCols列

                    // 访问当前 tile
                    callbacks.visit(globalTileOffset, localTileOffset, tileShape, calCount);
                    // AscendC::PipeBarrier<PIPE_ALL>();

                    c += tileCols;
                }
                
                r += 1;  // 处理完一行
            }

            callbacks.end_row(r);
        }

        callbacks.end_epilogue();
    }

private:
    Params params;
    FusionCallbacks fusion_callbacks;
};

} // namespace Catlass::Epilogue::Block

#endif
