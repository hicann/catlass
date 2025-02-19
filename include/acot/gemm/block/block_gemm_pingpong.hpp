#ifndef ACOT_GEMM_BLOCK_BLOCK_GEMM_PINGPONG_HPP
#define ACOT_GEMM_BLOCK_BLOCK_GEMM_PINGPONG_HPP

#include "acot/acot.hpp"
#include "acot/gemm/helper.hpp"
#include "acot/gemm/tile/tile_copy.hpp"
#include "acot/gemm/tile/tile_mmad.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/gemm/dispatch_policy.hpp"

// 流水修改
namespace acot::gemm::block{
template<
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockGemm<
    GemmAscendC910B3Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
>{
public:
    using DispatchPolicy = GemmAscendC910B3Pingpong<ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyGmToL1A::LayoutDst;
    using LayoutBInL1 = typename CopyGmToL1B::LayoutDst;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES; // 开启双缓冲机制的
    const uint32_t L1Size = ArchTag::L1Size;
    const uint32_t L1ASize = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    const uint32_t L1BSize = L1TileShape::K * L1TileShape::N * sizeof(ElementB);
    const uint32_t cSize = L1TileShape::M * L1TileShape::N * sizeof(ElementAccumulator);
    const uint32_t L0ASize = ArchTag::L0ASize;
    const uint32_t L0BSize = ArchTag::L0BSize;
    const uint32_t L0CSize = ArchTag::L0CSize;
    const uint32_t L0A_PINGPONG_BUF_LEN = (L0ASize / STAGES) / sizeof(ElementA);
    const uint32_t L0B_PINGPONG_BUF_LEN = (L0BSize / STAGES) / sizeof(ElementB);
    static constexpr bool RowOrColumn = std::is_same<LayoutA, layout::RowMajor>::value && std::is_same<LayoutB, layout::RowMajor>::value;

    // construct function
    // make space and sign
    ACOT_DEVICE
    BlockGemm(
        GM_ADDR gmA,
        GM_ADDR gmB,
        GM_ADDR gmC,
        LayoutA layoutA_,
        LayoutB layoutB_,
        LayoutC layoutC_
    ){
        aGm.SetGlobalBuffer((__gm__ ElementA*)gmA); // 强制类型转换
        bGm.SetGlobalBuffer((__gm__ ElementB*)gmB);
        cGm.SetGlobalBuffer((__gm__ ElementC*)gmC);
        layoutA = layoutA_;
        layoutB = layoutB_;
        layoutC = layoutC_;
        Resource();
        // 安排L1空间的流水
        // for(uint32_t i = 0; i < STAGES; i++){
        //     // 使能MTE2搬运单元
        //     AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)i);
        //     AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(i + 2));
        //     // 使能MTE1搬运单元
        //     AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((int32_t)i);
        //     AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((int32_t)(i + 2));
        // }
    }

    // destroy function
    ACOT_DEVICE
    ~BlockGemm(){
        // 安排L0空间的流水
        // for(uint32_t i = 0; i < STAGES; i++){
        //     // 使能MTE1搬运单元
        //     AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((int32_t)i);
        //     AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((int32_t)(i + 2));
        //     // 使能MTE2搬运单元
        //     AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)i);
        //     AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(i + 2));
        // }
        // 释放内存
        pipe.Destroy();
    }

    ACOT_DEVICE
    void operator()(
        uint32_t offsetA,
        uint32_t offsetB,
        uint32_t offsetC,
        MatmulCoord actualShape, // 传递真实矩阵大小
        uint32_t singleIdx
    ){
        // 这一部分完成矩阵乘法
        Matmul(offsetA, offsetB, offsetC, actualShape, singleIdx);
    }
private:
    // 资源利用
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<ElementA> aGm;
    AscendC::GlobalTensor<ElementB> bGm;
    AscendC::GlobalTensor<ElementC> cGm;
    AscendC::TQue<AscendC::TPosition::A1,1> inQueueL1;
    AscendC::TQue<AscendC::TPosition::A2,1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B2,1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1,1> outQueueCO1;
    // 双缓冲区
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;
    // 排布方式
    LayoutA layoutA;
    LayoutB layoutB;
    LayoutC layoutC;
    uint32_t l0CBlockNum;
    
    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;

    // 开辟空间
    ACOT_DEVICE
    void Resource(){
        // 初始化L0C内存空间
        l0CBlockNum = L0CSize / cSize;
        pipe.InitBuffer(outQueueCO1,1,L0CSize);
        l0CTensor = outQueueCO1.AllocTensor<ElementAccumulator>();
        // 初始化L1内存空间
        uint32_t L1ByteStart = 0;
        pipe.InitBuffer(inQueueL1,1,L1Size);
        AscendC::LocalTensor<uint8_t> l1Tensor = inQueueL1.AllocTensor<uint8_t>();
        l1ATensor[0] = l1Tensor[L1ByteStart].template ReinterpretCast<ElementA>();
        L1ByteStart += L1ASize;
        l1ATensor[1] = l1Tensor[L1ByteStart].template ReinterpretCast<ElementA>();
        L1ByteStart += L1ASize;
        l1BTensor[0] = l1Tensor[L1ByteStart].template ReinterpretCast<ElementB>();
        L1ByteStart += L1BSize;
        l1BTensor[1] = l1Tensor[L1ByteStart].template ReinterpretCast<ElementB>();
        L1ByteStart += L1BSize;
        // 初始化L0A / L0B 内存空间
        pipe.InitBuffer(inQueueA2,1,L0ASize);
        pipe.InitBuffer(inQueueB2,1,L0BSize);
        AscendC::LocalTensor<uint8_t> l0TensorA = inQueueA2.AllocTensor<uint8_t>();
        AscendC::LocalTensor<uint8_t> l0TensorB = inQueueB2.AllocTensor<uint8_t>();
        l0ATensor[0] = l0TensorA[0].template ReinterpretCast<ElementA>();
        l0ATensor[1] = l0TensorA[L0A_PINGPONG_BUF_LEN * sizeof(ElementA)].template ReinterpretCast<ElementA>();
        l0BTensor[0] = l0TensorB[0].template ReinterpretCast<ElementB>();
        l0BTensor[1] = l0TensorB[L0B_PINGPONG_BUF_LEN * sizeof(ElementB)].template ReinterpretCast<ElementB>();
    }
    
    ACOT_DEVICE
    void MatmulRowMajor(
        uint32_t offsetA,
        uint32_t offsetB,
        uint32_t offsetC,
        MatmulCoord actualShape, // 传递真实矩阵大小
        uint32_t singleIdx
    ){
        // 先实现行优先
        uint32_t MAlignment = C0_NUM_PER_FRACTAL;
        uint32_t NAlignment = BYTE_PER_C0 / sizeof(ElementB);
        if constexpr (std::is_same<ElementB, float>::value){
            NAlignment = C0_NUM_PER_FRACTAL; // N方向向16对齐
        }
        uint32_t MActual = actualShape.m();
        uint32_t NActual = actualShape.n();
        uint32_t MRound = RoundUp(MActual, MAlignment);
        uint32_t NRound = RoundUp(NActual, NAlignment);
        uint32_t K = actualShape.k();
        uint32_t maxKPerBlock = L1TileShape::K;
        uint32_t KLoops = CeilDiv(K, maxKPerBlock);
        // 进行切分操作
        for(uint32_t KIdx = 0; KIdx < KLoops; KIdx++){
            uint32_t KGmActual = (KIdx == KLoops - 1) ? (K - KIdx * maxKPerBlock) : maxKPerBlock;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(KIdx % STAGES));
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), KGmActual));
            auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(actualShape.m(), KGmActual);
            // 对齐API
            copyGmToL1A(
                l1ATensor[KIdx % STAGES],
                aGm[offsetA + KIdx * maxKPerBlock],
                layoutAInL1, layoutTileA
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((int32_t)(KIdx % STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(KIdx % STAGES + 2));
            auto layoutTileB = layoutB.GetTileLayout(MakeCoord(KGmActual, actualShape.n()));
            auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(KGmActual, actualShape.n());
            // 不同的数据类型，有不同的搬运方式，所以进行特化处理
            copyGmToL1B(
                l1BTensor[KIdx % STAGES],
                bGm[offsetB + KIdx * maxKPerBlock * layoutB.stride(0)],
                layoutBInL1, layoutTileB
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((int32_t)(KIdx % STAGES + 2));

            // 在K方向再进行一次切分处理
            uint32_t KL0TileSize = L0TileShape::K;
            uint32_t KL0Loops = CeilDiv(KGmActual, KL0TileSize);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((int32_t)(KIdx % STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((int32_t)(KIdx % STAGES + 2));
            for(uint32_t KL0Idx = 0; KL0Idx < KL0Loops; KL0Idx++){
                uint32_t KL0Actual = (KL0Idx == KL0Loops - 1) ? (KGmActual - KL0Idx * KL0TileSize) : KL0TileSize;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((int32_t)(KL0Idx % STAGES));
                auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(actualShape.m(), KL0Actual);
                copyL1ToL0A(
                    l0ATensor[KL0Idx % STAGES],
                    l1ATensor[KIdx % STAGES][KL0Idx * KL0TileSize * MRound],
                    layoutAInL0, layoutAInL1
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((int32_t)(KL0Idx % STAGES));
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((int32_t)(KL0Idx % STAGES + 2));
                // 偏移量不一样
                if constexpr (std::is_same<ElementB, int8_t>::value){
                    auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(KGmActual, actualShape.n());
                    copyL1ToL0B(
                        l0BTensor[KL0Idx % STAGES],
                        l1BTensor[KIdx % STAGES][KL0Idx * KL0TileSize * NAlignment],
                        layoutBInL0, layoutBInL1
                    );
                }else{
                    auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(KL0Actual, actualShape.n());
                    copyL1ToL0B(
                        l0BTensor[KL0Idx % STAGES],
                        l1BTensor[KIdx % STAGES][KL0Idx * KL0TileSize * NRound],
                        layoutBInL0, layoutBInL1
                    );
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((int32_t)(KL0Idx % STAGES + 2));
                // 进行计算
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((int32_t)(KL0Idx % STAGES));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((int32_t)(KL0Idx % STAGES + 2));
                tileMmad(
                    l0CTensor[(singleIdx * cSize) % l0CBlockNum],
                    l0ATensor[KL0Idx % STAGES],
                    l0BTensor[KL0Idx % STAGES],
                    MRound,
                    NRound,
                    KL0Actual,
                    (KIdx == 0) && (KL0Idx == 0)
                );
                // if(MActual != MRound || NActual != NRound){
                //     AscendC::PipeBarrier<PIPE_M>();  // 优化点
                // }
                AscendC::PipeBarrier<PIPE_ALL>();  // 优化点
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((int32_t)(KL0Idx % STAGES));
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((int32_t)(KL0Idx % STAGES + 2));
            }
            // 方便下次循环的使用
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(KIdx % STAGES));
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(KIdx % STAGES + 2));
            // if(KIdx == KLoops - 1){
            //     // 最后一个循环节点，进行搬运活动
            //     AscendC::SetFlag<AscendC::HardEvent::M_FIX>((int32_t)-1);
            // }
        }
    }

    ACOT_DEVICE
    void MatmulColumnMajor(
        uint32_t offsetA,
        uint32_t offsetB,
        uint32_t offsetC,
        MatmulCoord actualShape, // 传递真实矩阵大小
        uint32_t singleIdx  
    ){
        uint32_t MAlignment = C0_NUM_PER_FRACTAL; // 对齐32byte
        if constexpr (std::is_same<ElementA, int8_t>::value){
            MAlignment = BYTE_PER_C0 / sizeof(ElementA);
        }
        uint32_t NAlignment = C0_NUM_PER_FRACTAL; // 对齐16行
        uint32_t MActual = actualShape.m();
        uint32_t NActual = actualShape.n();
        uint32_t MRound = RoundUp(MActual, MAlignment);
        uint32_t NRound = RoundUp(NActual, NAlignment);
        uint32_t K = actualShape.k();
        uint32_t maxKPerBlock = L1TileShape::K;
        uint32_t KLoops = CeilDiv(K, maxKPerBlock);
        for(uint32_t KIdx = 0; KIdx < KLoops; KIdx++){
            uint32_t KGmActual = (KIdx == KLoops - 1) ? (K - KIdx * maxKPerBlock) : maxKPerBlock;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(KIdx % STAGES));
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), KGmActual));
            auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(actualShape.m(), KGmActual);
            copyGmToL1A(
                l1ATensor[KIdx % STAGES],
                aGm[offsetA + KIdx * maxKPerBlock * layoutA.stride(1)],
                layoutAInL1, layoutTileA
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((int32_t)(KIdx % STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(KIdx % STAGES + 2));
            auto layoutTileB = layoutB.GetTileLayout(MakeCoord(KGmActual, actualShape.n()));
            auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(KGmActual, actualShape.n());
            copyGmToL1B(
                l1BTensor[KIdx % STAGES],
                bGm[offsetB + KIdx * maxKPerBlock],
                layoutBInL1, layoutTileB
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((int32_t)(KIdx % STAGES + 2));

            uint32_t KL0TileSize = L0TileShape::K;
            uint32_t KL0Loops = CeilDiv(KGmActual, KL0TileSize);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((int32_t)(KIdx % STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((int32_t)(KIdx % STAGES + 2));
            for(uint32_t KL0Idx = 0; KL0Idx < KL0Loops; KL0Idx++){
                uint32_t KL0Actual = (KL0Idx == KL0Loops - 1) ? (KGmActual - KL0Idx * KL0TileSize) : KL0TileSize;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((int32_t)(KL0Idx % STAGES));
                if constexpr (std::is_same<ElementA, int8_t>::value){
                    auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(actualShape.m(), KGmActual);
                    copyL1ToL0A(
                        l0BTensor[KL0Idx % STAGES], // B2硬件位置 nZ
                        l1ATensor[KIdx % STAGES][KL0Idx * KL0TileSize * MAlignment],
                        layoutAInL0, layoutAInL1
                    );
                }else{
                    auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(actualShape.m(), KL0Actual);
                    copyL1ToL0A(
                        l0BTensor[KL0Idx % STAGES],
                        l1ATensor[KIdx % STAGES][KL0Idx * KL0TileSize * MRound],
                        layoutAInL0, layoutAInL1
                    );
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((int32_t)(KL0Idx % STAGES));
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((int32_t)(KL0Idx % STAGES + 2));
                auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(KL0Actual, actualShape.n());
                copyL1ToL0B(
                    l0ATensor[KL0Idx % STAGES],
                    l1BTensor[KIdx % STAGES][KL0Idx * KL0TileSize * NRound],
                    layoutBInL0, layoutBInL1
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((int32_t)(KL0Idx % STAGES + 2));
                // 进行计算
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((int32_t)(KL0Idx % STAGES));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((int32_t)(KL0Idx % STAGES + 2));
                tileMmad(
                    l0CTensor[(singleIdx * cSize) % l0CBlockNum],
                    l0ATensor[KL0Idx % STAGES],
                    l0BTensor[KL0Idx % STAGES],
                    NRound,
                    MRound,
                    KL0Actual,
                    (KIdx == 0) && (KL0Idx == 0)
                );
                // if(MActual != MRound || NActual != NRound){
                //     AscendC::PipeBarrier<PIPE_M>();  // 优化点
                // }
                AscendC::PipeBarrier<PIPE_ALL>();  // 优化点
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((int32_t)(KL0Idx % STAGES));
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((int32_t)(KL0Idx % STAGES + 2));
            }
            // 方便下次循环的使用
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(KIdx % STAGES));
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(KIdx % STAGES + 2));
            // if(KIdx == KLoops - 1){
            //     // 最后一个循环节点，进行搬运活动
            //     AscendC::SetFlag<AscendC::HardEvent::M_FIX>((int32_t)-1);
            // }
        }
    }

    ACOT_DEVICE
    void Matmul(
        uint32_t offsetA,
        uint32_t offsetB,
        uint32_t offsetC,
        MatmulCoord actualShape, // 传递真实矩阵大小
        uint32_t singleIdx
    ){
        for(uint32_t i = 0; i < STAGES; i++){
            // 使能MTE2搬运单元
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)i);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(i + 2));
            // 使能MTE1搬运单元
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((int32_t)i);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((int32_t)(i + 2));
        }
        // auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        // auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(L1TileShape::M, L1TileShape::N)); // 获得MNCoord
        // 不管是行优先还是列优先都是按照K方向进行切割的
        if constexpr (RowOrColumn){ // 想改成结构体  不知道怎么修改
            MatmulRowMajor(offsetA, offsetB, offsetC, actualShape, singleIdx);
        }else{
            MatmulColumnMajor(offsetA, offsetB, offsetC, actualShape, singleIdx);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((int32_t)-1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((int32_t)-1);
        LayoutC layoutBlock = layoutC.GetTileLayout(MakeCoord(actualShape.m(), actualShape.n()));
        copyL0CToGm(
            cGm[offsetC],
            l0CTensor[(singleIdx * cSize) % l0CBlockNum],
            layoutBlock, layoutInL0C
        );
        for(uint32_t i = 0; i < STAGES; i++){
            // 使能MTE1搬运单元
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((int32_t)i);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((int32_t)(i + 2));
            // 使能MTE2搬运单元
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)i);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(i + 2));
        }
    }
};
}

#endif // ACOT_GEMM_BLOCK_BLOCK_GEMM_PINGPONG_HPP