#ifndef ACOT_GEMM_BLOCK_BLOCK_GEMM_HPP
#define ACOT_GEMM_BLOCK_BLOCK_GEMM_HPP

#include "acot/acot.hpp"
#include "acot/gemm/helper.hpp"
#include "acot/gemm/tile/tile_copy.hpp"
#include "acot/gemm/tile/tile_mmad.hpp"
#include "acot/gemm_shape.hpp"

namespace acot::gemm::block{
template<
    class ArchTag_,
    class AType_,
    class BType_,
    class CType_,
    uint32_t l1MaxM,
    uint32_t l1MaxN,
    uint32_t l1MaxK,
    class TileCopy_ = acot::gemm::tile::TileCopy<ArchTag_, AType_, BType_, CType_>,
    class TileMmad_ = acot::gemm::tile::TileMmad<ArchTag_, AType_, BType_, CType_>
>
struct BlockGemm{
public:
    using ArchTag = ArchTag_;
    acot::L1TileShape l1TileShape{l1MaxM, l1MaxN, l1MaxK}; // 设置成员变量

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
    
    static constexpr uint32_t STAGES = 2; // 开启双缓冲机制
    const uint32_t L1Size = ArchTag::L1Size;
    const uint32_t L1ASize = l1TileShape.l1MaxM * l1TileShape.l1MaxK * sizeof(ElementA);
    const uint32_t L1BSize = l1TileShape.l1MaxK * l1TileShape.l1MaxN * sizeof(ElementB);
    const uint32_t cSize = l1TileShape.l1MaxM * l1TileShape.l1MaxN * sizeof(ElementAccumulator);
    const uint32_t L0ASize = ArchTag::L0ASize;
    const uint32_t L0BSize = ArchTag::L0BSize;
    const uint32_t L0CSize = ArchTag::L0CSize;
    const uint32_t L0A_PINGPONG_BUF_LEN = (L0ASize / STAGES) / sizeof(ElementA);
    const uint32_t L0B_PINGPONG_BUF_LEN = (L0BSize / STAGES) / sizeof(ElementB);

    // construct function
    // make space and sign
    ACOT_DEVICE
    BlockGemm(
        __gm__ ElementA* gmA,
        __gm__ ElementB* gmB,
        __gm__ ElementC* gmC
    ){
        aGm.SetGlobalBuffer((__gm__ ElementA*)gmA); // 强制类型转换
        bGm.SetGlobalBuffer((__gm__ ElementB*)gmB);
        cGm.SetGlobalBuffer((__gm__ ElementC*)gmC);
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
        // 安排L1空间的流水
        for(uint32_t i = 0; i < STAGES; i++){
            // 使能MTE2搬运单元
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)i);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(i + 2));
        }
        // 安排L0空间的流水
        for(uint32_t i = 0; i < STAGES; i++){
            // 使能MTE1搬运单元
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((int32_t)i);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((int32_t)(i + 2));
        }
    }

    // destroy function
    ACOT_DEVICE
    ~BlockGemm(){
        // 安排L0空间的流水
        for(uint32_t i = 0; i < STAGES; i++){
            // 使能MTE1搬运单元
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((int32_t)i);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((int32_t)(i + 2));
        }
        // 安排L1空间的流水
        for(uint32_t i = 0; i < STAGES; i++){
            // 使能MTE2搬运单元
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)i);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(i + 2));
        }
        // 释放内存
        pipe.Destroy();
    }

    ACOT_DEVICE
    void operator()(
        uint32_t offsetA,
        uint32_t offsetB,
        uint32_t offsetC,
        Gemm_Kernel2Block_Params kernelParams,
        uint32_t singleIdx
    ){
        if constexpr (std::is_same<LayoutA, acot::layout::RowMajor>::value && std::is_same<LayoutB, acot::layout::RowMajor>::value){
            // 单纯的行优先代码
            // 搬运函数还得修改，尽量把for循环放到函数中 for循环尽量只展示切分过程
            uint32_t KL1AlignmentB;
            if constexpr (std::is_same<ElementB, int8_t>::value){
                KL1AlignmentB = C0_NUM_PER_FRACTAL * 2;
            }else{
                KL1AlignmentB = C0_NUM_PER_FRACTAL;
            }
            uint32_t KL1AlignmentA = BYTE_PER_C0 / sizeof(ElementA);
            uint32_t K = kernelParams.K;
            uint32_t maxKPerBlock = l1TileShape.l1MaxK;
            uint32_t KLoops = CeilDiv(K, maxKPerBlock);

            // 这一块还缺个信号量
            for(uint32_t kIdx = 0; kIdx < KLoops; kIdx++){
                uint32_t KGmActual = (kIdx == KLoops - 1) ? (K - kIdx * maxKPerBlock) : maxKPerBlock;
                uint32_t KGmRoundB = RoundUp(KGmActual,KL1AlignmentB); // 第一阶段统一向16对齐
                uint32_t KGmRoundA = RoundUp(KGmActual,KL1AlignmentA);
                // 先抢MTE2信号量
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(kIdx % STAGES)); 
                copyGmToL1A( // 这里肯定填充到了32Byte的倍数
                    aGm[offsetA + kIdx * maxKPerBlock],
                    l1ATensor[kIdx % STAGES],
                    kernelParams.MActual,
                    kernelParams.MRound,
                    KGmActual,
                    KGmRoundA,
                    kernelParams.strideA
                );
                // AscendC::PipeBarrier<PIPE_ALL>();
                // 使能MTE1
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((int32_t)(kIdx % STAGES));
                // Gm -> L1B
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(kIdx % STAGES + 2));
                // 综合来看，写一个函数，根据数据类型来特例化
                copyGmToL1B(
                    bGm[offsetB + kIdx * maxKPerBlock * kernelParams.strideB],
                    l1BTensor[kIdx % STAGES],
                    KGmActual,
                    KGmRoundB,
                    kernelParams.NActual,
                    kernelParams.NRound,
                    kernelParams.strideB
                );
                // AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((int32_t)(kIdx % STAGES + 2));

                uint32_t KL0Alignment; // A矩阵FP32是8对齐 这边统一对齐16对齐 16的倍数就是8的倍数
                if constexpr (std::is_same<ElementB, int8_t>::value){
                    KL0Alignment = C0_NUM_PER_FRACTAL * 2;
                }else{
                    KL0Alignment = C0_NUM_PER_FRACTAL;
                }
                // 对K方向进行进一步的切分
                uint32_t kL0 = min( // 这里进行对齐操作
                    RoundDown(L0A_PINGPONG_BUF_LEN / kernelParams.MRound, KL0Alignment),
                    RoundDown(L0B_PINGPONG_BUF_LEN / kernelParams.NRound, KL0Alignment)
                );
                uint32_t KL0AlignmentA = BYTE_PER_C0 / sizeof(ElementA);
                uint32_t KL0AlignmentB;
                if constexpr (std::is_same<ElementB, int8_t>::value){
                    KL0AlignmentB = C0_NUM_PER_FRACTAL * 2;
                }else{
                    KL0AlignmentB = C0_NUM_PER_FRACTAL;
                }
                uint32_t NL0Alignment = BYTE_PER_C0 / sizeof(ElementB);
                uint32_t KL0Loops = CeilDiv(KGmActual, kL0);
                // 进行下一个循环
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((int32_t)(kIdx % STAGES));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((int32_t)(kIdx % STAGES + 2));
                for(uint32_t kL0Idx = 0; kL0Idx < KL0Loops; kL0Idx++){ // 这边建议统一进行切分处理
                    uint32_t KL0Actual = (kL0Idx == KL0Loops - 1) ? (KGmActual - kL0Idx * kL0) : kL0;
                    uint32_t KL0ARound = RoundUp(KL0Actual, KL0AlignmentA);
                    uint32_t KL0BRound = RoundUp(KL0Actual, KL0AlignmentB);
                    // 进行搬运工作 L1 -> L0A
                    // 先看dst有信号量吗
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((int32_t)(kL0Idx % STAGES));
                    copyL1ToL0A( // 这个应该没问题
                        l1ATensor[kIdx % STAGES][kL0Idx * kL0 * kernelParams.MRound],
                        l0ATensor[kL0Idx % STAGES],
                        kernelParams.MActual,
                        kernelParams.MRound,
                        KL0Actual,
                        KL0ARound
                    );
                    // 使能计算单元
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((int32_t)(kL0Idx % STAGES));
                    // L1 -> L0B 
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((int32_t)(kL0Idx % STAGES + 2));
                    // 数据类型传参特定的
                    if constexpr (std::is_same<ElementB, int8_t>::value){
                        copyL1ToL0B(
                            l1BTensor[kIdx % STAGES][kL0Idx * kL0 * NL0Alignment], // B矩阵是zN排布方式 注意K和N维度的
                            l0BTensor[kL0Idx % STAGES],
                            kernelParams.NActual,
                            kernelParams.NRound,
                            KL0Actual,
                            KGmRoundB // 这个参数注意  这个参数十分重要
                        );
                    }else{
                        copyL1ToL0B( // 这里注意要转置了 需要判断了
                            l1BTensor[kIdx % STAGES][kL0Idx * kL0 * kernelParams.NRound],
                            l0BTensor[kL0Idx % STAGES],
                            kernelParams.NActual,
                            kernelParams.NRound,
                            KL0Actual,
                            KL0BRound
                        );
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((int32_t)(kL0Idx % STAGES + 2));
                    // 进行计算
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((int32_t)(kL0Idx % STAGES));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((int32_t)(kL0Idx % STAGES + 2));
                    tileMmad(
                        l0CTensor[(singleIdx * cSize) % l0CBlockNum],
                        l0ATensor[kL0Idx % STAGES],
                        l0BTensor[kL0Idx % STAGES],
                        kernelParams.MRound,
                        kernelParams.NRound,
                        KL0Actual,
                        (kIdx == 0) && (kL0Idx == 0)
                    );
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((int32_t)(kL0Idx % STAGES));
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((int32_t)(kL0Idx % STAGES + 2));
                }
                // 方便下次循环的使用
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(kIdx % STAGES));
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((int32_t)(kIdx % STAGES + 2));
                if(kIdx == KLoops - 1){
                    // 最后一个循环节点，进行搬运活动
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>((int32_t)-1);
                }
            }
            // 上面计算完毕 L0C -> GM
            // 不能先搬走了 抢FIX搬运单元
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((int32_t)-1);
            copyL0CToGm(
                cGm[offsetC],
                l0CTensor[(singleIdx * cSize) % l0CBlockNum],
                kernelParams.MRound,
                kernelParams.NRound,
                kernelParams.MActual,
                kernelParams.NActual,
                kernelParams.strideC
            );
        }
    }
private:
    // 资源利用
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<ElementA> aGm;
    AscendC::GlobalTensor<ElementB> bGm;
    AscendC::GlobalTensor<ElementC> cGm;
    AscendC::TQue<AscendC::QuePosition::A1,1> inQueueL1;
    AscendC::TQue<AscendC::QuePosition::A2,1> inQueueA2;
    AscendC::TQue<AscendC::QuePosition::B2,1> inQueueB2;
    AscendC::TQue<AscendC::QuePosition::CO1,1> outQueueCO1;
    // 双缓冲区
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    uint32_t l0CBlockNum;
    
    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};
}

#endif // ACOT_GEMM_BLOCK_BLOCK_GEMM_HPP