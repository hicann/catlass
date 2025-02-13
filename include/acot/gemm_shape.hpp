/*
    gemm使用的数据结构，特制gemmshape和L1TileShape
*/
#ifndef ACOT_GEMM_SHAPE_HPP
#define ACOT_GEMM_SHAPE_HPP

#include "acot/acot.hpp"

namespace acot{
struct GemmShape{
    uint32_t M;
    uint32_t N;
    uint32_t K;
    // gemmshape construct function
    ACOT_HOST_DEVICE
    GemmShape(){}
    ACOT_HOST_DEVICE
    GemmShape(uint32_t M_, uint32_t N_, uint32_t K_):M(M_),N(N_),K(K_){}
};

struct L1TileShape{
    uint32_t l1MaxM;
    uint32_t l1MaxN;
    uint32_t l1MaxK;
    
    ACOT_HOST_DEVICE
    L1TileShape(){}
    ACOT_HOST_DEVICE
    L1TileShape(uint32_t l1MaxM_, uint32_t l1MaxN_, uint32_t l1MaxK_):l1MaxM(l1MaxM_), l1MaxN(l1MaxN_), l1MaxK(l1MaxK_){}
};

struct GemmShapeStride{
    uint32_t strideA;
    uint32_t strideB;
    uint32_t strideC;

    ACOT_HOST_DEVICE
    GemmShapeStride(){}
    ACOT_HOST_DEVICE
    GemmShapeStride(uint32_t strideA_, uint32_t strideB_, uint32_t strideC_):strideA(strideA_), strideB(strideB_), strideC(strideC_){}
};

struct Gemm_Kernel2Block_Params{
    uint32_t MActual;
    uint32_t NActual;
    uint32_t MRound;
    uint32_t NRound;
    uint32_t K;
    uint32_t strideA;
    uint32_t strideB;
    uint32_t strideC;

    ACOT_HOST_DEVICE
    Gemm_Kernel2Block_Params(){}
};
}

#endif // ACOT_GEMM_SHAPE_HPP