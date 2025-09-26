/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_DISPATCH_POLICY_HPP
#define CATLASS_GEMM_DISPATCH_POLICY_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Gemm {

// Block Mmad Policies

template <bool ASYNC_ = false>
struct MmadAtlasA2Base {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t ASYNC = ASYNC_;
};

using MmadAtlasA2 = MmadAtlasA2Base<false>;
using MmadAtlasA2Async = MmadAtlasA2Base<true>;

// Now ENABLE_UNIT_FLAG_ must be false when intput element is int8
template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2Pingpong : public MmadAtlasA2  {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2PingpongSliceK : public MmadAtlasA2  {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2Preload : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

struct MmadAtlasA2FAQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2FAPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2MLAQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2MLAPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2MLAQKTp1Spec : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2MLAPVTp1Spec : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
    uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2PreloadAsync : public MmadAtlasA2Async {
    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_;  // Stages of emitting load instruction in advance
    static constexpr uint32_t L1_STAGES = L1_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
    uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2PreloadAsyncWithCallback :
    public MmadAtlasA2PreloadAsync<
        PRELOAD_STAGES_,
        L1_STAGES_,
        L0A_STAGES_,
        L0B_STAGES_,
        L0C_STAGES_,
        ENABLE_UNIT_FLAG_,
        ENABLE_SHUFFLE_K_
    > {
};
////////////////////
// new add
template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false, bool ENABLE_ABBA_ = false>
struct GemmAtlasA2 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
    static constexpr bool ENABLE_ABBA = ENABLE_ABBA_;
};

struct GemvAtlasA2 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};
////////////////////

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2PingpongBias : public MmadAtlasA2  {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAIQK : public MmadAtlasA2{
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;

};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAIPV : public MmadAtlasA2{
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAITailQK : public MmadAtlasA2{
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAITailPV : public MmadAtlasA2{
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FullLoadA : public MmadAtlasA2  {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2W8A16 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};


template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2DynamicCommon : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t STAGES_, bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2DynamicSmall : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t STAGES_, bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2DynamicStreamk : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t STAGES_, bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2Small : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t SCALAR_BUFFER_ELE_NUM_ = 256, uint32_t STAGES_ = 2>
struct MmadAtlasA2Aiv : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = STAGES_;
    static constexpr uint32_t SCALAR_BUFFER_ELE_NUM = SCALAR_BUFFER_ELE_NUM_;
};

// UB must have a capacity of UBA(mTile) + UBB(nTile) + UBC(mTile, nTile) size.
// MmadAtlasA2AivSimple is suitable for cases: mTile <= mTileUbLimits && nTile >= params.n
template <uint32_t SCALAR_BUFFER_ELE_NUM_ = 256, bool IS_TILE_M_ = true>
struct MmadAtlasA2AivSimple : public MmadAtlasA2 {
    static constexpr uint32_t SCALAR_BUFFER_ELE_NUM = SCALAR_BUFFER_ELE_NUM_;
    static constexpr bool IS_TILE_M = IS_TILE_M_;
};

// mTile and nTile must be aligned to 16.
// UB must have a capacity of UBA(mTile) + UBB(nTile) + 2 * UBC(mTile, nTile) size.
// MmadAtlasA2AivTrans is suitable for cases: M > N
template <uint32_t SCALAR_BUFFER_ELE_NUM_ = 256>
struct MmadAtlasA2AivTrans : public MmadAtlasA2 {
    static constexpr uint32_t SCALAR_BUFFER_ELE_NUM = SCALAR_BUFFER_ELE_NUM_;
};

}  // namespace Catlass::Gemm

#endif  // CATLASS_GEMM_DISPATCH_POLICY_HPP
