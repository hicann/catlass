// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
// Host-only check: g++ -std=c++17 x_attention_tiling_contract.cpp -o /tmp/xattention-contract
#include <cstdint>

// Only erase device annotations/types; exercise the production Host tiling unchanged.
#define CATLASS_DEVICE
using GM_ADDR = uint8_t*;
#include "../kernels/78_x_attention/x_attention_tiling.hpp"

#ifndef ASSERT
#ifndef NDEBUG
#define ASSERT(condition)          \
    do {                           \
        if (!(condition)) {        \
            return 1;              \
        }                          \
    } while (0)
#else
#define ASSERT(condition) ((void)0)
#endif
#endif

int main()
{
    using namespace XAttentionTiling;
    Context valid{1, 4, 8, 2, 128, 33, 256, 24, true};
    CanImplement(valid);
    for (bool paged : {false, true}) {
        auto context = valid;
        context.sharedPaged = paged;
        XAttentionTilingData tiling{};
        uint64_t workspace = 0;
        const auto tilingKey = GetTiling(context, tiling, workspace);
        ASSERT(tilingKey == (paged ? 8 : 4));
        ASSERT(tiling.numTokens == 4 && workspace > 0);
    }

    auto rejects = [](Context context) {
        try {
            CanImplement(context);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };

    for (auto field : {&Context::batch, &Context::beamSize, &Context::numHeads,
                       &Context::kvHeads, &Context::sharedKvSeqLen, &Context::maxDecodeStep}) {
        auto context = valid;
        context.*field = 0;
        const bool rejected = rejects(context);
        ASSERT(rejected);
    }

    auto context = valid;
    context.numHeads = 9;
    bool rejected = rejects(context);
    ASSERT(rejected);
    context = valid; context.numHeads = 258; rejected = rejects(context); ASSERT(rejected);
    context = valid; context.embeddingSize = 64; rejected = rejects(context); ASSERT(rejected);
    context = valid; context.maxDecodeStep = 257; rejected = rejects(context); ASSERT(rejected);
    context = valid; context.coreNum = 1; rejected = rejects(context); ASSERT(rejected);
    context = valid; context.sharedKvSeqLen = UINT32_MAX; rejected = rejects(context); ASSERT(rejected);
    context = valid; context.batch = UINT32_MAX; context.beamSize = UINT32_MAX; rejected = rejects(context); ASSERT(rejected);
    context = valid; context.batch = 129; context.sharedKvSeqLen = INT32_MAX; rejected = rejects(context); ASSERT(rejected);
    context = valid; context.beamSize = 1; context.numHeads = 1; context.kvHeads = 1;
    context.batch = UINT32_MAX / 129;
    CanImplement(context);
    ++context.batch;
    rejected = rejects(context);
    ASSERT(rejected);
    const auto ceilDivResult = CeilDivHost(UINT32_MAX, 128);
    ASSERT(ceilDivResult == 33554432);
    return 0;
}
