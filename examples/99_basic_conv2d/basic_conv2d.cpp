/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0, 
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include <vector>

#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/conv2d/block/block_mmad.hpp"
#include "catlass/conv2d/block/block_swizzle.hpp"
#include "catlass/conv2d/dispatch_policy.hpp"
#include "catlass/conv2d/kernel/basic_conv2d.hpp"
#include "catlass/conv2d/conv2d_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/status.hpp"
#include "catlass/conv2d/device/device_conv2d.hpp"

#include "catlass/conv2d_coord.hpp"
#include "catlass/layout/conv2d.hpp"

using namespace Catlass;
using fp16_t = op::fp16_t;

struct Options {
  const std::string HELPER = 
    "24_basic_conv2d batch, hi, wi, cin, cout, kh, kw, padLeft, padRight, padTop, padBottom, strideH, strideW, dilationH, dilationW [device_id]";
  
  uint32_t dataSizes[5] = {2, 33, 43, 112, 80}; // {batch, hi, wi, cin, cout}
  uint8_t filterSizes[2] = {3, 3}; // {kh, kw}
  uint8_t pads[4] = {2, 2, 2, 2}; // {padLeft, padRight, padTop, padBottom}
  uint8_t strides[2] = {2, 2}; // {strideH, strideW}
  uint8_t dilations[2] = {1, 1}; // {dilationH, dilationW}
  int32_t deviceId{0};

  Catlass::Conv2dParams problemParams{};

  Options() = default;

  int Parse(int argc, const char **argv) {
    enum ArgsIndex {
      BATCH_INDEX = 1,
      HI_INDEX,
      WI_INDEX,
      CIN_INDEX,
      COUT_INDEX,
      KH_INDEX, KW_INDEX,
      PADLEFT_INDEX, PADRIGHT_INDEX, PADTOP_INDEX, PADBOTTOM_INDEX,
      STRIDEH_INDEX, STRIDEW_INDEX,
      DILATIONH_INDEX, DILATIONW_INDEX,
      DEVICE_ID_INDEX,
      ARGS_MAX
    };

    // if (argc > ARGS_MAX || argc <= DILATIONW_INDEX) {
    //   std::cerr << HELPER << std::endl;
    //   return 0;
    // }

    // dataSizes[0] = std::atoi(argv[BATCH_INDEX]);
    // dataSizes[1] = std::atoi(argv[HI_INDEX]);
    // dataSizes[2] = std::atoi(argv[WI_INDEX]);
    // dataSizes[3] = std::atoi(argv[CIN_INDEX]);
    // dataSizes[4] = std::atoi(argv[COUT_INDEX]);
    // filterSizes[0] = std::atoi(argv[KH_INDEX]);
    // filterSizes[1] = std::atoi(argv[KW_INDEX]);
    // pads[0] = std::atoi(argv[PADLEFT_INDEX]);
    // pads[1] = std::atoi(argv[PADRIGHT_INDEX]);
    // pads[2] = std::atoi(argv[PADTOP_INDEX]);
    // pads[3] = std::atoi(argv[PADBOTTOM_INDEX]);
    // strides[0] = std::atoi(argv[STRIDEH_INDEX]);
    // strides[1] = std::atoi(argv[STRIDEW_INDEX]);
    // dilations[0] = std::atoi(argv[DILATIONH_INDEX]);
    // dilations[1] = std::atoi(argv[DILATIONW_INDEX]);

    problemParams = Catlass::Conv2dParams::MakeConv2dParams(dataSizes, filterSizes, pads, strides, dilations);

    if (argc == ARGS_MAX) {
      deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
    }
    return 0;
  }
};

void Run(Options const &options) {
  aclrtStream stream{nullptr};

  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(options.deviceId));
  ACL_CHECK(aclrtCreateStream(&stream));

  uint32_t c0 = options.problemParams.C0;
  uint32_t batch = options.problemParams.batch();
  uint32_t hi = options.problemParams.hi();
  uint32_t wi = options.problemParams.wi();
  uint32_t cin1 = options.problemParams.cin1();
  uint32_t ho = options.problemParams.ho();
  uint32_t wo = options.problemParams.wo();
  uint32_t howo = options.problemParams.howo();
  uint32_t howoRound = options.problemParams.howoRound();
  uint32_t cout1 = options.problemParams.cout1();
  uint32_t cout = options.problemParams.cout();
  uint32_t coutRound = options.problemParams.coutRound();
  uint32_t kh = options.problemParams.kh();
  uint32_t kw = options.problemParams.kw();
  uint32_t padLeft = options.problemParams.padLeft();
  uint32_t padRight = options.problemParams.padRight();
  uint32_t padTop = options.problemParams.padTop();
  uint32_t padBottom = options.problemParams.padBottom();
  uint32_t strideH = options.problemParams.strideH();
  uint32_t strideW = options.problemParams.strideW();
  uint32_t dilationH = options.problemParams.dilationH();
  uint32_t dilationW = options.problemParams.dilationW();

  printf("c0 = %d\n", c0);
  printf("batch = %d\n", batch);
  printf("hi = %d\n", hi);
  printf("wi = %d\n", wi);
  printf("kh = %d\n", kh);
  printf("kw = %d\n", kw);
  printf("cin1 = %d\n", cin1);
  printf("ho = %d\n", ho);
  printf("wo = %d\n", wo);
  printf("howo = %d\n", howo);
  printf("howoRound = %d\n", howoRound);
  printf("cout1 = %d\n", cout1);
  printf("cout = %d\n", cout);
  printf("coutRound = %d\n", coutRound);
  
  size_t lenFmap = batch * cin1 * hi * wi * c0;
  size_t lenFilter = cin1 * kh * kw * cout * c0;
  size_t lenOutput = batch * ho * wo * coutRound;

  size_t sizeFmap = lenFmap * sizeof(fp16_t);
  size_t sizeFilter = lenFilter * sizeof(fp16_t);
  size_t sizeOutput = lenOutput * sizeof(fp16_t);

  using LayoutFmap = layout::Fmap;
  using LayoutFilter = layout::Filter;
  using LayoutOutput = layout::Output;
  LayoutFmap layoutFmap{batch, cin1, hi, wi, c0};
  LayoutFilter layoutFilter{cin1, kh, kw, cout, c0};
  LayoutOutput layoutOutput{batch, cout1, ho, wo, c0};

  std::vector<fp16_t> hostFmap(lenFmap);
  std::vector<fp16_t> hostFilter(lenFilter);
  golden::FillRandomData<fp16_t>(hostFmap, -5.0f, 5.0f);
  golden::FillRandomData<fp16_t>(hostFilter, -5.0f, 5.0f);

  uint8_t *deviceFmap{nullptr};
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceFmap), sizeFmap, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMemcpy(deviceFmap, sizeFmap, hostFmap.data(), sizeFmap, ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t *deviceFilter{nullptr};
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceFilter), sizeFilter, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMemcpy(deviceFilter, sizeFilter, hostFilter.data(), sizeFilter, ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t *deviceOutput{nullptr};
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceOutput), sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST));
  
  // Get the number of cube cores of the current hardware
  auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

  using ArchTag = Arch::AtlasA2;
  constexpr bool ENABLE_UNIT_FLAG = false;
  using DispatchPolicy = Conv2d::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;
  using FmapL1TileShape = Catlass::Conv2dFmapL1Shape<8, 12, 2>; // (hoBlock, woBlock, cin1BlockSmall)
  using FilterL1TileShape = Catlass::Conv2dFilterL1Shape<32, 4>; // (coutBlock, cin1BlockBig)
  using L0TileShape = Catlass::Conv2dL0Shape<16, 16, 16>; // (mL0, nL0, kL0)

  uint32_t hoBlock = FmapL1TileShape::Ho;
  uint32_t woBlock = FmapL1TileShape::Wo;
  uint32_t cin1FmapL1Block = FmapL1TileShape::Cin1;
  uint32_t coutBlock = RoundUp(FilterL1TileShape::Cout, c0);
  uint32_t cin1FilterL1Block = FilterL1TileShape::Cin1;
  uint32_t hiBlock = (hoBlock - 1) * strideH + (kh - 1) * dilationH + 1;
  uint32_t wiBlock = (woBlock - 1) * strideW + (kw - 1) * dilationW + 1;

  uint32_t hoL0Tile = Max(L0TileShape::M / woBlock, 1);
  uint32_t mL0A = hoL0Tile * woBlock;
  uint32_t mL0C = RoundUp<C0_NUM_PER_FRACTAL>(mL0A);
  uint32_t mPartLoop = CeilDiv(hoBlock, hoL0Tile);
  uint32_t mL1C = mL0C * mPartLoop;
  uint32_t cin1L0Block = Max(L0TileShape::K / (kh * kw * c0), 1);
  uint32_t coutL0Block = RoundUp(L0TileShape::N, c0);

  uint32_t l1DataSize = 
      2 * (cin1FmapL1Block * hiBlock * wiBlock * c0 + 
           cin1FilterL1Block * kh * kw * coutBlock *c0) * sizeof(fp16_t);
  uint32_t l0ADataSize = 
      2 * mL0A * (cin1L0Block * kh * kw * c0) * sizeof(fp16_t);
  uint32_t l0BDataSize = 
      2 * (cin1L0Block * kh * kw * c0) * coutL0Block * sizeof(fp16_t); 
  uint32_t l0CDataSize = mL1C * coutBlock * sizeof(float); 
  
  printf("l1DataSize=%d, L1Size=%d\n", l1DataSize, ArchTag::L1_SIZE);
  printf("l0ADataSize=%d, L0A_SIZE=%d\n", l0ADataSize, ArchTag::L0A_SIZE);
  printf("l0BDataSize=%d, L0B_SIZE=%d\n", l0BDataSize, ArchTag::L0B_SIZE);
  printf("l0CDataSize=%d, L0C_SIZE=%d\n", l0CDataSize, ArchTag::L0C_SIZE);
  
  using FmapType = Conv2d::Conv2dType<half, LayoutFmap>;
  using FilterType = Conv2d::Conv2dType<half, LayoutFilter>;
  using OutputType = Conv2d::Conv2dType<half, LayoutOutput>;

  using BlockMmad = Conv2d::Block::BlockMmad<DispatchPolicy,
      FmapL1TileShape, FilterL1TileShape, L0TileShape,
      FmapType, FilterType, OutputType>;
  using BlockEpilogue = void;

  // Swizzle offset is 3 and direction is 0.
  using BlockScheduler = typename Conv2d::Block::Conv2dIdentityBlockSwizzle<3, 0>;

  // kernel level
  using Conv2dKernel = Conv2d::Kernel::BasicConv2d<BlockMmad, BlockEpilogue, BlockScheduler>;

  using Conv2dAdapter = Conv2d::Device::DeviceConv2d<Conv2dKernel>;
  Conv2dKernel::Arguments arguments{options.problemParams, deviceFmap, deviceFilter, deviceOutput};
  Conv2dAdapter conv2d_op;
  conv2d_op.CanImplement(arguments);
  size_t sizeWorkspace = conv2d_op.GetWorkspaceSize(arguments);
  uint8_t *deviceWorkspace = nullptr;
  if (sizeWorkspace > 0) {
    ACL_CHECK(
      aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace),
                  sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
  }
  conv2d_op.Initialize(arguments, deviceWorkspace);
  conv2d_op(stream, aicCoreNum);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  if (sizeWorkspace > 0) {
    ACL_CHECK(aclrtFree(deviceWorkspace));
  }
  
  std::vector<fp16_t> hostOutput(lenOutput);
  ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput,
                        deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));

  std::vector<float> hostGolden(lenOutput);
  golden::ComputeConv2d(options.problemParams,
                        hostFmap, layoutFmap, 
                        hostFilter, layoutFilter, 
                        hostGolden, layoutOutput);

  std::vector<uint64_t> errorIndices =
      golden::CompareData(hostOutput, hostGolden, cin1 * kh * kw * c0);
  if (errorIndices.empty()) {
    std::cout << "Compare success." << std::endl;
  } else {
    std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
  }

  ACL_CHECK(aclrtFree(deviceFmap));
  ACL_CHECK(aclrtFree(deviceFilter));
  ACL_CHECK(aclrtFree(deviceOutput));

  ACL_CHECK(aclrtDestroyStream(stream));
  ACL_CHECK(aclrtResetDevice(options.deviceId));
  ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
