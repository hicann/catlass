# 算子模板编写

catlass_test使用`bisheng`编译算子代码，输出一个名称类似于`libbasic_matmul_half_rowmajor_half_rowmajor_half_rowmajor.so`的动态库。

在catlass增加example后，需要同步在`catlass_test/csrc/examples`增加可用的example的对应算子模板。以basic_matmul为例，它的代码如下：

```cpp
// catlass提供公共函数头文件
#include "common.hpp"

// catlass头文件
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/layout/matrix.hpp"
#include "catlass/status.hpp"

using namespace Catlass;
// 所测算子提供的最小化模板
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
inline void BasicMatmul(aclrtStream stream, GemmCoord problemShape, uint8_t *deviceA, uint8_t *deviceB, uint8_t *deviceC) {
    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;

    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using AType = Gemm::GemmType<ElementA, LayoutA>;
    using BType = Gemm::GemmType<ElementB, LayoutB>;
    using CType = Gemm::GemmType<ElementC, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    typename MatmulKernel::Arguments arguments{problemShape, deviceA, deviceB, deviceC};
    MatmulAdapter matmulOp;
    RUN_ADAPTER(matmulOp, arguments, stream, aicCoreNum);
}
```

#
注意：

- 为方便框架对类型别名进行整行替换，模板代码**不**遵守每行120字符的限制
- 该函数必须使用模板，模板参数和函数参数的名称必须完全符合[变量名规范](1_1_variable_names.md)。若使用不规范的参数，框架难以识别
