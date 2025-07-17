# 使用printf和DumpTensor调试算子

CATLASS算子直接使用毕昇编译器编译，不使用算子工程进行开发，因此无法直接使用kernel调测api. 但经过适配，使用这些api成为可能. 本示例演示了使用kernel调测api需要的改动.

关于kernel调测api的详细介绍，可参考[DumpTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/ascendcopapi/atlasascendc_api_07_0192.html)和[printf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/ascendcopapi/atlasascendc_api_07_0193.html).

# 改动点

1. 引入调测接口

```cpp
// debug_helper.hpp
namespace Adx {
void AdumpPrintWorkSpace(const void *dumpBufferAddr,
                         const size_t dumpBufferSize, void *stream,
                         const char *opType);
}
```

2. kernel增加一个gmDump参数，并设置其为dump输出位置

```cpp
#define  ALL_DUMPSIZE (75 * 1024 * 1024) // default value
CATLASS_GLOBAL
void kernel(
//...
GM_ADDR gmDump
){
AscendC::InitDump(false, gmDump, ALL_DUMPSIZE);
//...
}
```

3. 申请dump内存；kernel执行、同步流之后，调用host调测函数

```cpp
uint8_t *deviceDump{nullptr};
ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceDump), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
kernel<<<...>>>(..., deviceDump);
ACL_CHECK(aclrtSynchronizeStream(stream));
Adx::AdumpPrintWorkSpace(deviceDump, ALL_DUMPSIZE, stream, "basic_matmul");
ACL_CHECK(aclrtFree(deviceDump));
```

4. 编译时增加宏与动态库

- 增加宏：`-DASCENDC_DUMP=1 -DASCENDC_DEBUG`
- 增加动态库：`-lascend_dump`

- 以上过程封装在此example中，通过宏定义动态区分，可作为参考.
- 编译命令：

```bash
bash scripts/build.sh --enable_dump print_and_dump
```
# 原理

将dump数据存入NPU内存，host侧搬出后解析打印.

# 注意事项

- 在不使用调测API时，要将调测API的调用删除，否则必然报错，即使不使用`--enable_dump`.
- 该API仅用于功能性Debug，在功能调测通过，进行性能调优时，建议将此功能关闭，以免影响性能.

## 版权声明

Copyright (c) 2025 Huawei Technologies Co., Ltd.

This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY, OR FITNESS FOR A PARTICULAR   PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.

## 许可证

[CANN Open Software License Agreement Version 1.0](../../LICENSE)
