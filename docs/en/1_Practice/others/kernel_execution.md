# Directly Launching a Newly Developed Operator with `<<<>>>`

## Before You Start

This document mainly explains how developers can directly launch a newly developed CATLASS operator through Ascend C's `<<<>>>` operator invocation syntax after completing its development, along with the related precautions.

## `<<<>>>` Syntax

The operator invocation syntax `<<<...>>>` is a syntax provided by Ascend C. It encapsulates calls to Runtime APIs, making it convenient for developers to schedule operators onto the NPU's AI Core for execution.
The following is an example of invoking a Kernel through `<<<>>>`:

```cpp
kernel_name<<<blockDim, l2ctrl, stream>>>(argument_list);
```

The meanings of the three parameters are as follows:

| Parameter  | Type          | Description                                                   |
| ---------- | ------------- | ------------------------------------------------------------- |
| `blockDim` | `uint32_t`    | How many AI Cores are used to execute the operator            |
| `l2ctrl`   | `void*`       | Reserved parameter, always set to `nullptr`                   |
| `stream`   | `aclrtStream` | Stream object that manages the execution order of async operations |

Here, `blockDim` indicates how many hardware AI Cores are needed to execute the operator, and is generally obtained through the `platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()` interface. Inside the operator, you can obtain the current core index through `GetBlockIdx()` and the total number of cores through `GetBlockNum()`.

> **Note**: Operator invocation is asynchronous. After the `<<<>>>` call ends, control immediately returns to the Host side. To wait for execution to complete, call `aclrtSynchronizeStream(stream)`.

## Direct Invocation Flow

After developing a new Kernel based on CATLASS template components, the overall flow for using `<<<>>>` direct invocation is as follows:

1. Environment initialization: `aclInit`, `aclrtSetDevice`, `aclrtCreateStream`.
2. Data preparation: `aclrtMallocHost` allocates and initializes Host memory, `aclrtMalloc` allocates Device memory, and `aclrtMemcpy` copies the data to the Device.
3. Assemble template components: select the ArchTag, DispatchPolicy, TileShape, and data types, assemble components such as BlockMmad, and put together the Kernel type.
4. Invoke the operator using `<<<>>>`:

   ```cpp
   auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
   Catlass::KernelAdapter<MyKernel><<<aicCoreNum, nullptr, stream>>>(params);
   ```

5. Copy results back: copy the computation results on the Device back to the Host via `aclrtMemcpy`.
6. Synchronous wait: `aclrtSynchronizeStream(stream)`.
7. Resource release: `aclrtDestroyStream(stream)`, `aclrtResetDevice(deviceId)`, `aclFinalize()`.

## Comparison Between `<<<>>>` Direct Invocation and DeviceGemm

CATLASS's Device layer (such as `DeviceGemm`) internally launches operators through `<<<>>>`, and additionally encapsulates convenient features such as `CanImplement` checking and Workspace management.

| Comparison Dimension | `<<<>>>` Direct Invocation               | `DeviceGemm`                 |
| -------------------- | ---------------------------------------- | ---------------------------- |
| Invocation method    | Hand-written `KernelAdapter<...><<<>>>(params)` | `matmulOp(stream, blockDim)` |
| Workspace management | Must be handled manually                 | Encapsulated within the adapter |
| Recommended scenarios | Prototype development, debugging phase, non-standard Kernels | Standard GEMM/GEMV/Conv |

In short, for developing standard operators, `DeviceGemm` is recommended; choose direct invocation when you need more control.
