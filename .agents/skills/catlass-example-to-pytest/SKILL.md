---
name: "catlass-example-to-pytest"
description: "Integrate numbered CATLASS examples from examples/ into tests/optest with consistent ABI, torch.ops registration, Python wrappers, and pytest verification."
---

# Catlass Example to Pytest Skill

将 `examples/` 下带编号算子接入 `tests/optest/` 测试框架，产出可构建、可导入、可测试的完整链路：

```text
example source
  -> kernel entry (jit/prebuilt)
  -> ABI declaration (include/catlass_kernel_*.h)
  -> torch C++ adapter (src/)
  -> python wrapper (torch_catlass/ops/)
  -> pytest (tests/)
```

## Scope

- 工作目录：`tests/optest/`
- 输入：`examples/<nn>_<name>/`
- 输出：
  - 头文件预留接口（按模块）
  - C++/Python 调用链接入
  - pytest 用例
  - README / docs 必要更新（仅与接入相关）

## Non-negotiable Rules

1. 参数语义统一使用 `TParams` + `Params`。
2. `matmul/gemm/gemv` 归入 JIT 接口头：`include/catlass_kernel_jit.h`。
3. 其他（如 conv / flash-attention / mla）归入 prebuilt 接口头：`include/catlass_kernel_prebuilt.h`。
4. 不大量引入 `using XxxParams = ...` 去伪装兼容；优先使用统一结构体或结构体继承表达兼容关系。
5. grouped quant 场景避免把运行时 dtype 再塞回 `Params`；dtype 应放在 `QuantMatmulTParams` 这类模板参数结构。
6. 架构识别逻辑遵循现有实现：JIT 侧从 `GetCurrentNPUArch()` 获取，不新增环境变量分支。
7. 任何已有用户改动不可回滚；发现冲突先停并问用户。

## Phase 0: Intake & Classification

### 0.1 解析 example

- 读取：
  - `examples/<dir>/CMakeLists.txt`
  - `examples/<dir>/README*`（若存在）
  - 主 `.cpp` / `.h`
- 提取：
  - 算子家族：`matmul/gemm/gemv` 或 `other`
  - 入口函数名（建议 PascalCase C++ + snake_case Python）
  - 参数特征：是否包含 bias / quant / grouped / rope / mask / workspace

### 0.2 分类规则

- dirname 含 `matmul|gemm|gemv` -> JIT 类
- dirname 含 `conv|flash_attention|mla|fai` -> prebuilt 类
- 不确定时：
  - 先按源码依赖（是否强依赖模板宏生成/JIT）判断
  - 仍不明确则问用户

### 0.3 重名检测

检查以下是否已存在：

- `include/catlass_kernel_jit.h` / `include/catlass_kernel_prebuilt.h` 中同名声明
- `src/catlass_torch.cpp` 中同名注册
- `torch_catlass/ops/<name>.py`
- `tests/test_<nn>_<name>.py` 或 `tests/test_<name>.py`

重复时默认“增量更新不覆盖”，除非用户明确要求覆盖。

## Phase 1: ABI Reservation First

先预留 ABI，再写实现，避免后续接口漂移。

### 1.1 JIT ABI（matmul family）

在 `include/catlass_kernel_jit.h`：

- 复用或扩展统一结构体：
  - `MatmulTParams`
  - `MatmulParams`
  - `GroupedMatmulParams`
  - `QuantMatmulTParams`
  - `GemmParams`
- 添加函数声明，并写明 example 编号和目录名的 docstring：
  - `@brief Reserved JIT interface for example <nn>_<name>.`

### 1.2 Prebuilt ABI（others）

在 `include/catlass_kernel_prebuilt.h`：

- 复用或扩展统一结构体：
  - `PrebuiltParams`
  - `ConvParams`
  - `FlashAttentionParams`
  - `MlaParams`
- 添加函数声明并保留编号 docstring。

### 1.3 聚合头

- `include/catlass_kernel.h` 只做聚合 include，不承载具体参数定义。

## Phase 2: Kernel Integration

### 2.1 JIT 类接入

在 `kernels/<nn>_<name>/` 创建或更新：

1. `<name>.cpp`：JIT 入口（调用 `JitCompiler::instance().getKernel(...)`）
2. `<name>_impl.cpp`：模板实现，支持 `CATLASS_JIT_*` 宏
3. `CMakeLists.txt`：`add_kernel(... jit ... TEMPLATE ...)`

要求：

- 模板默认宏齐全（dtype/layout/transpose 等）。
- 运行入口签名与 ABI 声明一致。
- 不把 example 的命令行/数据生成逻辑带入内核模板。

### 2.2 Prebuilt 类接入

在 `kernels/<nn>_<name>/`：

- 使用 prebuilt 方式编译并导出与 ABI 一致的入口函数。
- 若需 workspace，参数和实际 launch 必须一致（shape、dtype、flags 全对齐）。

### 2.3 构建入口

在 `kernels/CMakeLists.txt` 补 `add_subdirectory(<nn>_<name>)`（保持编号顺序）。

## Phase 3: Torch Adapter & Python Wrapper

### 3.1 C++ 注册

在 `src/catlass_torch.cpp`：

- JIT matmul 类：优先复用 `MatmulLike<...>` 或同类通用适配器。
- 非 matmul：按参数语义新增轻量 adapter。
- 使用既有 `REGISTER_TORCH_FUNC(...)` 注册到 `torch.ops.catlass.*`。

### 3.2 Python 包装

在 `torch_catlass/ops/<name>.py`：

- 参数名与 C++ adapter 对齐。
- dtype 字符串解析保持白名单/显式映射，不使用隐式 `eval`。
- docstring 写清：
  - 来源 example 编号名称
  - 参数语义
  - 输出语义

并在 `torch_catlass/ops/__init__.py` 导出。

## Phase 4: Tests

### 4.1 pytest 用例

新增 `tests/test_<nn>_<name>.py`（建议保留编号，避免重名）。

最小断言：

1. shape 正确
2. dtype 正确
3. device 为 NPU
4. 与基线（`torch.matmul` 或 reference kernel）数值对齐（给定 `rtol/atol`）

### 4.2 无设备处理

- 用 `pytest.mark.skipif(torch_npu.npu.device_count() <= 0, ...)`。
- 跳过只用于环境不可用，不用于掩盖接口错误。

## Phase 5: Verify & Fix Loop

每次接入至少执行：

1. `bash build.sh`（必要时 `--skip-wheel`）
2. `pytest tests/test_<nn>_<name>.py -v`
3. `python -c "import torch_catlass"`

失败处理：

- 编译错误：先修 ABI/签名/CMake 路径问题。
- 导入错误：先查 `torch_catlass/__init__.py` 动态库加载顺序与符号。
- 运行错误：核对 `TParams/Params` 填充及 dtype/layout 映射。

## Checklists

### Interface Checklist

- [ ] example 分类正确（jit or prebuilt）
- [ ] 头文件声明落在正确模块（jit/prebuilt）
- [ ] docstring 含 example 编号名称
- [ ] 参数模型与 `TParams + Params` 保持一致
- [ ] 无不必要 `using XxxParams` 别名滥用

### Runtime Checklist

- [ ] `torch.ops.catlass.<name>` 可见
- [ ] `import torch_catlass` 成功
- [ ] pytest 通过或因无 NPU 合理 skip
- [ ] 与参考实现数值一致

## Output Contract

完成后必须给出：

1. 修改文件列表（按模块分组）
2. ABI 变更摘要（新增/复用了哪些 `TParams/Params`）
3. 构建与测试结果（命令 + 结论）
4. 若有环境阻塞，明确区分“代码问题”与“环境问题”
