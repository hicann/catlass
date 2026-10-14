# Gemm/Kernel Class Template Overview

## API List

- [basic_matmul](./basic_matmul.md): Basic matrix multiplication of the
  Common template.
- [batched_matmul_softmax_tla](./batched_matmul_softmax_tla.md): Fused
  batched matmul and row-wise softmax template.
- [basic_matmul_tla_visitor](./basic_matmul_tla_visitor.md): EVG GM
  workspace kernel entry point.
- [basic_matmul_tla_ub_visitor](./basic_matmul_tla_ub_visitor.md): EVG UB
  workspace kernel entry point.

## API Breakdown

The namespace is `Catlass::Gemm::Kernel`. The class template contains the
following core members.

- `Params`: Input arguments used when invoking the device kernel function.
- `Arguments`: Input arguments encapsulated after device execution.
- `bool CanImplement`: `Arguments` validation interface.
- `size_t GetWorkspaceSize`: Calculates the required workspace size based on
  `Arguments`.
- `Params ToUnderlyingArguments`: Converts `Arguments` into kernel arguments.
- `void operator()<AscendC::AIC>`: Inputs `Params` and executes MMAD
  computations on the AIC.
- `void operator()<AscendC::AIV>`: Inputs `Params` and executes calculation on
  the AIV, such as prologue and epilogue.
