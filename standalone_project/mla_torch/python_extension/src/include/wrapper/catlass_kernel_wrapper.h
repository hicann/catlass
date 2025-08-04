#ifndef PY_EXT_CATLASS_KERNEL_WRAPPER_H
#define PY_EXT_CATLASS_KERNEL_WRAPPER_H

#include <pybind11/stl.h>
#include <torch/extension.h>

#include "catlass_kernel.h"

namespace CatlassKernelWrapper {

at::Tensor RunMLA(
    const at::Tensor &q,
    const at::Tensor &q_rope,
    const at::Tensor &k,
    const at::Tensor &k_rope,
    const at::Tensor &block_table,
    const std::string &dtype_str
);

} // namespace CatlassKernelWrapper

#endif // PY_EXT_CATLASS_KERNEL_WRAPPER_H
