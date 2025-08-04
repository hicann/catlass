#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "catlass_kernel.h"
#include "wrapper/catlass_kernel_wrapper.h"

namespace py = pybind11;
using namespace CatlassKernelWrapper;

PYBIND11_MODULE(_C, m) {
    m.doc() = "Python bindings for CatlassKernel";
    m.def("mla", &RunMLA, "");
}
