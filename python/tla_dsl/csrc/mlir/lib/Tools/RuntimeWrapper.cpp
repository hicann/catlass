#include "acl/acl.h"
#include "acl/error_codes/rt_error_codes.h"
#include "runtime/kernel.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct RegisteredKernel {
  std::unique_ptr<char, decltype(&std::free)> buffer{nullptr, &std::free};
  std::unique_ptr<uint64_t> stub;
};

std::mutex g_mutex;
std::unordered_map<uint64_t, RegisteredKernel> g_registered_kernels;
thread_local std::string g_last_error;

char *read_bin_file(const char *file_name, uint32_t *file_size) {
  std::ifstream file(file_name, std::ios::binary);
  if (!file) {
    g_last_error = std::string("failed to open kernel file: ") + file_name;
    return nullptr;
  }

  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (size < 0) {
    g_last_error = std::string("failed to stat kernel file: ") + file_name;
    return nullptr;
  }

  char *buffer = static_cast<char *>(std::malloc(static_cast<size_t>(size)));
  if (!buffer) {
    g_last_error = "failed to allocate kernel file buffer";
    return nullptr;
  }
  if (!file.read(buffer, size)) {
    std::free(buffer);
    g_last_error = std::string("failed to read kernel file: ") + file_name;
    return nullptr;
  }
  *file_size = static_cast<uint32_t>(size);
  return buffer;
}

void set_rt_error(const char *op_name, rtError_t ret) {
  g_last_error =
      std::string(op_name) + " failed: 0x" + std::to_string(static_cast<unsigned int>(ret));
}

void set_acl_error(const char *op_name, aclError ret) {
  g_last_error =
      std::string(op_name) + " failed: 0x" + std::to_string(static_cast<unsigned int>(ret));
}

} // namespace

extern "C" const char *tla_runtime_last_error() { return g_last_error.c_str(); }

extern "C" int tla_runtime_load_kernel(const char *file_path, const char *stub_func,
                                       const char *kernel_mode, uint64_t *module_out,
                                       uint64_t *function_out) {
  if (!file_path || !stub_func || !kernel_mode || !module_out || !function_out) {
    g_last_error = "tla_runtime_load_kernel received null argument";
    return -1;
  }

  uint32_t buffer_size = 0;
  char *buffer = read_bin_file(file_path, &buffer_size);
  if (!buffer) {
    return -1;
  }

  rtDevBinary_t binary;
  binary.data = buffer;
  binary.length = buffer_size;
  std::string mode{kernel_mode};
  binary.magic = mode == "aiv" ? RT_DEV_BINARY_MAGIC_ELF_AIVEC : RT_DEV_BINARY_MAGIC_ELF;
  binary.version = 0;

  void *module = nullptr;
  rtError_t rt_ret = rtDevBinaryRegister(&binary, &module);
  if (rt_ret != RT_ERROR_NONE) {
    std::free(buffer);
    set_rt_error("rtDevBinaryRegister", rt_ret);
    return -1;
  }

  auto stub = std::make_unique<uint64_t>(0);
  void *stub_ptr = reinterpret_cast<void *>(stub.get());
  rt_ret = rtFunctionRegister(module, stub_ptr, stub_func, const_cast<char *>(stub_func), 0);
  if (rt_ret != RT_ERROR_NONE) {
    std::free(buffer);
    set_rt_error("rtFunctionRegister", rt_ret);
    return -1;
  }

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    RegisteredKernel kernel;
    kernel.buffer.reset(buffer);
    kernel.stub = std::move(stub);
    g_registered_kernels.emplace(reinterpret_cast<uint64_t>(module), std::move(kernel));
  }

  *module_out = reinterpret_cast<uint64_t>(module);
  *function_out = reinterpret_cast<uint64_t>(stub_ptr);
  return 0;
}

extern "C" int tla_runtime_launch_kernel(uint64_t function_handle, uint64_t stream_handle, int gx,
                                         int gy, int gz, const uint64_t *args, size_t num_args) {
  const void *function = reinterpret_cast<const void *>(function_handle);
  rtStream_t stream = reinterpret_cast<rtStream_t>(stream_handle);
  uint32_t block_num =
      static_cast<uint32_t>(gx) * static_cast<uint32_t>(gy) * static_cast<uint32_t>(gz);

  std::vector<uint64_t> values;
  if (args && num_args > 0) {
    values.assign(args, args + num_args);
  }

  void *args_array = values.empty() ? nullptr : static_cast<void *>(values.data());
  rtError_t rt_ret = rtKernelLaunch(function, block_num, args_array,
                                    values.size() * sizeof(uint64_t), nullptr, stream);
  if (rt_ret != RT_ERROR_NONE) {
    set_rt_error("rtKernelLaunch", rt_ret);
    return -1;
  }

  aclError acl_ret = aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream_handle));
  if (acl_ret != ACL_RT_SUCCESS) {
    set_acl_error("aclrtSynchronizeStream", acl_ret);
    return -1;
  }
  return 0;
}
