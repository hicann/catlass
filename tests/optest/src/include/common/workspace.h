#pragma once
#include <cstddef>

#include <acl/acl.h>

namespace catlass_torch {
namespace common {
/**
 * @brief Allocate temporary memory for certain operator. For being managed by torch_npu, do not use `aclrtMalloc` or
 * `aclrtMallocAlign32` to allocate workspace memory.
 *
 * @param wkspAddr An address pointing to the workspace address variable.
 * @param wkspSize The size of the workspace to allocate.
 * @return aclError
 */
aclError workspaceMalloc(void **wkspAddr, size_t wkspSize);
} // namespace common
} // namespace catlass_torch
