#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "acl/error_codes/rt_error_codes.h"
#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/stream.h"

namespace {

struct RegisteredKernel {
  std::unique_ptr<char, decltype(&std::free)> buffer{nullptr, &std::free};
  std::unique_ptr<uint64_t> stub;
  bool uses_asc_debug_fifo = false;
  bool uses_print_tensor = false;
};

std::mutex g_mutex;
std::unordered_map<uint64_t, RegisteredKernel> g_registered_kernels;
thread_local std::string g_last_error;

void set_rt_error(const char *op_name, rtError_t ret);
void set_acl_error(const char *op_name, aclError ret);

namespace cce {
namespace internal {

constexpr uint64_t encode_ascii_marker(const char (&text)[9]) {
  uint64_t marker = 0;
  for (unsigned i = 0; i != 8; ++i)
    marker = (marker << 8) | static_cast<unsigned char>(text[i]);
  return marker;
}
constexpr uint64_t kDebugPrintWorkspaceSentinel =
    encode_ascii_marker("TLA_PRNT");
constexpr uint64_t kPrintTensorWorkspaceSentinel =
    encode_ascii_marker("TLA_TPRN");
namespace AscDebugFifo {

// CANN FIFO wire layouts: field order/types and size assertions are ABI.
constexpr uint32_t kDebugCoreRecords = 108;
constexpr uint32_t kRingBufferBytes = 1024 * 1024;
constexpr uint16_t kMagic = 0xAE86;

enum class FifoRecordType : uint32_t {
  Scalar = 1,
  Tensor = 2,
  Shape = 3,
  BufIn = 8,
  BufOut = 9,
};

struct DebugBlockHeadInfo {
  uint32_t length;
  uint32_t coreId;
  uint32_t blockNum;
  uint32_t ringBufLen;
  uint16_t magic;
  uint16_t flag;
  uint32_t rsv;
  uint64_t ringBufAddr;
  uint64_t debugBusAddr;
  uint32_t resvMem[4];
};

struct DebugBlockWriteInfo {
  uint32_t type;
  uint32_t length;
  uint64_t bufOffset;
  uint64_t packIdx;
};

struct DebugBlockReadInfo {
  uint32_t type;
  uint32_t length;
  uint64_t bufOffset;
  uint64_t resv;
};

struct PrintTlv {
  uint32_t type;
  uint32_t length;
  uint32_t blockIdx;
  uint32_t reserved;
  uint64_t fmtOffset;
};

struct PrintTensorTlv {
  uint32_t type;
  uint32_t length;
  uint32_t tensor_addr;
  uint32_t data_type;
  uint32_t desc;
  uint32_t buffer_id;
  uint16_t position;
  uint16_t block_idx;
  uint32_t dim;
  uint32_t shape[8];
  uint32_t reserved;
  uint32_t dump_size;
};

struct PrintShapeTlv {
  uint32_t type;
  uint32_t length;
  uint32_t dim;
  uint32_t shape[8];
  uint32_t reserved;
};

struct FifoData {
  void *device_region = nullptr;
  size_t region_size = 0;
  uint32_t record_count = 0;
  uint32_t block_length = 0;
  uint32_t ring_buffer_offset = 0;
  uint32_t ring_buffer_bytes = 0;
};

static_assert(sizeof(DebugBlockHeadInfo) == 56,
              "DebugBlockHeadInfo must match CANN asc_debug_types.h");
static_assert(sizeof(DebugBlockReadInfo) == 24,
              "DebugBlockReadInfo must match CANN asc_debug_types.h");
static_assert(sizeof(DebugBlockWriteInfo) == 24,
              "DebugBlockWriteInfo must match CANN asc_debug_types.h");
static_assert(sizeof(PrintTlv) == 24,
              "PrintTlv must match CANN AICore scalar printf layout");
static_assert(sizeof(PrintTensorTlv) == 72,
              "PrintTensorTlv must match CANN C310 tensor print layout");
static_assert(sizeof(PrintShapeTlv) == 48,
              "PrintShapeTlv must match CANN C310 shape print layout");

uint32_t align_up(uint32_t value, uint32_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

uint32_t debug_fifo_ring_offset() {
  constexpr uint32_t data_block_size = 32;
  constexpr uint32_t fixed_prefix =
      sizeof(DebugBlockHeadInfo) + sizeof(DebugBlockReadInfo);
  constexpr uint32_t tlv_alignment_reserve = 72;
  constexpr uint32_t payload_offset = fixed_prefix + tlv_alignment_reserve;
  return align_up(payload_offset, data_block_size) - tlv_alignment_reserve;
}

FifoData *open(unsigned block_num) {
  auto fifo = std::make_unique<FifoData>();
  fifo->record_count = std::max<uint32_t>(kDebugCoreRecords, block_num);
  fifo->ring_buffer_bytes = kRingBufferBytes;
  fifo->ring_buffer_offset = debug_fifo_ring_offset();
  fifo->block_length =
      align_up(static_cast<uint32_t>(fifo->ring_buffer_offset +
                                     fifo->ring_buffer_bytes +
                                     sizeof(DebugBlockWriteInfo)),
               64);
  fifo->region_size =
      static_cast<size_t>(fifo->block_length) * static_cast<size_t>(fifo->record_count);

  rtError_t error = rtMalloc(&fifo->device_region, fifo->region_size, RT_MEMORY_HBM, 8);
  if (error != RT_ERROR_NONE) {
    set_rt_error("rtMalloc(AscDebugFifo)", error);
    return nullptr;
  }

  char *host_mem = nullptr;
  error = rtMallocHost(reinterpret_cast<void **>(&host_mem), fifo->region_size, 8);
  if (error != RT_ERROR_NONE) {
    rtFree(fifo->device_region);
    set_rt_error("rtMallocHost(AscDebugFifo init)", error);
    return nullptr;
  }
  std::memset(host_mem, 0, fifo->region_size);

  auto device_base = reinterpret_cast<uintptr_t>(fifo->device_region);
  for (uint32_t i = 0; i < fifo->record_count; ++i) {
    char *record = host_mem + static_cast<size_t>(i) * fifo->block_length;
    auto *head = reinterpret_cast<DebugBlockHeadInfo *>(record);
    head->length = fifo->block_length;
    head->coreId = i;
    head->blockNum = fifo->record_count;
    head->ringBufLen = fifo->ring_buffer_bytes;
    head->magic = kMagic;
    head->flag = 0;
    uintptr_t record_device_addr = device_base + static_cast<uintptr_t>(i) * fifo->block_length;
    head->ringBufAddr = record_device_addr + fifo->ring_buffer_offset;

    auto *read = reinterpret_cast<DebugBlockReadInfo *>(record + sizeof(DebugBlockHeadInfo));
    read->type = static_cast<uint32_t>(FifoRecordType::BufOut);
    read->length = 16;

    auto *write = reinterpret_cast<DebugBlockWriteInfo *>(
        record + fifo->ring_buffer_offset + fifo->ring_buffer_bytes);
    write->type = static_cast<uint32_t>(FifoRecordType::BufIn);
    write->length = 16;
  }

  error = rtMemcpy(fifo->device_region, fifo->region_size, host_mem, fifo->region_size,
                   RT_MEMCPY_HOST_TO_DEVICE);
  rtError_t free_error = rtFreeHost(host_mem);
  if (error != RT_ERROR_NONE) {
    rtFree(fifo->device_region);
    set_rt_error("rtMemcpy(AscDebugFifo host-to-device)", error);
    return nullptr;
  }
  if (free_error != RT_ERROR_NONE) {
    rtFree(fifo->device_region);
    set_rt_error("rtFreeHost(AscDebugFifo init)", free_error);
    return nullptr;
  }

  return fifo.release();
}

void destroy(FifoData *fifo) {
  if (!fifo)
    return;
  if (fifo->device_region)
    rtFree(fifo->device_region);
  delete fifo;
}

constexpr uint64_t kPrintSlotBytes = 8;
constexpr uint64_t kPrintFmtOffsetBase = 16;

bool range_fits(uint64_t offset, uint64_t size, uint64_t limit) {
  return offset <= limit && size <= limit - offset;
}

size_t bounded_c_string_length(const char *text, size_t max_length) {
  for (size_t i = 0; i < max_length; ++i) {
    if (text[i] == '\0')
      return i;
  }
  return max_length;
}

bool read_print_slot(const uint8_t *args, uint64_t arg_bytes, uint64_t &arg_offset,
                     uint64_t &slot) {
  if (!range_fits(arg_offset, kPrintSlotBytes, arg_bytes))
    return false;
  std::memcpy(&slot, args + arg_offset, sizeof(slot));
  arg_offset += kPrintSlotBytes;
  return true;
}

float load_print_slot_float(uint64_t slot) {
  float value = 0.0f;
  std::memcpy(&value, &slot, sizeof(value));
  return value;
}

enum class PrintFormatResult { Printed, Malformed, Unsupported };

bool is_supported_scalar_printf_format(const char *fmt, size_t fmt_length) {
  return (fmt_length == 4 && std::memcmp(fmt, "x=%d", 4) == 0) ||
         (fmt_length == 4 && std::memcmp(fmt, "v=%f", 4) == 0);
}

PrintFormatResult format_scalar_printf(const char *fmt, size_t fmt_length,
                                        const uint8_t *args, uint64_t arg_bytes) {
  if (!fmt)
    return PrintFormatResult::Unsupported;

  uint64_t arg_offset = 0;

  uint64_t slot = 0;
  if (fmt_length == 4 && std::memcmp(fmt, "x=%d", 4) == 0) {
    if (!read_print_slot(args, arg_bytes, arg_offset, slot))
      return PrintFormatResult::Malformed;
    std::printf("x=%d", static_cast<int32_t>(slot));
    return PrintFormatResult::Printed;
  }

  if (fmt_length == 4 && std::memcmp(fmt, "v=%f", 4) == 0) {
    if (!read_print_slot(args, arg_bytes, arg_offset, slot))
      return PrintFormatResult::Malformed;
    std::printf("v=%f", static_cast<double>(load_print_slot_float(slot)));
    return PrintFormatResult::Printed;
  }

  return PrintFormatResult::Unsupported;
}

bool print_malformed_scalar_tlv(const char *reason, uint32_t core) {
  std::printf("TLA printf: core=%u malformed scalar printf TLV (%s)\n", core,
              reason);
  return true;
}

bool print_scalar_tlv(const PrintTlv *tlv, uint64_t total, uint32_t core) {
  if (total < sizeof(PrintTlv))
    return print_malformed_scalar_tlv("short record", core);

  if (tlv->fmtOffset < kPrintSlotBytes)
    return print_malformed_scalar_tlv("fmtOffset before argument area", core);
  if ((tlv->fmtOffset % kPrintSlotBytes) != 0)
    return print_malformed_scalar_tlv("unaligned fmtOffset", core);

  uint64_t fmt_start = kPrintFmtOffsetBase + tlv->fmtOffset;
  if (!range_fits(fmt_start, 1, total))
    return print_malformed_scalar_tlv("fmtOffset out of bounds", core);

  uint64_t args_start = kPrintFmtOffsetBase + kPrintSlotBytes;
  if (fmt_start < args_start)
    return print_malformed_scalar_tlv("fmtOffset overlaps header", core);
  uint64_t arg_bytes = fmt_start - args_start;
  if ((arg_bytes % kPrintSlotBytes) != 0)
    return print_malformed_scalar_tlv("argument slots are not 8-byte aligned",
                                      core);

  auto *record = reinterpret_cast<const uint8_t *>(tlv);
  auto *fmt = reinterpret_cast<const char *>(record + fmt_start);
  size_t max_fmt_length = static_cast<size_t>(total - fmt_start);
  size_t fmt_length = bounded_c_string_length(fmt, max_fmt_length);
  if (fmt_length == max_fmt_length)
    return print_malformed_scalar_tlv("unterminated format string", core);
  if (!is_supported_scalar_printf_format(fmt, fmt_length))
    return false;

  std::printf("TLA printf: core=%u block=%u ", core, tlv->blockIdx);
  if (format_scalar_printf(fmt, fmt_length, record + args_start, arg_bytes) ==
      PrintFormatResult::Malformed)
    std::printf("<malformed scalar printf TLV: missing argument slot>");
  std::printf("\n");
  return true;
}

struct PrintTensorDType {
  const char *name;
  uint32_t byte_width;
  bool supports_unified_buffer;
};

bool get_print_tensor_dtype(uint32_t code, PrintTensorDType &dtype) {
  // CANN DataType wire values.
  switch (code) {
  case 0:
    dtype = {"float32", 4, true};
    return true;
  case 1:
    dtype = {"float16", 2, true};
    return true;
  case 2:
    dtype = {"int8", 1, true};
    return true;
  case 3:
    dtype = {"int32", 4, true};
    return true;
  case 4:
    dtype = {"uint8", 1, true};
    return true;
  case 6:
    dtype = {"int16", 2, true};
    return true;
  case 7:
    dtype = {"uint16", 2, true};
    return true;
  case 8:
    dtype = {"uint32", 4, true};
    return true;
  default:
    return false;
  }
}

float decode_float16(uint16_t bits) {
  uint32_t sign = static_cast<uint32_t>(bits & 0x8000) << 16;
  uint32_t exponent = (bits >> 10) & 0x1f;
  uint32_t mantissa = bits & 0x03ff;
  uint32_t result = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      result = sign;
    } else {
      exponent = 113;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ff;
      result = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1f) {
    result = sign | 0x7f800000 | (mantissa << 13);
  } else {
    result = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  float value = 0;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

template <typename T>
T read_tensor_value(const uint8_t *payload, uint32_t index) {
  T value{};
  std::memcpy(&value, payload + static_cast<size_t>(index) * sizeof(T),
              sizeof(T));
  return value;
}

template <typename T>
void print_integer_tensor_value(const uint8_t *payload, uint32_t index) {
  T value = read_tensor_value<T>(payload, index);
  if constexpr (std::is_signed_v<T>)
    std::printf("%lld", static_cast<long long>(value));
  else
    std::printf("%llu", static_cast<unsigned long long>(value));
}

bool print_tensor_tlv(const PrintTensorTlv *tlv, uint64_t total, uint32_t core,
                      const PrintShapeTlv *shape_tlv = nullptr) {
  constexpr uint16_t kGlobalMemoryPosition = 0;
  constexpr uint16_t kUnifiedBufferPosition = 1;
  constexpr uint32_t kTensorPayloadAlignment = 32;
  PrintTensorDType dtype{};
  if (total < sizeof(PrintTensorTlv) ||
      tlv->length != total - 2 * sizeof(uint32_t) ||
      !get_print_tensor_dtype(tlv->data_type, dtype) ||
      (tlv->position != kGlobalMemoryPosition &&
       tlv->position != kUnifiedBufferPosition) ||
      tlv->dump_size == 0 || tlv->dump_size % dtype.byte_width != 0 ||
      tlv->dump_size > 262112 * dtype.byte_width ||
      total != sizeof(PrintTensorTlv) +
                   align_up(tlv->dump_size, kTensorPayloadAlignment))
    return false;
  uint32_t dim = tlv->dim;
  const uint32_t *shape = tlv->shape;
  if (shape_tlv) {
    if (shape_tlv->type != static_cast<uint32_t>(FifoRecordType::Shape) ||
        shape_tlv->length != sizeof(PrintShapeTlv) - 2 * sizeof(uint32_t) ||
        shape_tlv->reserved != 0)
      return false;
    dim = shape_tlv->dim;
    shape = shape_tlv->shape;
  }
  if (dim != 1 && dim != 2)
    return false;
  uint64_t shape_elements = 1;
  for (uint32_t index = 0; index < 8; ++index) {
    const uint32_t extent = shape[index];
    if (index < dim) {
      if (extent == 0)
        return false;
      shape_elements *= extent;
    } else if (!shape_tlv && extent != 0) {
      return false;
    }
  }
  const uint32_t count = tlv->dump_size / dtype.byte_width;
  if (shape_elements < count)
    return false;
  auto *payload =
      reinterpret_cast<const uint8_t *>(tlv) + sizeof(PrintTensorTlv);
  const char *position =
      tlv->position == kGlobalMemoryPosition ? "GM" : "UB";
  std::printf(
      "DumpTensor: core=%u, data_type=%s, position=%s, shape=[", core,
      dtype.name, position);
  for (uint32_t i = 0; i < dim; ++i)
    std::printf("%s%u", i == 0 ? "" : ",", shape[i]);
  std::printf("], dump_size=%u [", count);
  for (uint32_t i = 0; i < count; ++i) {
    std::printf("%s", i == 0 ? "" : ", ");
    switch (tlv->data_type) {
    case 0:
      std::printf("%.9g",
                  static_cast<double>(read_tensor_value<float>(payload, i)));
      break;
    case 1:
      std::printf(
          "%.9g",
          static_cast<double>(
              decode_float16(read_tensor_value<uint16_t>(payload, i))));
      break;
    case 2:
      print_integer_tensor_value<int8_t>(payload, i);
      break;
    case 3:
      print_integer_tensor_value<int32_t>(payload, i);
      break;
    case 4:
      print_integer_tensor_value<uint8_t>(payload, i);
      break;
    case 6:
      print_integer_tensor_value<int16_t>(payload, i);
      break;
    case 7:
      print_integer_tensor_value<uint16_t>(payload, i);
      break;
    case 8:
      print_integer_tensor_value<uint32_t>(payload, i);
      break;
    default:
      return false;
    }
  }
  std::printf("]\n");
  return true;
}

bool print_fifo_records(const char *host_mem, const FifoData *fifo) {
  bool printed_any_record = false;
  for (uint32_t i = 0; i < fifo->record_count; ++i) {
    const char *record = host_mem + static_cast<size_t>(i) * fifo->block_length;
    auto *head = reinterpret_cast<const DebugBlockHeadInfo *>(record);
    if (head->magic != kMagic)
      continue;
    const char *ring = record + fifo->ring_buffer_offset;
    auto *write = reinterpret_cast<const DebugBlockWriteInfo *>(
        ring + fifo->ring_buffer_bytes);
    uint64_t offset = 0;
    const PrintShapeTlv *pending_shape = nullptr;
    uint64_t written =
        std::min<uint64_t>(write->bufOffset, fifo->ring_buffer_bytes);
    while (offset + 8 <= written) {
      auto type = *reinterpret_cast<const uint32_t *>(ring + offset);
      auto length =
          *reinterpret_cast<const uint32_t *>(ring + offset + sizeof(uint32_t));
      uint64_t total = 8ULL + length;
      if (length == 0 || total > written - offset) {
        if (type == static_cast<uint32_t>(FifoRecordType::Scalar))
          printed_any_record =
              print_malformed_scalar_tlv("length out of bounds", i) ||
              printed_any_record;
        break;
      }
      if (type == static_cast<uint32_t>(FifoRecordType::Scalar)) {
        pending_shape = nullptr;
        auto *tlv = reinterpret_cast<const PrintTlv *>(ring + offset);
        printed_any_record =
            print_scalar_tlv(tlv, total, i) || printed_any_record;
      } else if (type == static_cast<uint32_t>(FifoRecordType::Tensor)) {
        auto *tlv = reinterpret_cast<const PrintTensorTlv *>(ring + offset);
        printed_any_record =
            print_tensor_tlv(tlv, total, i, pending_shape) ||
            printed_any_record;
        pending_shape = nullptr;
      } else if (type == static_cast<uint32_t>(FifoRecordType::Shape)) {
        pending_shape =
            total == sizeof(PrintShapeTlv)
                ? reinterpret_cast<const PrintShapeTlv *>(ring + offset)
                : nullptr;
      } else {
        pending_shape = nullptr;
      }
      offset += total;
    }
  }
  if (!printed_any_record)
    std::printf("TLA debug: no records captured\n");
  return printed_any_record;
}

bool close(FifoData *fifo, rtStream_t stream) {
  if (!fifo)
    return true;

  bool ok = true;
  char *host_mem = nullptr;
  rtError_t error = RT_ERROR_NONE;
  aclError acl_ret = aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
  if (acl_ret != ACL_RT_SUCCESS) {
    set_acl_error("aclrtSynchronizeStream(AscDebugFifo)", acl_ret);
    ok = false;
  } else {
    error = rtMallocHost(reinterpret_cast<void **>(&host_mem), fifo->region_size, 8);
    if (error != RT_ERROR_NONE) {
      set_rt_error("rtMallocHost(AscDebugFifo finish)", error);
      ok = false;
    } else {
      error = rtMemcpy(host_mem, fifo->region_size, fifo->device_region,
                       fifo->region_size, RT_MEMCPY_DEVICE_TO_HOST);
      if (error != RT_ERROR_NONE) {
        set_rt_error("rtMemcpy(AscDebugFifo device-to-host)", error);
        ok = false;
      } else {
        print_fifo_records(host_mem, fifo);
      }
    }
  }

  rtError_t free_host_error = RT_ERROR_NONE;
  if (host_mem)
    free_host_error = rtFreeHost(host_mem);
  error = RT_ERROR_NONE;
  if (fifo->device_region)
    error = rtFree(fifo->device_region);
  delete fifo;
  if (free_host_error != RT_ERROR_NONE) {
    if (ok)
      set_rt_error("rtFreeHost(AscDebugFifo finish)", free_host_error);
    ok = false;
  }
  if (error != RT_ERROR_NONE) {
    if (ok)
      set_rt_error("rtFree(AscDebugFifo)", error);
    ok = false;
  }
  return ok;
}

} // namespace AscDebugFifo

} // namespace internal
} // namespace cce

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

bool contains_bytes(const char *buffer, size_t buffer_size, const char *needle) {
  size_t needle_size = std::strlen(needle);
  if (needle_size == 0 || buffer_size < needle_size)
    return false;
  const char *end = buffer + buffer_size - needle_size + 1;
  for (const char *cur = buffer; cur < end; ++cur) {
    if (std::memcmp(cur, needle, needle_size) == 0)
      return true;
  }
  return false;
}

bool uses_asc_debug_fifo(const char *buffer, size_t buffer_size) {
  // CANN 9.1 does not retain the diagnostics section-name string in every
  // kernel binary, but FIFO-enabled kernels retain this transport global.
  return contains_bytes(buffer, buffer_size, "g_sysPrintFifoSpace");
}

bool uses_print_tensor(const char *buffer, size_t buffer_size) {
  return contains_bytes(buffer, buffer_size, "tla_print_tensor_abi_v3");
}

bool validate_debug_print_fifo_contract(const std::vector<uint64_t> &values,
                                        bool expects_debug_fifo,
                                        bool binary_uses_debug_fifo) {
  if (expects_debug_fifo != binary_uses_debug_fifo) {
    g_last_error =
        "debug print FIFO intent does not match registered binary metadata";
    return false;
  }
  if (!expects_debug_fifo)
    return true;

  if (values.empty()) {
    g_last_error = "debug print FIFO marker is missing from packed kernel arguments";
    return false;
  }
  if (values.back() != cce::internal::kDebugPrintWorkspaceSentinel) {
    g_last_error = "debug print FIFO marker must occupy the final packed kernel argument";
    return false;
  }
  return true;
}

bool replace_debug_print_workspace_marker(std::vector<uint64_t> &values,
                                          uint64_t workspace) {
  if (values.empty() ||
      values.back() != cce::internal::kDebugPrintWorkspaceSentinel)
    return false;
  values.back() = workspace;
  return true;
}

bool move_print_tensor_workspace_to_first_argument(
    std::vector<uint64_t> &values, uint64_t workspace) {
  if (values.empty() ||
      values.back() != cce::internal::kPrintTensorWorkspaceSentinel)
    return false;
  values.pop_back();
  values.insert(values.begin(), workspace);
  return true;
}

bool validate_native_print_tensor_contract(const std::vector<uint64_t> &values,
                                   bool expects_print_tensor,
                                   bool binary_uses_print_tensor) {
  if (expects_print_tensor != binary_uses_print_tensor) {
    g_last_error =
        "native dump intent does not match registered binary metadata";
    return false;
  }
  if (!expects_print_tensor)
    return true;
  if (values.empty() ||
      values.back() != cce::internal::kPrintTensorWorkspaceSentinel) {
    g_last_error =
        "native dump marker is missing from packed kernel arguments";
    return false;
  }
  return true;
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

extern "C" const char* tla_runtime_last_error()
{
    thread_local std::vector<char> error_buffer;
    error_buffer.assign(g_last_error.begin(), g_last_error.end());
    error_buffer.push_back('\0');
    return error_buffer.data();
}

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
    kernel.uses_asc_debug_fifo = uses_asc_debug_fifo(buffer, buffer_size);
    kernel.uses_print_tensor = uses_print_tensor(buffer, buffer_size);
    g_registered_kernels.emplace(reinterpret_cast<uint64_t>(module), std::move(kernel));
  }

  *module_out = reinterpret_cast<uint64_t>(module);
  *function_out = reinterpret_cast<uint64_t>(stub_ptr);
  return 0;
}

extern "C" int tla_runtime_launch_kernel(uint64_t function_handle, uint64_t stream_handle, int gx,
                                         int gy, int gz, const uint8_t *args, size_t arg_size,
                                         int expects_debug_fifo,
                                         int expects_print_tensor) {
  const void *function = reinterpret_cast<const void *>(function_handle);
  rtStream_t stream = reinterpret_cast<rtStream_t>(stream_handle);
  uint32_t block_num =
      static_cast<uint32_t>(gx) * static_cast<uint32_t>(gy) * static_cast<uint32_t>(gz);

  bool binary_uses_debug_fifo = false;
  bool binary_uses_print_tensor = false;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto &entry : g_registered_kernels) {
      if (entry.second.stub &&
          reinterpret_cast<uint64_t>(entry.second.stub.get()) == function_handle) {
        binary_uses_debug_fifo = entry.second.uses_asc_debug_fifo;
        binary_uses_print_tensor = entry.second.uses_print_tensor;
        break;
      }
    }
  }

  std::vector<uint64_t> values;
  if (expects_debug_fifo || expects_print_tensor) {
    if (arg_size % sizeof(uint64_t) != 0) {
      g_last_error =
          "debug workspace kernel arguments must be a multiple of 8 bytes";
      return -1;
    }
    if (args && arg_size > 0) {
      values.resize(arg_size / sizeof(uint64_t));
      std::memcpy(values.data(), args, arg_size);
    }
  }

  if (expects_print_tensor) {
    if (!binary_uses_debug_fifo) {
      g_last_error =
          "tensor print FIFO intent does not match registered binary metadata";
      return -1;
    }
  } else if (!validate_debug_print_fifo_contract(
                 values, expects_debug_fifo != 0, binary_uses_debug_fifo)) {
    return -1;
  }
  if (!validate_native_print_tensor_contract(values, expects_print_tensor != 0,
                                     binary_uses_print_tensor))
    return -1;
  if (expects_debug_fifo && expects_print_tensor) {
    g_last_error = "scalar debug FIFO and native tensor print cannot share a v1 launch";
    return -1;
  }

  cce::internal::AscDebugFifo::FifoData *asc_debug_fifo = nullptr;
  cce::internal::AscDebugFifo::FifoData *print_tensor_fifo = nullptr;
  if (expects_debug_fifo) {
    asc_debug_fifo = cce::internal::AscDebugFifo::open(block_num);
    if (!asc_debug_fifo)
      return -1;
    if (!replace_debug_print_workspace_marker(
            values, reinterpret_cast<uint64_t>(asc_debug_fifo->device_region))) {
      cce::internal::AscDebugFifo::destroy(asc_debug_fifo);
      g_last_error = "debug print FIFO marker must occupy the final packed kernel argument";
      return -1;
    }
  }

  if (expects_print_tensor) {
    print_tensor_fifo = cce::internal::AscDebugFifo::open(block_num);
    if (!print_tensor_fifo)
      return -1;
    if (!move_print_tensor_workspace_to_first_argument(
            values,
            reinterpret_cast<uint64_t>(print_tensor_fifo->device_region))) {
      cce::internal::AscDebugFifo::destroy(print_tensor_fifo);
      g_last_error =
          "tensor print FIFO marker must occupy the final packed kernel argument";
      return -1;
    }
  }

  // Debug-workspace kernels need slot-level marker replacement. Ordinary
  // kernels must preserve the exact native-width parameter buffer.
  const bool uses_debug_workspace = expects_debug_fifo || expects_print_tensor;
  void *args_array =
      uses_debug_workspace
          ? (values.empty() ? nullptr : static_cast<void *>(values.data()))
          : const_cast<uint8_t *>(args);
  const size_t launch_arg_size =
      uses_debug_workspace ? values.size() * sizeof(uint64_t) : arg_size;
  rtError_t rt_ret = rtKernelLaunch(function, block_num, args_array,
                                    launch_arg_size, nullptr, stream);
  if (rt_ret != RT_ERROR_NONE) {
    if (asc_debug_fifo)
      cce::internal::AscDebugFifo::close(asc_debug_fifo, stream);
    if (print_tensor_fifo)
      cce::internal::AscDebugFifo::destroy(print_tensor_fifo);
    set_rt_error("rtKernelLaunch", rt_ret);
    return -1;
  }

  if (print_tensor_fifo) {
    if (!cce::internal::AscDebugFifo::close(print_tensor_fifo, stream))
      return -1;
    return 0;
  }

  if (asc_debug_fifo) {
    if (!cce::internal::AscDebugFifo::close(asc_debug_fifo, stream))
      return -1;
    return 0;
  }

  aclError acl_ret = aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream_handle));
  if (acl_ret != ACL_RT_SUCCESS) {
    set_acl_error("aclrtSynchronizeStream", acl_ret);
    return -1;
  }
  return 0;
}
