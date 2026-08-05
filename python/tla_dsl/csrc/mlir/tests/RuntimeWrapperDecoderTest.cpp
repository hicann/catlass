#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

// The decoder intentionally remains implementation-local to RuntimeWrapper.
// Include it here to exercise its byte-level CANN-compatible TLV contract
// without adding a production parser interface solely for tests.
#include "../lib/Tools/RuntimeWrapper.cpp"

namespace {

using cce::internal::AscDebugFifo::PrintTensorTlv;
using cce::internal::AscDebugFifo::PrintShapeTlv;
using cce::internal::AscDebugFifo::FifoRecordType;
using cce::internal::AscDebugFifo::kPrintTensorDescriptorNamespace;
using cce::internal::AscDebugFifo::PrintFormatResult;
using cce::internal::AscDebugFifo::PrintTlv;

std::vector<uint8_t> scalar_tlv(const char *format,
                                const std::vector<uint64_t> &slots) {
  constexpr size_t kArgumentOffset = 16 + 8;
  const size_t kFormatOffset =
      kArgumentOffset + slots.size() * sizeof(uint64_t);
  const size_t format_length = std::strlen(format) + 1;
  const size_t raw_size = kFormatOffset + format_length;
  const size_t record_size =
      (raw_size + sizeof(uint64_t) - 1) & ~(sizeof(uint64_t) - 1);
  std::vector<uint8_t> bytes(record_size, 0);
  auto *tlv = reinterpret_cast<PrintTlv *>(bytes.data());
  tlv->type = static_cast<uint32_t>(FifoRecordType::Scalar);
  tlv->length = static_cast<uint32_t>(bytes.size() - 8);
  tlv->blockIdx = 0;
  tlv->fmtOffset = sizeof(uint64_t) + slots.size() * sizeof(uint64_t);
  if (!slots.empty())
    std::memcpy(bytes.data() + kArgumentOffset, slots.data(),
                slots.size() * sizeof(uint64_t));
  std::memcpy(bytes.data() + kFormatOffset, format, format_length);
  return bytes;
}

std::vector<uint8_t> scalar_tlv(const char *format, uint64_t slot) {
  return scalar_tlv(format, std::vector<uint64_t>{slot});
}

std::vector<uint8_t> literal_scalar_tlv(const char *format) {
  constexpr size_t kFormatOffset = 24;
  const size_t format_length = std::strlen(format) + 1;
  const size_t raw_size = kFormatOffset + format_length;
  const size_t record_size =
      (raw_size + sizeof(uint64_t) - 1) & ~(sizeof(uint64_t) - 1);
  std::vector<uint8_t> bytes(record_size, 0);
  auto *tlv = reinterpret_cast<PrintTlv *>(bytes.data());
  tlv->type = static_cast<uint32_t>(FifoRecordType::Scalar);
  tlv->length = static_cast<uint32_t>(bytes.size() - 8);
  tlv->blockIdx = 0;
  tlv->fmtOffset = 8;
  std::memcpy(bytes.data() + kFormatOffset, format, format_length);
  return bytes;
}

std::vector<uint8_t> tensor_tlv(uint32_t data_type = 0,
                                uint32_t element_width = sizeof(float),
                                uint16_t call = 0, uint16_t block = 0,
                                uint16_t position = 0,
                                int32_t subblock = -1) {
  constexpr uint32_t kValueCount = 4;
  constexpr uint32_t kPayloadBytes = 32;
  std::vector<uint8_t> bytes(sizeof(PrintTensorTlv) + kPayloadBytes, 0);
  auto *tlv = reinterpret_cast<PrintTensorTlv *>(bytes.data());
  tlv->type = static_cast<uint32_t>(FifoRecordType::Tensor);
  tlv->length = static_cast<uint32_t>(bytes.size() - 8);
  tlv->data_type = data_type;
  const uint32_t subblockTag =
      subblock < 0 ? 0U : static_cast<uint32_t>(subblock + 1);
  tlv->desc =
      kPrintTensorDescriptorNamespace | (subblockTag << 16) | call;
  tlv->block_idx = block;
  tlv->position = position;
  tlv->dump_size = kValueCount * element_width;
  if (data_type == 0) {
    auto *values =
        reinterpret_cast<float *>(bytes.data() + sizeof(PrintTensorTlv));
    for (uint32_t i = 0; i < kValueCount; ++i)
      values[i] = static_cast<float>(i);
  }
  return bytes;
}

PrintShapeTlv shape_tlv() {
  PrintShapeTlv tlv{};
  tlv.type = static_cast<uint32_t>(FifoRecordType::Shape);
  tlv.length = sizeof(PrintShapeTlv) - 8;
  tlv.dim = 2;
  tlv.shape[0] = 2;
  tlv.shape[1] = 2;
  return tlv;
}

std::vector<uint8_t> tensor_record(
    std::vector<uint8_t> tensor = tensor_tlv(),
    PrintShapeTlv shape = shape_tlv()) {
  std::vector<uint8_t> bytes(sizeof(shape));
  std::memcpy(bytes.data(), &shape, sizeof(shape));
  bytes.insert(bytes.end(), tensor.begin(), tensor.end());
  return bytes;
}

bool expect(bool condition, const char *message) {
  if (condition)
    return true;
  std::fprintf(stderr, "RuntimeWrapperDecoderTest failure: %s\n", message);
  return false;
}

std::string capture_stdout(const std::function<bool()> &callback,
                           bool *result = nullptr) {
  int descriptors[2];
  if (pipe(descriptors) != 0)
    return {};
  std::fflush(stdout);
  const int saved_stdout = dup(STDOUT_FILENO);
  if (saved_stdout < 0 || dup2(descriptors[1], STDOUT_FILENO) < 0) {
    close(descriptors[0]);
    close(descriptors[1]);
    if (saved_stdout >= 0)
      close(saved_stdout);
    return {};
  }
  close(descriptors[1]);
  const bool callback_result = callback();
  if (result)
    *result = callback_result;
  std::fflush(stdout);
  dup2(saved_stdout, STDOUT_FILENO);
  close(saved_stdout);
  std::string output;
  char buffer[256];
  for (ssize_t count; (count = read(descriptors[0], buffer, sizeof(buffer))) > 0;)
    output.append(buffer, static_cast<size_t>(count));
  close(descriptors[0]);
  return output;
}

struct SyntheticFifo {
  cce::internal::AscDebugFifo::FifoData fifo;
  std::vector<char> bytes;
};

SyntheticFifo tensor_fifo(const std::vector<uint8_t> &ring_bytes,
                          uint64_t write_offset =
                              std::numeric_limits<uint64_t>::max(),
                          uint32_t launch_blocks = 1,
                          uint32_t record_slot = 0,
                          bool mixed_handoff = false) {
  using namespace cce::internal::AscDebugFifo;
  SyntheticFifo result;
  result.fifo.record_count = record_slot + 1;
  result.fifo.launch_block_count = launch_blocks;
  result.fifo.mixed_handoff = mixed_handoff;
  result.fifo.ring_buffer_offset = debug_fifo_ring_offset();
  result.fifo.ring_buffer_bytes = kRingBufferBytes;
  result.fifo.block_length =
      result.fifo.ring_buffer_offset + result.fifo.ring_buffer_bytes +
      sizeof(DebugBlockWriteInfo);
  result.fifo.region_size =
      static_cast<size_t>(result.fifo.block_length) * result.fifo.record_count;
  result.bytes.resize(result.fifo.region_size, 0);
  for (uint32_t i = 0; i < result.fifo.record_count; ++i) {
    auto *record = result.bytes.data() +
                   static_cast<size_t>(i) * result.fifo.block_length;
    auto *head = reinterpret_cast<DebugBlockHeadInfo *>(record);
    head->magic = kMagic;
    head->coreId = i;
    auto *write = reinterpret_cast<DebugBlockWriteInfo *>(
        record + result.fifo.ring_buffer_offset +
        result.fifo.ring_buffer_bytes);
    write->type = static_cast<uint32_t>(FifoRecordType::BufIn);
    write->length = 16;
  }
  auto *record = result.bytes.data() +
                 static_cast<size_t>(record_slot) * result.fifo.block_length;
  auto *ring = record + result.fifo.ring_buffer_offset;
  if (ring_bytes.size() <= result.fifo.ring_buffer_bytes)
    std::memcpy(ring, ring_bytes.data(), ring_bytes.size());
  auto *write = reinterpret_cast<DebugBlockWriteInfo *>(
      ring + result.fifo.ring_buffer_bytes);
  write->bufOffset =
      write_offset == std::numeric_limits<uint64_t>::max()
          ? ring_bytes.size()
          : write_offset;
  return result;
}

bool render_tensor_fifo(SyntheticFifo &fifo, std::string &output) {
  bool result = false;
  output = capture_stdout(
      [&] { return print_fifo_records(fifo.bytes.data(), &fifo.fifo, true); },
      &result);
  return result;
}

bool expect_rejected_without_output(SyntheticFifo &fifo,
                                    const char *message) {
  g_last_error.clear();
  std::string output;
  const bool result = render_tensor_fifo(fifo, output);
  return expect(!result, message) &&
         expect(output.empty(), "malformed FIFO emitted partial output") &&
         expect(!g_last_error.empty(), "malformed FIFO did not set last error");
}

bool expect_rejected_ring(
    const std::vector<uint8_t> &bytes, const char *message,
    uint64_t write_offset = std::numeric_limits<uint64_t>::max(),
    uint32_t launch_blocks = 1) {
  auto fifo = tensor_fifo(bytes, write_offset, launch_blocks);
  return expect_rejected_without_output(fifo, message);
}

template <typename Mutator>
bool expect_rejected_tensor(const char *message, Mutator mutate) {
  auto bytes = tensor_tlv();
  mutate(*reinterpret_cast<PrintTensorTlv *>(bytes.data()));
  return expect_rejected_ring(tensor_record(std::move(bytes)), message);
}

bool validate_debug_print_fifo_contract(const std::vector<uint64_t> &values,
                                        bool expects_debug_fifo,
                                        bool binary_uses_debug_fifo);
bool replace_debug_print_workspace_marker(std::vector<uint64_t> &values,
                                          uint64_t workspace);
bool replace_print_tensor_workspace_marker(std::vector<uint64_t> &values,
                                           uint64_t workspace);

} // namespace

int main() {
  using namespace cce::internal::AscDebugFifo;

  constexpr uint64_t kMarker = cce::internal::kDebugPrintWorkspaceSentinel;
  constexpr uint64_t kWorkspace = 0x123456789abcdef0ULL;

  {
    g_last_error = "first runtime error";
    const char *message = tla_runtime_last_error();
    g_last_error = "new runtime error";
    if (!expect(std::strcmp(message, "first runtime error") == 0,
                "last-error ABI pointer changed with string storage"))
      return 1;
  }
  {
    constexpr char kOrdinary[] = "ordinary";
    constexpr char kLegacy[] = "__tla_print_tensor_legacy_abi";
    constexpr char kCurrent[] = "__tla_print_tensor_abi";
    if (!expect(!uses_print_tensor(kOrdinary, sizeof(kOrdinary)),
                "ordinary binary was classified as tensor print") ||
        !expect(!uses_print_tensor(kLegacy, sizeof(kLegacy)),
                "legacy tensor metadata satisfied the current contract") ||
        !expect(uses_print_tensor(kCurrent, sizeof(kCurrent)),
                "current tensor metadata was not recognized"))
      return 1;
  }

  {
    constexpr char kOrdinaryKernelMetadata[] =
        "__asc_debug_meta_section__\0.ParamSummary_basic_vadd";
    constexpr char kPrintfKernelMetadata[] =
        "__asc_debug_meta_section__\0g_sysPrintFifoSpace\0.ParamSummary_printf";
    constexpr char kCANN91PrintfKernelMetadata[] =
        "g_sysPrintFifoSpace\0.ParamSummary_printf";
    if (!expect(!uses_asc_debug_fifo(kOrdinaryKernelMetadata,
                                     sizeof(kOrdinaryKernelMetadata)),
                "ordinary CANN diagnostics metadata was classified as FIFO") ||
        !expect(uses_asc_debug_fifo(kPrintfKernelMetadata,
                                    sizeof(kPrintfKernelMetadata)),
                "printf FIFO transport metadata was not classified as FIFO") ||
        !expect(uses_asc_debug_fifo(kCANN91PrintfKernelMetadata,
                                    sizeof(kCANN91PrintfKernelMetadata)),
                "CANN 9.1 FIFO symbol without section-name string was not "
                "classified as FIFO"))
      return 1;
  }

  {
    std::vector<uint64_t> values{kMarker, 17, kMarker};
    if (!expect(validate_debug_print_fifo_contract(values, true, true),
                "user sentinel before final debug FIFO marker was rejected") ||
        !expect(replace_debug_print_workspace_marker(values, kWorkspace),
                "final debug FIFO marker was not replaced") ||
        !expect(values == std::vector<uint64_t>{kMarker, 17, kWorkspace},
                "debug FIFO replacement changed a user sentinel"))
      return 1;
  }
  {
    constexpr uint64_t kPrintMarker =
        cce::internal::kPrintTensorWorkspaceSentinel;
    std::vector<uint64_t> pureValues{17, kPrintMarker};
    if (!expect(move_print_tensor_workspace_to_first_argument(pureValues,
                                                              kWorkspace),
                "pure tensor workspace was not moved to argument zero") ||
        !expect(pureValues == std::vector<uint64_t>{kWorkspace, 17},
                "pure tensor workspace placement is incorrect"))
      return 1;

    std::vector<uint64_t> mixedValues{17, kPrintMarker};
    if (!expect(replace_print_tensor_workspace_marker(mixedValues, kWorkspace),
                "mixed tensor workspace marker was not replaced") ||
        !expect(mixedValues == std::vector<uint64_t>{17, kWorkspace},
                "mixed tensor workspace was not kept trailing"))
      return 1;
  }
  {
    std::vector<uint64_t> values{17};
    if (!expect(!validate_debug_print_fifo_contract(values, true, true),
                "missing debug FIFO marker was accepted"))
      return 1;
  }
  {
    std::vector<uint64_t> values{kMarker, 17};
    if (!expect(!validate_debug_print_fifo_contract(values, true, true),
                "non-final debug FIFO marker was accepted"))
      return 1;
  }
  {
    std::vector<uint64_t> values{17, kMarker};
    if (!expect(!validate_debug_print_fifo_contract(values, true, false),
                "expected FIFO without binary metadata was accepted"))
      return 1;
  }
  {
    std::vector<uint64_t> values{17};
    if (!expect(!validate_debug_print_fifo_contract(values, false, true),
                "binary FIFO metadata without host intent was accepted"))
      return 1;
  }
  {
    std::vector<uint64_t> values{17, kMarker};
    if (!expect(validate_debug_print_fifo_contract(values, false, false),
                "non-print scalar equal to marker was treated as debug FIFO") ||
        !expect(values.back() == kMarker,
                "non-print scalar equal to marker was modified"))
      return 1;
  }

  if (!expect(is_supported_scalar_printf_format("x=%d", 4),
              "i32 format was rejected") ||
      !expect(is_supported_scalar_printf_format("x=%u", 4),
              "unsigned format was rejected") ||
      !expect(is_supported_scalar_printf_format("v=%f", 4),
              "f32 format was rejected") ||
      !expect(is_supported_scalar_printf_format("x=%d y=%f",
                                                std::strlen("x=%d y=%f")),
              "mixed generated format was rejected") ||
      !expect(is_supported_scalar_printf_format(
                  "repeat %d %d", std::strlen("repeat %d %d")),
              "repeated generated format was rejected") ||
      !expect(is_supported_scalar_printf_format(
                  "progress=50%% x=%d", std::strlen("progress=50%% x=%d")),
              "literal percent generated format was rejected") ||
      !expect(is_supported_scalar_printf_format("braces={} x=%d",
                                                std::strlen("braces={} x=%d")),
              "escaped braces generated format was rejected") ||
      !expect(!is_supported_scalar_printf_format("ptr=%p", 6),
              "legacy pointer format was accepted") ||
      !expect(!is_supported_scalar_printf_format("hex=%x", 6),
              "native hex format was accepted") ||
      !expect(!is_supported_scalar_printf_format("bad %q", 6),
              "unsupported native format was accepted") ||
      !expect(!is_supported_scalar_printf_format("progress=50%",
                                                 std::strlen("progress=50%")),
              "raw literal percent was accepted"))
    return 1;

  uint64_t i32_slot = static_cast<uint32_t>(-37);
  if (!expect(format_scalar_printf(
                  "x=%d", 4, reinterpret_cast<const uint8_t *>(&i32_slot),
                  sizeof(i32_slot)) == PrintFormatResult::Printed,
              "valid i32 slot did not print"))
    return 1;
  constexpr uint32_t kUnsignedMaxima[] = {0, 255, 65535, 4294967295U};
  for (uint32_t value : kUnsignedMaxima) {
    uint64_t unsigned_slot = value;
    if (!expect(load_print_slot_unsigned(unsigned_slot) == value,
                "unsigned slot changed its value") ||
        !expect(format_scalar_printf(
                    "x=%u", 4,
                    reinterpret_cast<const uint8_t *>(&unsigned_slot),
                    sizeof(unsigned_slot)) == PrintFormatResult::Printed,
                "valid unsigned slot did not print"))
      return 1;
  }
  float f32_value = 1.25f;
  uint64_t f32_slot = 0;
  std::memcpy(&f32_slot, &f32_value, sizeof(f32_value));
  const uint64_t promoted_f16_slot = f32_slot;
  float negative_f32_value = -2.5f;
  uint64_t negative_f32_slot = 0;
  std::memcpy(&negative_f32_slot, &negative_f32_value,
              sizeof(negative_f32_value));
  uint64_t slots[] = {i32_slot, f32_slot, static_cast<uint32_t>(11)};
  if (!expect(format_scalar_printf(
                  "v=%f", 4, reinterpret_cast<const uint8_t *>(&f32_slot),
                  sizeof(f32_slot)) == PrintFormatResult::Printed,
              "valid f32 slot did not print") ||
      !expect(format_scalar_printf(
                  "x=%d y=%f", std::strlen("x=%d y=%f"),
                  reinterpret_cast<const uint8_t *>(slots),
                  2 * sizeof(uint64_t)) == PrintFormatResult::Printed,
              "valid mixed slots did not print") ||
      !expect(format_scalar_printf(
                  "repeat %d %d", std::strlen("repeat %d %d"),
                  reinterpret_cast<const uint8_t *>(slots),
                  2 * sizeof(uint64_t)) == PrintFormatResult::Printed,
              "valid repeated slots did not print") ||
      !expect(format_scalar_printf(
                  "progress=50%% x=%d", std::strlen("progress=50%% x=%d"),
                  reinterpret_cast<const uint8_t *>(slots),
                  sizeof(uint64_t)) == PrintFormatResult::Printed,
              "literal percent generated format did not print") ||
      !expect(format_scalar_printf(
                  "braces={} x=%d", std::strlen("braces={} x=%d"),
                  reinterpret_cast<const uint8_t *>(slots),
                  sizeof(uint64_t)) == PrintFormatResult::Printed,
              "escaped braces generated format did not print") ||
      !expect(format_scalar_printf("x=%d", 4, nullptr, 0) ==
                  PrintFormatResult::Malformed,
              "truncated scalar slot was accepted") ||
      !expect(format_scalar_printf("x=%u", 4, nullptr, 0) ==
                  PrintFormatResult::Malformed,
              "truncated unsigned scalar slot was accepted") ||
      !expect(format_scalar_printf(
                  "x=%d y=%f", std::strlen("x=%d y=%f"),
                  reinterpret_cast<const uint8_t *>(slots),
                  sizeof(uint64_t)) == PrintFormatResult::Malformed,
              "missing mixed scalar slot was accepted") ||
      !expect(format_scalar_printf(
                  "hex=%x", 6, reinterpret_cast<const uint8_t *>(slots),
                  sizeof(uint64_t)) == PrintFormatResult::Unsupported,
               "unsupported native format sequence was not rejected"))
    return 1;

  {
    auto palette = scalar_tlv(
        "all=%d %d %d %u %u %u %f %f",
        {
            static_cast<uint64_t>(static_cast<int64_t>(-37)),
            static_cast<uint64_t>(static_cast<int64_t>(-30000)),
            0,
            255,
            65535,
            4294967295U,
            promoted_f16_slot,
            negative_f32_slot,
        });
    bool rendered = false;
    const std::string output = capture_stdout(
        [&] {
          return print_scalar_tlv(
              reinterpret_cast<const PrintTlv *>(palette.data()),
              palette.size(), 7);
        },
        &rendered);
    constexpr char kExpected[] =
        "TLA printf: core=7 block=0 "
        "all=-37 -30000 0 255 65535 4294967295 1.250000 -2.500000\n";
    if (!expect(rendered, "eight-type scalar palette TLV was rejected") ||
        !expect(output == kExpected,
                "eight-type scalar palette output was not exact"))
      return 1;
  }

  auto valid = scalar_tlv("x=%d", i32_slot);
  auto valid_unsigned = scalar_tlv("x=%u", 4294967295U);
  if (!expect(print_scalar_tlv(reinterpret_cast<const PrintTlv *>(valid.data()),
                               valid.size(), 7),
              "valid scalar TLV was rejected") ||
      !expect(print_scalar_tlv(
                  reinterpret_cast<const PrintTlv *>(valid_unsigned.data()),
                  valid_unsigned.size(), 7),
              "valid unsigned scalar TLV was rejected") ||
      !expect(print_scalar_tlv(reinterpret_cast<const PrintTlv *>(valid.data()),
                               sizeof(PrintTlv) - 1, 7),
              "short scalar TLV was not diagnosed"))
    return 1;
  auto out_of_bounds = valid;
  reinterpret_cast<PrintTlv *>(out_of_bounds.data())->fmtOffset = 4096;
  if (!expect(print_scalar_tlv(
                  reinterpret_cast<const PrintTlv *>(out_of_bounds.data()),
                  out_of_bounds.size(), 7),
              "out-of-bounds format offset was not diagnosed"))
    return 1;

  {
    std::vector<uint8_t> records;
    for (auto tensor : {
             tensor_tlv(0, sizeof(float), 3, 1),
             tensor_tlv(0, sizeof(float), 4, 0),
             tensor_tlv(0, sizeof(float), 4, 1, 1, 1),
         }) {
      auto record = tensor_record(std::move(tensor));
      records.insert(records.end(), record.begin(), record.end());
    }
    auto fifo = tensor_fifo(records, records.size(), 2, 1);
    std::string output;
    if (!expect(render_tensor_fifo(fifo, output),
                "valid multi-record float32 tensor FIFO was rejected") ||
        !expect(output.find("call=3, block=1") != std::string::npos,
                "valid tensor FIFO lost record identity") ||
        !expect(output.find("call=4, block=0") != std::string::npos,
                "valid tensor FIFO lost a second record") ||
        !expect(output.find("call=4, block=1, subblock=1, data_type=float32, "
                           "position=UB") != std::string::npos,
                "valid tensor FIFO lost UB subblock metadata"))
      return 1;
  }
  {
    auto bytes =
        tensor_record(tensor_tlv(0, sizeof(float), 0, 1, 1, 1));
    auto fifo = tensor_fifo(bytes, bytes.size(), 1, 1, true);
    std::string output;
    if (!expect(render_tensor_fifo(fifo, output),
                "valid mixed AIV subblock record was rejected") ||
        !expect(output.find("call=0, block=0, subblock=1") !=
                    std::string::npos,
                "mixed AIV native block index was not normalized"))
      return 1;
  }

  auto literal = literal_scalar_tlv("hello");
  if (!expect(print_scalar_tlv(
                  reinterpret_cast<const PrintTlv *>(literal.data()),
                  literal.size(), 7),
              "valid literal string scalar TLV was rejected"))
    return 1;
  auto literal_percent = literal_scalar_tlv("progress=50%%");
  if (!expect(print_scalar_tlv(
                  reinterpret_cast<const PrintTlv *>(literal_percent.data()),
                  literal_percent.size(), 7),
              "valid generated literal-percent scalar TLV was rejected"))
    return 1;
  auto raw_percent = literal_scalar_tlv("progress=50%");
  if (!expect(!print_scalar_tlv(
                  reinterpret_cast<const PrintTlv *>(raw_percent.data()),
                  raw_percent.size(), 7),
              "raw literal-percent scalar TLV was accepted"))
    return 1;
  auto mixed = scalar_tlv("x=%d y=%f", {i32_slot, f32_slot});
  if (!expect(print_scalar_tlv(
                  reinterpret_cast<const PrintTlv *>(mixed.data()),
                  mixed.size(), 7),
              "valid mixed scalar TLV was rejected"))
    return 1;
  auto missing_slot = scalar_tlv("x=%d y=%f", std::vector<uint64_t>{i32_slot});
  if (!expect(print_scalar_tlv(
                  reinterpret_cast<const PrintTlv *>(missing_slot.data()),
                  missing_slot.size(), 7),
              "missing-slot scalar TLV was not diagnosed"))
    return 1;
  auto unsupported = scalar_tlv("hex=%x", std::vector<uint64_t>{i32_slot});
  if (!expect(!print_scalar_tlv(
                  reinterpret_cast<const PrintTlv *>(unsupported.data()),
                  unsupported.size(), 7),
              "unsupported native scalar TLV was accepted"))
    return 1;

  constexpr struct {
    uint32_t data_type;
    uint32_t element_width;
    const char *name;
  } kRequiredTensorDTypes[] = {
      {0, 4, "float32"}, {1, 2, "float16"}, {2, 1, "int8"},
      {3, 4, "int32"},   {4, 1, "uint8"},   {6, 2, "int16"},
      {7, 2, "uint16"},  {8, 4, "uint32"},
  };
  for (const auto &dtype : kRequiredTensorDTypes) {
    auto bytes = tensor_record(
        tensor_tlv(dtype.data_type, dtype.element_width, 2, 0, 1));
    auto fifo = tensor_fifo(bytes);
    std::string output;
    if (!expect(render_tensor_fifo(fifo, output),
                "required typed tensor FIFO was rejected") ||
        !expect(output.find(dtype.name) != std::string::npos,
                "required typed tensor FIFO lost its dtype") ||
        !expect(output.find("call=2, block=0") != std::string::npos,
                "typed tensor FIFO lost record identity"))
      return 1;
  }

  if (!expect_rejected_tensor("bad descriptor was accepted",
                              [](auto &tlv) { tlv.desc = 0x50524e54; }) ||
      !expect_rejected_tensor(
          "invalid tensor subblock tag was accepted",
          [](auto &tlv) {
            tlv.desc = kPrintTensorDescriptorNamespace | (3U << 16);
          }) ||
      !expect_rejected_tensor("unsupported tensor dtype was accepted",
                              [](auto &tlv) { tlv.data_type = 5; }) ||
      !expect_rejected_tensor("invalid TLV length was accepted",
                              [](auto &tlv) { tlv.length = 0; }) ||
      !expect_rejected_tensor("invalid tensor position was accepted",
                              [](auto &tlv) { tlv.position = 2; }) ||
      !expect_rejected_tensor("invalid tensor dimension was accepted",
                              [](auto &tlv) { tlv.dim = 1; }) ||
      !expect_rejected_tensor("invalid tensor shape was accepted",
                              [](auto &tlv) { tlv.shape[0] = 1; }) ||
      !expect_rejected_tensor("zero tensor dump size was accepted",
                              [](auto &tlv) { tlv.dump_size = 0; }) ||
      !expect_rejected_tensor("misaligned tensor dump size was accepted",
                              [](auto &tlv) { tlv.dump_size = 3; }) ||
      !expect_rejected_tensor("oversized tensor dump was accepted",
                              [](auto &tlv) { tlv.dump_size = 17 * 4; }))
    return 1;

  if (!expect_rejected_ring(std::vector<uint8_t>(4, 0),
                            "truncated TLV header was accepted"))
    return 1;
  {
    auto bytes = tensor_tlv();
    bytes.resize(sizeof(PrintTensorTlv) + 4);
    if (!expect_rejected_ring(tensor_record(std::move(bytes)),
                              "truncated tensor payload was accepted"))
      return 1;
  }
  {
    auto bytes = tensor_tlv();
    reinterpret_cast<uint32_t *>(bytes.data())[0] = 99;
    if (!expect_rejected_ring(bytes,
                              "unknown tensor FIFO record was accepted"))
      return 1;
  }
  {
    auto bytes = tensor_record(tensor_tlv(0, sizeof(float), 0, 2));
    if (!expect_rejected_ring(bytes, "out-of-range tensor block was accepted",
                              bytes.size(), 2))
      return 1;
  }
  {
    auto bytes =
        tensor_record(tensor_tlv(0, sizeof(float), 0, 0, 1, 1));
    auto fifo = tensor_fifo(bytes, bytes.size(), 1, 0, true);
    if (!expect_rejected_without_output(
            fifo, "mixed AIV block/subblock mismatch was accepted"))
      return 1;
  }
  if (!expect_rejected_ring(
          {}, "ring buffer overflow was accepted",
          static_cast<uint64_t>(kRingBufferBytes) + 1))
    return 1;
  {
    auto valid_record =
        tensor_record(tensor_tlv(0, sizeof(float), 0, 0));
    auto malformed_record = tensor_tlv(0, sizeof(float), 1, 0);
    reinterpret_cast<PrintTensorTlv *>(malformed_record.data())->shape[3] = 1;
    auto malformed_pair = tensor_record(std::move(malformed_record));
    valid_record.insert(valid_record.end(), malformed_pair.begin(),
                        malformed_pair.end());
    if (!expect_rejected_ring(
            valid_record,
            "valid record before malformed record was partially accepted"))
      return 1;
  }

  auto *cleanup_only = new FifoData();
  destroy(cleanup_only);
  std::puts("runtime_wrapper_decoder_test_ok=True");
  return 0;
}
