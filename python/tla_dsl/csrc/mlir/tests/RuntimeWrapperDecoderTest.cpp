#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// The decoder intentionally remains implementation-local to RuntimeWrapper.
// Include it here to exercise its byte-level CANN-compatible TLV contract
// without adding a production parser interface solely for tests.
#include "../lib/Tools/RuntimeWrapper.cpp"

namespace {

using cce::internal::AscDebugFifo::PrintTlv;
using cce::internal::AscDebugFifo::PrintFormatResult;

std::vector<uint8_t> scalar_tlv(const char *format, uint64_t slot) {
  constexpr size_t kArgumentOffset = 16 + 8;
  constexpr size_t kFormatOffset = kArgumentOffset + 8;
  const size_t format_length = std::strlen(format) + 1;
  std::vector<uint8_t> bytes(kFormatOffset + format_length, 0);
  auto *tlv = reinterpret_cast<PrintTlv *>(bytes.data());
  tlv->length = static_cast<uint32_t>(bytes.size() - 8);
  tlv->blockIdx = 0;
  tlv->fmtOffset = 16;
  std::memcpy(bytes.data() + kArgumentOffset, &slot, sizeof(slot));
  std::memcpy(bytes.data() + kFormatOffset, format, format_length);
  return bytes;
}

bool expect(bool condition, const char *message) {
  if (condition)
    return true;
  std::fprintf(stderr, "RuntimeWrapperDecoderTest failure: %s\n", message);
  return false;
}

bool validate_debug_print_fifo_contract(const std::vector<uint64_t> &values,
                                        bool expects_debug_fifo,
                                        bool binary_uses_debug_fifo);
bool replace_debug_print_workspace_marker(std::vector<uint64_t> &values,
                                          uint64_t workspace);

} // namespace

int main() {
  using namespace cce::internal::AscDebugFifo;

  constexpr uint64_t kMarker = cce::internal::kDebugPrintWorkspaceSentinel;
  constexpr uint64_t kWorkspace = 0x123456789abcdef0ULL;

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
                "CANN 9.1 FIFO symbol without section-name string was not classified as FIFO"))
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
      !expect(is_supported_scalar_printf_format("v=%f", 4),
              "f32 format was rejected") ||
      !expect(!is_supported_scalar_printf_format("ptr=%p", 6),
              "legacy pointer format was accepted"))
    return 1;

  uint64_t i32_slot = static_cast<uint32_t>(-37);
  if (!expect(format_scalar_printf("x=%d", 4,
                                   reinterpret_cast<const uint8_t *>(&i32_slot),
                                   sizeof(i32_slot)) == PrintFormatResult::Printed,
              "valid i32 slot did not print"))
    return 1;
  float f32_value = 1.25f;
  uint64_t f32_slot = 0;
  std::memcpy(&f32_slot, &f32_value, sizeof(f32_value));
  if (!expect(format_scalar_printf("v=%f", 4,
                                   reinterpret_cast<const uint8_t *>(&f32_slot),
                                   sizeof(f32_slot)) == PrintFormatResult::Printed,
              "valid f32 slot did not print") ||
      !expect(format_scalar_printf("x=%d", 4, nullptr, 0) ==
                  PrintFormatResult::Malformed,
              "truncated scalar slot was accepted"))
    return 1;

  auto valid = scalar_tlv("x=%d", i32_slot);
  if (!expect(print_scalar_tlv(reinterpret_cast<const PrintTlv *>(valid.data()),
                               valid.size(), 7),
              "valid scalar TLV was rejected") ||
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

  auto *cleanup_only = new FifoData();
  destroy(cleanup_only);
  std::puts("runtime_wrapper_decoder_test_ok=True");
  return 0;
}
