#include "Dialect/Tla/IR/TlaDialect.h"

#include <string>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaOps.h"
#include "Dialect/Tla/IR/TlaTypes.h"

#include "tla/Enums.cpp.inc"

#if !defined(MLIR_ODS_FIELD_PARSER_COMPAT)
#define MLIR_ODS_FIELD_PARSER_COMPAT
namespace mlir {
template <typename T> struct ODSFieldParserCompat {
  static FailureOr<T> parse(AsmParser &parser) { return FieldParser<T, T>::parse(parser); }
};

template <> struct FieldParser<::mlir::Type, ::mlir::Type> {
  static FailureOr<::mlir::Type> parse(AsmParser &parser) {
    ::mlir::Type type;
    if (parser.parseType(type))
      return failure();
    return type;
  }
};
} // namespace mlir
#endif

// PtrTypeStorage must be complete here before addTypes<::tla::PtrType>() instantiates
// StorageUniquer (generated in Types.cpp.inc under GET_TYPEDEF_CLASSES).
#define FieldParser ODSFieldParserCompat
#define GET_TYPEDEF_CLASSES
#include "tla/Types.cpp.inc"
// Custom TypeDef parsers call TypeDef::get(), which requires the generated
// storage classes above to be complete in this translation unit.
#include "TlaTypesImpl.inc"

#define GET_ATTRDEF_CLASSES
#include "tla/Attrs.cpp.inc"
#undef FieldParser

using namespace mlir;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::tla::TlaDialect)

namespace tla {

TlaDialect::TlaDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TlaDialect>()) {
  initialize();
}

void TlaDialect::initialize() {
  allowUnknownTypes();

#define GET_TYPEDEF_LIST
  addTypes<
#include "tla/Types.cpp.inc"
      >();
#undef GET_TYPEDEF_LIST

#define GET_ATTRDEF_LIST
  addAttributes<
#include "tla/Attrs.cpp.inc"
      >();
#undef GET_ATTRDEF_LIST

#define GET_OP_LIST
  addOperations<
#include "tla/Ops.cpp.inc"
      >();
#undef GET_OP_LIST
}

Type TlaDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  Type result;
  OptionalParseResult parseResult = generatedTypeParser(parser, &mnemonic, result);
  if (!parseResult.has_value()) {
    if (mnemonic == "flag" || mnemonic == "cross_flag")
      return OpaqueType::get(StringAttr::get(getContext(), getDialectNamespace()),
                             parser.getFullSymbolSpec());
    parser.emitError(parser.getNameLoc()) << "unknown tla type: " << mnemonic;
    return Type{};
  }
  return failed(*parseResult) ? Type{} : result;
}

void TlaDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  llvm_unreachable("unknown tla type");
}

Attribute TlaDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  StringRef attrTag;
  if (parser.parseKeyword(&attrTag))
    return {};

  if (attrTag == "pipe") {
    if (parser.parseLess())
      return {};
    StringRef pipeKeyword;
    if (parser.parseKeyword(&pipeKeyword) || parser.parseGreater())
      return {};
    auto symbolized = ::symbolizePipe(pipeKeyword);
    if (!symbolized) {
      parser.emitError(parser.getNameLoc()) << "invalid tla.pipe value: " << pipeKeyword;
      return {};
    }
    return ::tla::PipeAttr::get(getContext(), *symbolized);
  }

  if (attrTag == "cross_mode") {
    if (parser.parseLess())
      return {};
    StringRef modeKeyword;
    if (parser.parseKeyword(&modeKeyword) || parser.parseGreater())
      return {};
    auto symbolized = ::symbolizeCrossMode(modeKeyword);
    if (!symbolized) {
      parser.emitError(parser.getNameLoc()) << "invalid tla.cross_mode value: " << modeKeyword;
      return {};
    }
    return ::tla::CrossModeAttr::get(getContext(), *symbolized);
  }

  if (attrTag == "quant_mode") {
    if (parser.parseLess())
      return {};
    StringRef modeKeyword;
    if (parser.parseKeyword(&modeKeyword) || parser.parseGreater())
      return {};
    auto symbolized = ::symbolizeQuantMode(modeKeyword);
    if (!symbolized) {
      parser.emitError(parser.getNameLoc()) << "invalid tla.quant_mode value: " << modeKeyword;
      return {};
    }
    return ::tla::QuantModeAttr::get(getContext(), *symbolized);
  }

  if (attrTag == "l0c2ub_mode") {
    if (parser.parseLess())
      return {};
    StringRef modeKeyword;
    if (parser.parseKeyword(&modeKeyword) || parser.parseGreater())
      return {};
    auto symbolized = ::symbolizeL0C2UBMode(modeKeyword);
    if (!symbolized) {
      parser.emitError(parser.getNameLoc()) << "invalid tla.l0c2ub_mode value: " << modeKeyword;
      return {};
    }
    return ::tla::L0C2UBModeAttr::get(getContext(), *symbolized);
  }

  StringRef mnemonic = attrTag;
  Attribute value;
  OptionalParseResult parseResult = generatedAttributeParser(parser, &mnemonic, type, value);
  if (parseResult.has_value())
    return parseResult.value() ? value : Attribute();

  parser.emitError(parser.getNameLoc()) << "unknown tla attribute: " << attrTag;
  return {};
}

void TlaDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  if (auto pipeAttr = llvm::dyn_cast<::tla::PipeAttr>(attr)) {
    printer << "pipe<" << ::stringifyPipe(pipeAttr.getPipe()) << ">";
    return;
  }
  if (auto modeAttr = llvm::dyn_cast<::tla::CrossModeAttr>(attr)) {
    printer << "cross_mode<" << ::stringifyCrossMode(modeAttr.getCrossMode()) << ">";
    return;
  }
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;

  llvm_unreachable("unknown tla attribute");
}

} // namespace tla
