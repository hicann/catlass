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
template <typename T>
struct ODSFieldParserCompat {
    static FailureOr<T> parse(AsmParser& parser)
    {
        return FieldParser<T, T>::parse(parser);
    }
};

template <>
struct FieldParser<::mlir::Type, ::mlir::Type> {
    static FailureOr<::mlir::Type> parse(AsmParser& parser)
    {
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

TlaDialect::TlaDialect(MLIRContext* context) : Dialect(getDialectNamespace(), context, TypeID::get<TlaDialect>())
{
    initialize();
}

void TlaDialect::initialize()
{
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

} // namespace tla
