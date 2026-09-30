//===----------------------------------------------------------------------===//
// Tla dialect definition and the shared TableGen base classes.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TlaDialect : Dialect {
  let name = "tla";
  let summary = "Tla dialect schema for TlaIR matmul prototyping.";
  let description = [{
    Tla dialect schema for TlaIR matmul prototyping.
  }];
  let cppNamespace = "::tla";
  // auto generate parseType/printType/parseAttribute/printAttribute
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class Tla_Op<string mnemonic, list<Trait> traits = []> : Op<TlaDialect, mnemonic, traits>;

class Tla_EnumAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TlaDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class Tla_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
}

class Tla_I32EnumAttr<string attrMnemonic, Tla_I32Enum enumInfo>
    : EnumAttr<TlaDialect, enumInfo, attrMnemonic> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::tla";
}
