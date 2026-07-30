#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaDialect.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "Passes.h"
#include "Passes/TlaTensorDescriptor.h"
#include "Tools/AddressSpaceConversion.h"
#include "Tools/CompilePipeline.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "pybind11/stl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlir;

namespace py = pybind11;

namespace {

InFlightDiagnostic emitBridgeError(MLIRContext *ctx, StringRef message) {
  return emitError(UnknownLoc::get(ctx), message);
}

MLIRContext *bridgeContext(MlirContext context) {
  MLIRContext *ctx = unwrap(context);
  if (!ctx)
    throw py::value_error("expected a non-null mlir.ir.Context");
  ctx->getOrLoadDialect<::tla::TlaDialect>();
  // Bridged GM memref types carry #hivm.address_space; Attribute storage requires
  // HIVMDialect to be loaded before AddressSpaceAttr::get.
  ctx->getOrLoadDialect<hivm::HIVMDialect>();
  return ctx;
}

void loadTlaDialect(MlirContext context) { (void)bridgeContext(context); }

ModuleOp moduleFromCapsule(MlirModule cModule) {
  if (mlirModuleIsNull(cModule))
    throw py::type_error("Expected an mlir.ir.Module object.");
  ModuleOp module = unwrap(cModule);
  if (!module)
    throw py::value_error("Failed to unwrap mlir.ir.Module capsule.");
  return module;
}

bool containsPassName(const std::vector<std::string> &names, Pass *pass) {
  StringRef argument = pass->getArgument();
  return llvm::any_of(names, [&](const std::string &name) { return argument == name; });
}

Type bridgeType(MlirType type, StringRef description = "mlir.ir.Type") {
  Type unwrapped = unwrap(type);
  if (!unwrapped)
    throw py::type_error(std::string("expected a non-null ") + description.str());
  return unwrapped;
}

MlirType toMlirType(Type type, StringRef description) {
  if (!type)
    throw py::value_error(std::string("failed to construct ") + description.str());
  return wrap(type);
}

ArrayRef<int64_t> treeRef(const std::vector<int64_t> &tree, StringRef description) {
  if (tree.empty())
    throw py::value_error(description.str() + " must be non-empty");
  return ArrayRef<int64_t>(tree.data(), tree.size());
}

::std::optional<::AddressSpace> parseRequiredAddressSpace(const std::string &addrspaceText) {
  auto addrspace = ::symbolizeAddressSpace(addrspaceText);
  if (!addrspace)
    throw py::value_error("unknown Tla address space: " + addrspaceText);
  return addrspace;
}

::std::optional<::LayoutTag> parseRequiredLayoutTag(const std::string &layoutText) {
  auto layout = ::symbolizeLayoutTag(layoutText);
  if (!layout)
    throw py::value_error("unknown Tla layout tag: " + layoutText);
  return layout;
}

::tla::ShapeType makeShape(MLIRContext *ctx, const std::vector<int64_t> &tree,
                           StringRef diagnostic) {
  return ::tla::ShapeType::getChecked([&] { return emitBridgeError(ctx, diagnostic); }, ctx,
                                      treeRef(tree, "tla.shape"));
}

::tla::StrideType makeStride(MLIRContext *ctx, const std::vector<int64_t> &tree,
                             StringRef diagnostic) {
  return ::tla::StrideType::getChecked([&] { return emitBridgeError(ctx, diagnostic); }, ctx,
                                       treeRef(tree, "tla.stride"));
}

::tla::CoordType makeCoord(MLIRContext *ctx, const std::vector<int64_t> &tree,
                           StringRef diagnostic) {
  return ::tla::CoordType::getChecked([&] { return emitBridgeError(ctx, diagnostic); }, ctx,
                                      treeRef(tree, "tla.coord"));
}

MlirType shapeTypeGet(MlirContext context, const std::vector<int64_t> &tree) {
  MLIRContext *ctx = bridgeContext(context);
  return toMlirType(makeShape(ctx, tree, "invalid tla.shape type bridge input"), "tla.shape");
}

MlirType coordTypeGet(MlirContext context, const std::vector<int64_t> &tree) {
  MLIRContext *ctx = bridgeContext(context);
  return toMlirType(makeCoord(ctx, tree, "invalid tla.coord type bridge input"), "tla.coord");
}

MlirType strideTypeGet(MlirContext context, const std::vector<int64_t> &tree) {
  MLIRContext *ctx = bridgeContext(context);
  return toMlirType(makeStride(ctx, tree, "invalid tla.stride type bridge input"), "tla.stride");
}

template <typename TlaTypeT> bool typeIs(MlirType type) {
  return static_cast<bool>(dyn_cast_or_null<TlaTypeT>(bridgeType(type)));
}

template <typename TlaTypeT> TlaTypeT checkedTlaType(MlirType type, StringRef description) {
  auto tlaType = dyn_cast_or_null<TlaTypeT>(bridgeType(type));
  if (!tlaType)
    throw py::type_error(std::string("expected ") + description.str());
  return tlaType;
}

MlirType ptrTypeGet(MlirContext context, MlirType pointeeType, const std::string &addrspaceText,
                    unsigned alignment) {
  MLIRContext *ctx = bridgeContext(context);
  Type pointee = bridgeType(pointeeType, "pointee type");
  auto addrspace = parseRequiredAddressSpace(addrspaceText);
  return toMlirType(::tla::PtrType::get(ctx, pointee, *addrspace, alignment), "tla.ptr");
}

MlirType ptrPointeeTypeGet(MlirType ptrType) {
  auto ptr = checkedTlaType<::tla::PtrType>(ptrType, "!tla.ptr type");
  return toMlirType(ptr.getPointee(), "ptr pointee type");
}

std::string ptrAddrspace(MlirType ptrType) {
  auto ptr = checkedTlaType<::tla::PtrType>(ptrType, "!tla.ptr type");
  return stringifyAddressSpace(ptr.getAddrspace()).str();
}

unsigned ptrAlignment(MlirType ptrType) {
  auto ptr = checkedTlaType<::tla::PtrType>(ptrType, "!tla.ptr type");
  return ptr.getAlignment();
}

MlirType tensorPtrTypeGet(MlirType tensorType) {
  auto tensor = checkedTlaType<::tla::TlaTensorType>(tensorType, "!tla.tensor type");
  return toMlirType(tensor.getPtr(), "tla.tensor backing ptr type");
}

MlirType layoutTypeFromComponentsGet(MlirContext context, MlirType shapeType, MlirType strideType,
                                     py::object originShapeType, const std::string &layoutText) {
  MLIRContext *ctx = bridgeContext(context);
  auto shape = checkedTlaType<::tla::ShapeType>(shapeType, "!tla.shape type");
  auto stride = checkedTlaType<::tla::StrideType>(strideType, "!tla.stride type");
  ::tla::ShapeType origin = shape;
  if (!originShapeType.is_none())
    origin = checkedTlaType<::tla::ShapeType>(originShapeType.cast<MlirType>(), "!tla.shape type");
  auto layout = parseRequiredLayoutTag(layoutText);
  Type type = ::tla::LayoutType::getChecked(
      [&] { return emitBridgeError(ctx, "invalid tla.layout component type bridge input"); }, ctx,
      shape, stride, origin, *layout);
  return toMlirType(type, "tla.layout");
}

MlirType layoutTypeGet(MlirContext context, const std::vector<int64_t> &shapeTree,
                       const std::vector<int64_t> &strideTree, py::object originTreeObject,
                       const std::string &layoutText) {
  MLIRContext *ctx = bridgeContext(context);
  auto shape = makeShape(ctx, shapeTree, "invalid tla.layout shape bridge input");
  auto stride = makeStride(ctx, strideTree, "invalid tla.layout stride bridge input");
  ::tla::ShapeType origin = shape;
  if (!originTreeObject.is_none()) {
    auto originTree = originTreeObject.cast<std::vector<int64_t>>();
    origin = makeShape(ctx, originTree, "invalid tla.layout origin bridge input");
  }
  auto layout = parseRequiredLayoutTag(layoutText);
  Type type = ::tla::LayoutType::getChecked(
      [&] { return emitBridgeError(ctx, "invalid tla.layout type bridge input"); }, ctx, shape,
      stride, origin, *layout);
  return toMlirType(type, "tla.layout");
}

MlirType vectorSSAElementTypeGet(MlirType vectorType) {
  auto vector = checkedTlaType<::tla::VectorSSAType>(vectorType, "!tla.vector type");
  return toMlirType(vector.getElementType(), "vector element type");
}

py::object vectorSSAValidLanesGet(MlirType vectorType) {
  auto vector = checkedTlaType<::tla::VectorSSAType>(vectorType, "!tla.vector type");
  if (vector.getValidLanes() == ShapedType::kDynamic)
    return py::none();
  return py::int_(vector.getValidLanes());
}

MlirType vectorSSATypeGet(MlirContext context, py::object validLanes,
                          MlirType elementType) {
  MLIRContext *ctx = bridgeContext(context);
  int64_t lanes = validLanes.is_none() ? ShapedType::kDynamic
                                       : validLanes.cast<int64_t>();
  Type element = bridgeType(elementType, "element type");
  Type type = ::tla::VectorSSAType::getChecked(
      [&] { return emitBridgeError(ctx, "invalid tla.vector type bridge input"); },
      ctx, lanes, element);
  return toMlirType(type, "tla.vector");
}

MlirType maskSSATypeGet(MlirContext context, int64_t physicalLanes) {
  MLIRContext *ctx = bridgeContext(context);
  Type type = ::tla::MaskSSAType::getChecked(
      [&] {
        return emitBridgeError(ctx, "invalid tla.mask type bridge input");
      },
      ctx, physicalLanes);
  return toMlirType(type, "tla.mask");
}

int64_t maskSSAPhysicalLanesGet(MlirType maskType) {
  return checkedTlaType<::tla::MaskSSAType>(maskType, "!tla.mask<N> type").getPhysicalLanes();
}

MlirType flagTypeGet(MlirContext context) {
  MLIRContext *ctx = bridgeContext(context);
  return toMlirType(::tla::FlagType::get(ctx), "tla.flag");
}

MlirType crossFlagTypeGet(MlirContext context, int64_t mode)
{
    MLIRContext* ctx = bridgeContext(context);
    return toMlirType(::tla::CrossFlagType::get(ctx, mode), "tla.cross_flag");
}

int64_t crossFlagMode(MlirType type)
{
    return checkedTlaType<::tla::CrossFlagType>(type, "!tla.cross_flag type").getMode();
}

MlirType mutexTypeGet(MlirContext context) {
  MLIRContext *ctx = bridgeContext(context);
  return toMlirType(::tla::MutexType::get(ctx), "tla.mutex");
}

MlirType copyL0C2DstParamsTypeGet(MlirContext context) {
  MLIRContext *ctx = bridgeContext(context);
  return toMlirType(::tla::CopyL0C2DstParamsType::get(ctx), "tla.CopyL0C2DstParams");
}

MlirType tensorTypeGet(MlirContext context, const std::vector<int64_t> &shapeTree,
                       const std::vector<int64_t> &strideTree,
                       const std::vector<int64_t> &coordTree,
                       const std::vector<int64_t> &originShapeTree, MlirType elementType,
                       const std::string &addrspaceText, const std::string &layoutText,
                       unsigned ptrAlignment) {
  MLIRContext *ctx = bridgeContext(context);
  auto shape = makeShape(ctx, shapeTree, "invalid tla.tensor shape bridge input");
  auto stride = makeStride(ctx, strideTree, "invalid tla.tensor stride bridge input");
  auto coord = makeCoord(ctx, coordTree, "invalid tla.tensor coord bridge input");
  auto originShape = makeShape(ctx, originShapeTree, "invalid tla.tensor origin bridge input");
  Type element = bridgeType(elementType, "element type");
  auto addrspace = parseRequiredAddressSpace(addrspaceText);
  auto layoutTag = parseRequiredLayoutTag(layoutText);
  auto layout = ::tla::LayoutType::getChecked(
      [&] { return emitBridgeError(ctx, "invalid tla.tensor layout bridge input"); }, ctx, shape,
      stride, originShape, *layoutTag);
  auto ptr = ::tla::PtrType::get(ctx, element, *addrspace, ptrAlignment);
  Type type = ::tla::TlaTensorType::getChecked(
      [&] { return emitBridgeError(ctx, "invalid tla.tensor type bridge input"); }, ctx, layout,
      coord, ptr);
  return toMlirType(type, "tla.tensor");
}

/// Unified dynamic GM ABI memref from ``!tla.tensor`` (schema-v4: 4 size/stride slots).
MlirType dynamicGmMemrefTypeGet(MlirType tensorType) {
  Type tlaTensor = bridgeType(tensorType, "tla.tensor type");
  if (!tlaTensor.getContext())
    throw py::value_error("expected tla.tensor type with a live MLIRContext");
  (void)bridgeContext(wrap(tlaTensor.getContext()));
  FailureOr<MemRefType> bridged = ::tla::bridgeTlaTensorType(tlaTensor);
  if (failed(bridged))
    throw py::value_error("failed to bridge tla.tensor storage type to memref");
  Type elem = bridged->getElementType();
  MLIRContext *ctx = elem.getContext();
  Attribute gmSpace = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::GM);
  SmallVector<int64_t, 4> dynShape(4, ShapedType::kDynamic);
  SmallVector<int64_t, 4> dynStrides(4, ShapedType::kDynamic);
  auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, dynStrides);
  Type memref = MemRefType::get(dynShape, elem, layout, gmSpace);
  return toMlirType(memref, "dynamic GM memref type");
}

std::optional<std::string> tlaTypeCategory(MlirType type) {
  Type unwrapped = bridgeType(type);
  if (isa<::tla::TlaTensorType>(unwrapped))
    return "tensor";
  if (isa<::tla::VectorSSAType>(unwrapped))
    return "vector_ssa";
  if (isa<::tla::MaskSSAType>(unwrapped))
    return "mask_ssa";
  if (isa<::tla::ShapeType>(unwrapped))
    return "shape";
  if (isa<::tla::CoordType>(unwrapped))
    return "coord";
  if (isa<::tla::StrideType>(unwrapped))
    return "stride";
  if (isa<::tla::LayoutType>(unwrapped))
    return "layout";
  if (isa<::tla::PtrType>(unwrapped))
    return "pointer";
  if (isa<::tla::FlagType>(unwrapped))
    return "flag";
  if (isa<::tla::CrossFlagType>(unwrapped))
    return "cross_flag";
  if (isa<::tla::MutexType>(unwrapped))
    return "mutex";
  if (isa<::tla::CopyL0C2DstParamsType>(unwrapped))
    return "CopyL0C2DstParams";
  return std::nullopt;
}

// 0 is a scalar, 1 is a launchable GM pointer, and -1 is a pointer whose
// address space cannot participate in the host launch ABI.
using KernelPointerProvenance = llvm::StringMap<SmallVector<int8_t, 8>>;

KernelPointerProvenance collectKernelPointerProvenance(ModuleOp module) {
  KernelPointerProvenance result;
  module.walk([&](FunctionOpInterface function) {
    if (function.isExternal() || function.getOperation()->hasAttr("sym_visibility") &&
                                     function.getOperation()
                                             ->getAttrOfType<StringAttr>("sym_visibility")
                                             .getValue() == "private")
      return;
    SmallVector<int8_t, 8> pointers;
    for (Type type : function.getArgumentTypes()) {
      int8_t pointerKind = 0;
      if (auto ptr = dyn_cast<::tla::PtrType>(type))
        pointerKind = ptr.getAddrspace() == ::AddressSpace::gm ? 1 : -1;
      else if (auto tensor = dyn_cast<::tla::TlaTensorType>(type))
        pointerKind =
            tensor.getPtr().getAddrspace() == ::AddressSpace::gm ? 1 : -1;
      pointers.push_back(pointerKind);
    }
    result[function.getName()] = std::move(pointers);
  });
  return result;
}

std::string printType(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return text;
}

std::optional<unsigned> scalarStorageSize(Type type) {
  if (type.isIndex())
    return 8;
  if (auto integer = dyn_cast<IntegerType>(type))
    switch (integer.getWidth()) {
    case 1:
    case 8:
      return 1;
    case 16:
      return 2;
    case 32:
      return 4;
    case 64:
      return 8;
    default:
      return std::nullopt;
    }
  if (type.isF16() || type.isBF16())
    return 2;
  if (type.isF32())
    return 4;
  return std::nullopt;
}

std::optional<py::dict> scalarAbiDescriptor(Type type) {
  py::dict descriptor;
  if (type.isIndex()) {
    descriptor["category"] = "index";
    descriptor["bit_width"] = 64;
    descriptor["integer_signedness"] = py::none();
    descriptor["float_format"] = py::none();
    return descriptor;
  }
  if (auto integer = dyn_cast<IntegerType>(type)) {
    unsigned width = integer.getWidth();
    if (width != 1 && width != 8 && width != 16 && width != 32 && width != 64)
      return std::nullopt;
    descriptor["category"] = "integer";
    descriptor["bit_width"] = width;
    descriptor["integer_signedness"] =
        integer.isSigned() ? "signed"
                           : integer.isUnsigned() ? "unsigned" : "signless";
    descriptor["float_format"] = py::none();
    return descriptor;
  }
  StringRef format;
  unsigned width = 0;
  if (type.isF16()) {
    format = "f16";
    width = 16;
  } else if (type.isBF16()) {
    format = "bf16";
    width = 16;
  } else if (type.isF32()) {
    format = "f32";
    width = 32;
  } else {
    return std::nullopt;
  }
  descriptor["category"] = "float";
  descriptor["bit_width"] = width;
  descriptor["integer_signedness"] = py::none();
  descriptor["float_format"] = format.str();
  return descriptor;
}

static void appendAbiArgument(py::list &arguments, unsigned abiIndex,
                              unsigned logicalIndex, const char *kind,
                              std::optional<py::dict> scalar, const std::string &mlirType,
                              uint64_t &offset, unsigned storageSize,
                              const char *field = nullptr) {
  offset = (offset + 3) & ~uint64_t(3);
  py::dict argument;
  argument["index"] = abiIndex;
  argument["logical_index"] = logicalIndex;
  argument["kind"] = kind;
  argument["scalar"] = scalar.has_value() ? py::object(*scalar) : py::none();
  argument["mlir_type"] = mlirType;
  argument["offset"] = offset;
  argument["storage_size"] = storageSize;
  argument["alignment"] = 4;
  if (field != nullptr)
    argument["field"] = field;
  else
    argument["field"] = py::none();
  arguments.append(argument);
  offset += storageSize;
}

static bool appendDynamicGmMemrefFields(py::list &arguments, unsigned &abiIndex,
                                        unsigned logicalIndex, MemRefType memrefType,
                                        uint64_t &offset) {
  // Schema v4: unified 13-slot descriptor for all dynamic GM ranks.
  // Device signature is unified dynamic GM memref + originShape0/1 index args.
  if (memrefType.getRank() != 4)
    return false;
  std::string mlirType = printType(memrefType);
  static constexpr const char *kFields[] = {
      "allocated", "aligned", "offset", "size0",        "size1",
      "size2",     "size3",   "stride0", "stride1",     "stride2",
      "stride3",   "originShape0", "originShape1"};
  for (const char *field : kFields) {
    appendAbiArgument(arguments, abiIndex++, logicalIndex, "memref_field", std::nullopt,
                      mlirType, offset, /*storageSize=*/8, field);
  }
  return true;
}

std::optional<py::dict> buildKernelAbi(ModuleOp module,
                                       const KernelPointerProvenance &provenance) {
  SmallVector<py::dict, 2> layouts;
  bool sawMixAic = false;
  bool sawMixAiv = false;
  std::optional<std::string> mixAicBase;
  std::optional<std::string> mixAivBase;
  bool anyMemrefField = false;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isDeclaration() || !function->hasAttr("hacc.entry"))
      continue;
    StringRef name = function.getSymName();
    StringRef logicalName = name;
    if (name.ends_with("_mix_aic")) {
      sawMixAic = true;
      mixAicBase = name.drop_back(8).str();
    }
    if (name.ends_with("_mix_aiv")) {
      sawMixAiv = true;
      mixAivBase = name.drop_back(8).str();
    }
    if (name.ends_with("_mix_aic") || name.ends_with("_mix_aiv"))
      logicalName = name.drop_back(8);
    auto provenanceIt = provenance.find(logicalName);
    ArrayRef<int8_t> pointerArgs;
    if (provenanceIt != provenance.end())
      pointerArgs = provenanceIt->second;
    if (function.getNumResults() != 0)
      return std::nullopt;

    py::list arguments;
    uint64_t offset = 0;
    bool supported = true;
    unsigned logicalIndex = 0;
    unsigned abiIndex = 0;
    unsigned skipOriginIndexArgs = 0;
    ArrayRef<Type> argTypes = function.getArgumentTypes();
    for (auto [index, type] : llvm::enumerate(argTypes)) {
      if (function.getArgAttr(index, "tla.debug_print.workspace") ||
          function.getArgAttr(index, "tla.print_tensor.workspace"))
        continue;
      // originShape0/1 are already folded into the 13-slot memref_field list.
      if (skipOriginIndexArgs > 0) {
        if (!type.isIndex()) {
          supported = false;
          break;
        }
        --skipOriginIndexArgs;
        continue;
      }
      unsigned provenanceIndex = logicalIndex;
      if (provenanceIndex < pointerArgs.size() &&
          pointerArgs[provenanceIndex] < 0) {
        supported = false;
        break;
      }
      if (auto memrefType = dyn_cast<MemRefType>(type);
          memrefType && ::tla::isGmMemRef(memrefType) &&
          !memrefType.hasStaticShape()) {
        if (!appendDynamicGmMemrefFields(arguments, abiIndex, logicalIndex, memrefType,
                                         offset)) {
          supported = false;
          break;
        }
        // Expect the next two function args to be originShape index companions.
        if (index + 2 >= argTypes.size() || !argTypes[index + 1].isIndex() ||
            !argTypes[index + 2].isIndex()) {
          supported = false;
          break;
        }
        skipOriginIndexArgs = 2;
        anyMemrefField = true;
        ++logicalIndex;
        continue;
      }
      bool pointer = isa<MemRefType, LLVM::LLVMPointerType>(type) ||
                     (provenanceIndex < pointerArgs.size() &&
                      pointerArgs[provenanceIndex] > 0);
      unsigned size = 0;
      std::optional<py::dict> scalar;
      if (pointer) {
        size = 8;
      } else if (auto scalarSize = scalarStorageSize(type)) {
        size = *scalarSize;
        scalar = scalarAbiDescriptor(type);
      } else {
        supported = false;
        break;
      }
      if (!pointer && !scalar) {
        supported = false;
        break;
      }
      appendAbiArgument(arguments, abiIndex++, logicalIndex++,
                        pointer ? "pointer" : "scalar", scalar, printType(type), offset,
                        size);
    }
    if (!supported || skipOriginIndexArgs != 0)
      return std::nullopt;
    offset = (offset + 7) & ~uint64_t(7);
    py::dict layout;
    // schema v4 when any dynamic GM memref_field is present; otherwise keep v3
    // for backward-compatible static pointer/scalar layouts.
    layout["schema_version"] = anyMemrefField ? 4 : 3;
    layout["entrypoint"] = logicalName.str();
    layout["total_size"] = offset;
    layout["arguments"] = arguments;
    layouts.push_back(layout);
  }
  if (layouts.empty())
    return std::nullopt;
  if (layouts.size() > 1) {
    if (layouts.size() != 2 || !sawMixAic || !sawMixAiv ||
        mixAicBase != mixAivBase)
      throw std::runtime_error(
          "Mixed kernel ABI collection expected exactly one AIC/AIV split "
          "entrypoint pair.");
    std::string expected = py::str(layouts.front()["arguments"]);
    for (const py::dict &layout : llvm::drop_begin(layouts))
      if (std::string(py::str(layout["arguments"])) != expected)
        throw std::runtime_error(
            "Mixed AIC/AIV kernel ABI mismatch: split entrypoints have "
            "different logical argument layouts.");
  }
  return layouts.front();
}

py::dict lowerToMlir(MlirModule cModule, std::vector<std::string> printBefore,
                     std::vector<std::string> printAfter, bool printBeforeAll, bool printAfterAll) {
  ModuleOp module = moduleFromCapsule(cModule);
  ::tla::registerTlaPasses();
  MLIRContext *context = module.getContext();
  context->allowUnregisteredDialects(true);
  context->disableMultithreading();
  ::tla::tools::loadTlaCompileDialects(*context);

  PassManager tlaPm(context);
  PassManager llvmPm(context);
  ::tla::tools::buildTlaCompilePassManagers(*context, tlaPm, llvmPm);
  KernelPointerProvenance pointerProvenance =
      collectKernelPointerProvenance(module);

  std::string passDump;
  llvm::raw_string_ostream passDumpStream(passDump);
  if (printBeforeAll || printAfterAll || !printBefore.empty() || !printAfter.empty()) {
    auto shouldPrintBefore = [printBeforeAll, printBefore](Pass *pass, Operation *) {
      return printBeforeAll || containsPassName(printBefore, pass);
    };
    auto shouldPrintAfter = [printAfterAll, printAfter](Pass *pass, Operation *) {
      return printAfterAll || containsPassName(printAfter, pass);
    };
    tlaPm.enableIRPrinting(shouldPrintBefore, shouldPrintAfter,
                           /*printModuleScope=*/true,
                           /*printAfterOnlyOnChange=*/false,
                           /*printAfterOnlyOnFailure=*/false, passDumpStream);
    llvmPm.enableIRPrinting(shouldPrintBefore, shouldPrintAfter,
                            /*printModuleScope=*/true,
                            /*printAfterOnlyOnChange=*/false,
                            /*printAfterOnlyOnFailure=*/false, passDumpStream);
  }

  std::string output;
  std::string error;
  bool success = ::tla::tools::runTlaCompilePipelinesWithManagers(
      module, StringRef("mlir"), tlaPm, llvmPm, output, error,
      /*rewriteTileSignaturesToLLVMPointer=*/true);
  passDumpStream.flush();
  py::dict result;
  result["success"] = success;
  result["error"] = success ? "" : (error.empty() ? "Failed to run Tla pipeline." : error);
  result["lowered_mlir"] = output;
  result["pass_ir_dump"] = passDump;
  if (auto kernelAbi = buildKernelAbi(module, pointerProvenance))
    result["kernel_abi"] = *kernelAbi;
  else
    result["kernel_abi"] = py::none();
  return result;
}

} // namespace

PYBIND11_MODULE(_tla_type_bridge_native, m) {
  m.doc() = "Native Tla TypeDef construction/accessors for Python MLIR types.";

  m.def("load_tla_dialect", &loadTlaDialect, py::arg("context"));

  m.def("shape_type_get", &shapeTypeGet, py::arg("context"), py::arg("tree"));
  m.def("coord_type_get", &coordTypeGet, py::arg("context"), py::arg("tree"));
  m.def("stride_type_get", &strideTypeGet, py::arg("context"), py::arg("tree"));

  m.def("layout_type_get", &layoutTypeGet, py::arg("context"), py::arg("shape_tree"),
        py::arg("stride_tree"), py::arg("origin_tree") = py::none(),
        py::arg("layout") = "row_major");
  m.def("layout_type_from_components_get", &layoutTypeFromComponentsGet, py::arg("context"),
        py::arg("shape_type"), py::arg("stride_type"), py::arg("origin_shape_type") = py::none(),
        py::arg("layout") = "row_major");
  m.def("tensor_type_get", &tensorTypeGet, py::arg("context"), py::arg("shape_tree"),
        py::arg("stride_tree"), py::arg("coord_tree"), py::arg("origin_shape_tree"),
        py::arg("element_type"), py::arg("addrspace"), py::arg("layout"), py::arg("ptr_alignment"));
  m.def("dynamic_gm_memref_type", &dynamicGmMemrefTypeGet, py::arg("tensor_type"),
        "Build the unified dynamic GM memref type for schema-v4 ABI from a !tla.tensor.");
  m.def("ptr_type_get", &ptrTypeGet, py::arg("context"), py::arg("pointee"), py::arg("addrspace"),
        py::arg("alignment"));
  m.def("vector_ssa_type_get", &vectorSSATypeGet, py::arg("context"),
        py::arg("valid_lanes"), py::arg("element_type"));
  m.def("mask_ssa_type_get", &maskSSATypeGet, py::arg("context"),
        py::arg("physical_lanes"));
  m.def("flag_type_get", &flagTypeGet, py::arg("context"));
  m.def("cross_flag_type_get", &crossFlagTypeGet, py::arg("context"), py::arg("mode"));
  m.def("cross_flag_mode", &crossFlagMode, py::arg("type"));
  m.def("mutex_type_get", &mutexTypeGet, py::arg("context"));
  m.def("copy_l0c2dst_params_type_get", &copyL0C2DstParamsTypeGet, py::arg("context"));

  m.def("type_is_ptr", &typeIs<::tla::PtrType>, py::arg("type"));
  m.def("type_is_tensor", &typeIs<::tla::TlaTensorType>, py::arg("type"));
  m.def("type_is_shape", &typeIs<::tla::ShapeType>, py::arg("type"));
  m.def("type_is_coord", &typeIs<::tla::CoordType>, py::arg("type"));
  m.def("type_is_stride", &typeIs<::tla::StrideType>, py::arg("type"));
  m.def("type_is_layout", &typeIs<::tla::LayoutType>, py::arg("type"));
  m.def("type_is_vector_ssa", &typeIs<::tla::VectorSSAType>, py::arg("type"));
  m.def("type_is_mask_ssa", &typeIs<::tla::MaskSSAType>, py::arg("type"));
  m.def("type_is_flag", &typeIs<::tla::FlagType>, py::arg("type"));
  m.def("type_is_cross_flag", &typeIs<::tla::CrossFlagType>, py::arg("type"));
  m.def("type_is_mutex", &typeIs<::tla::MutexType>, py::arg("type"));
  m.def("type_is_copy_l0c2dst_params", &typeIs<::tla::CopyL0C2DstParamsType>, py::arg("type"));
  m.def("tla_type_category", &tlaTypeCategory, py::arg("type"));

  m.def("ptr_pointee_type_get", &ptrPointeeTypeGet, py::arg("ptr_type"));
  m.def("ptr_addrspace", &ptrAddrspace, py::arg("ptr_type"));
  m.def("ptr_alignment", &ptrAlignment, py::arg("ptr_type"));
  m.def("tensor_ptr_type_get", &tensorPtrTypeGet, py::arg("tensor_type"));
  m.def("vector_ssa_element_type_get", &vectorSSAElementTypeGet,
        py::arg("vector_type"));
  m.def("vector_ssa_valid_lanes_get", &vectorSSAValidLanesGet,
        py::arg("vector_type"));
  m.def("mask_ssa_physical_lanes_get", &maskSSAPhysicalLanesGet, py::arg("mask_type"));
  m.def("lower_to_mlir", &lowerToMlir, py::arg("module"), py::arg("mlir_print_ir_before"),
        py::arg("mlir_print_ir_after"), py::arg("mlir_print_ir_before_all"),
        py::arg("mlir_print_ir_after_all"),
        "Lower an mlir.ir.Module through the typed MLIR Python bridge.");
}
