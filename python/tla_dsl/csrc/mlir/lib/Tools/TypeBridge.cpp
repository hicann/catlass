#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaDialect.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "Passes.h"
#include "Tools/CompilePipeline.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "pybind11/stl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

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

py::dict lowerToMlir(MlirModule cModule, std::vector<std::string> printBefore,
                     std::vector<std::string> printAfter, bool printBeforeAll, bool printAfterAll) {
  ModuleOp module = moduleFromCapsule(cModule);
  tla::registerTlaPasses();
  MLIRContext *context = module.getContext();
  context->allowUnregisteredDialects(true);
  context->disableMultithreading();
  tla::tools::loadTlaCompileDialects(*context);

  PassManager tlaPm(context);
  PassManager llvmPm(context);
  tla::tools::buildTlaCompilePassManagers(*context, tlaPm, llvmPm);

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
  if (!tla::tools::runTlaCompilePipelinesWithManagers(
          module, StringRef("mlir"), tlaPm, llvmPm, output, error,
          /*rewriteTileSignaturesToLLVMPointer=*/true)) {
    throw std::runtime_error(error.empty() ? "Failed to run Tla pipeline." : error);
  }
  passDumpStream.flush();
  py::dict result;
  result["lowered_mlir"] = output;
  result["pass_ir_dump"] = passDump;
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
