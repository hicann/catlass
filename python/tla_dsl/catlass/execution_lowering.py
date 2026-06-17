"""Execution-mode lowering that emits Tla MLIR directly while running Python frontend code."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mlir import ir as mlir_ir  # type: ignore[assignment]

from . import _tla_type_bridge
from . import runtime as runtime_mod
from . import tla_ast_decorators as ast_decorators
from .base_dsl.ast_preprocessor import maybe_transform_for_lowering
from .base_dsl import DSLLocation
from .base_dsl.typing import Constexpr, _elem_token_to_mlir_type
from .mixed_kernel_attrs import (
    MixedKernelModuleAttrInputs,
    build_mixed_kernel_module_attrs,
)
from .tla.typing import Tensor


class TlaLoweringError(RuntimeError):
    """Raised when Tla DSL lowering fails."""


class UnsupportedExecutionLowering(RuntimeError):
    """Raised when execution-mode lowering cannot safely handle a function."""


@dataclass
class LoweredTlaIR:
    """Structured result of execution-mode lowering to TLA MLIR (``tla`` dialect)."""

    context: mlir_ir.Context
    module: mlir_ir.Module
    generic: bool = False
    _asm: str | None = None

    def asm(self, *, generic: bool | None = None) -> str:
        emit_generic = self.generic if generic is None else bool(generic)
        if self._asm is None or emit_generic != self.generic:
            with self.context:
                self._asm = self.module.operation.get_asm(
                    print_generic_op_form=emit_generic,
                    assume_verified=False,
                )
            self.generic = emit_generic
        return self._asm


_DEFAULT_MIXED_KERNEL_TARGET_SYSTEM_SPEC = (
    '#dlti.target_system_spec<"NPU" : #hacc.target_device_spec<'
    '#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, '
    '#dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, '
    '#dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, '
    '#dlti.dl_entry<"UB_SIZE", 2031616 : i32>, '
    '#dlti.dl_entry<"L1_SIZE", 4194304 : i32>, '
    '#dlti.dl_entry<"L0A_SIZE", 524288 : i32>, '
    '#dlti.dl_entry<"L0B_SIZE", 524288 : i32>, '
    '#dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, '
    '#dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, '
    '#dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, '
    '#dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, '
    '#dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, '
    '#dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, '
    '#dlti.dl_entry<"ARCH", "dav-c310">>>'
)


def lower_jit_to_tlair_by_execution(
    fn: Any,
    *,
    kind: str,
    options: Mapping[str, Any] | None = None,
    generic: bool = False,
    type_args: Sequence[Any] | None = None,
    location: DSLLocation | None = None,
) -> str:
    return lower_jit_to_tlair_module_by_execution(
        fn,
        kind=kind,
        options=options,
        generic=generic,
        type_args=type_args,
        location=location,
    ).asm(generic=generic)


def lower_jit_to_tlair_module_by_execution(
    fn: Any,
    *,
    kind: str,
    options: Mapping[str, Any] | None = None,
    generic: bool = False,
    type_args: Sequence[Any] | None = None,
    location: DSLLocation | None = None,
) -> LoweredTlaIR:
    del kind, options
    fn = maybe_transform_for_lowering(
        fn,
        internal_for=ast_decorators._internal_frontend_for,
        internal_region=runtime_mod._internal_frontend_region,
        internal_if=ast_decorators._internal_frontend_if,
        internal_if_expr=ast_decorators._internal_frontend_if_expr,
        internal_bool_and=ast_decorators._internal_frontend_bool_and,
        internal_bool_or=ast_decorators._internal_frontend_bool_or,
        internal_bool_not=ast_decorators._internal_frontend_bool_not,
        internal_compare=ast_decorators._internal_frontend_compare,
        internal_any=ast_decorators._internal_frontend_any,
        internal_all=ast_decorators._internal_frontend_all,
        internal_bool=ast_decorators._internal_frontend_bool,
        internal_min=ast_decorators._internal_frontend_min,
        internal_max=ast_decorators._internal_frontend_max,
    )
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    arg_names = [p.name for p in params]
    constexpr_names = {p.name for p in params if _is_constexpr_annotation(p.annotation)}
    call_args = _prepare_call_args(arg_names=arg_names, type_args=type_args)

    ctx = mlir_ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        _load_execution_dialects(ctx)
        arg_types = _resolve_execution_arg_types(
            fn=fn,
            arg_names=arg_names,
            arg_values=dict(zip(arg_names, call_args, strict=False))
            if call_args
            else None,
            ctx=ctx,
        )
        with mlir_ir.Location.unknown(ctx):
            module = mlir_ir.Module.create()
            with mlir_ir.InsertionPoint(module.body):
                fn_loc = _coerce_location(ctx, location)
                _build_tla_func(
                    fn=fn,
                    module=module,
                    fn_name=fn.__name__,
                    arg_names=arg_names,
                    constexpr_names=constexpr_names,
                    arg_types=arg_types,
                    call_args=call_args,
                    ctx=ctx,
                    fn_loc=fn_loc,
                )
                _set_module_exec_units_attr(module)
    lowered = LoweredTlaIR(context=ctx, module=module, generic=bool(generic))
    lowered._asm = module.operation.get_asm(
        print_generic_op_form=bool(generic),
        assume_verified=False,
    )
    return lowered


def _prepare_call_args(
    *, arg_names: Sequence[str], type_args: Sequence[Any] | None
) -> tuple[Any, ...]:
    if type_args is None:
        if arg_names:
            raise UnsupportedExecutionLowering(
                "Execution-mode lowering requires type_args for non-empty signatures."
            )
        return ()
    if len(type_args) != len(arg_names):
        raise TlaLoweringError(
            "type_args length must match function arguments: "
            f"expected {len(arg_names)}, got {len(type_args)}"
        )
    return tuple(type_args)


def _build_tla_func(
    *,
    fn: Any,
    module: mlir_ir.Module,
    fn_name: str,
    arg_names: Sequence[str],
    constexpr_names: set[str],
    arg_types: Mapping[str, Any],
    call_args: Sequence[Any],
    ctx: mlir_ir.Context,
    fn_loc: mlir_ir.Location,
) -> None:
    runtime_arg_names = [name for name in arg_names if name not in constexpr_names]
    mlir_arg_types = [
        _coerce_type(ctx, arg_types.get(name)) for name in runtime_arg_names
    ]
    fn_type = mlir_ir.FunctionType.get(mlir_arg_types, [])
    func_op = mlir_ir.Operation.create(
        "tla.func",
        attributes={
            "sym_name": mlir_ir.StringAttr.get(fn_name),
            "function_type": mlir_ir.TypeAttr.get(fn_type),
        },
        regions=1,
        loc=fn_loc,
    )
    entry = func_op.regions[0].blocks.append(*mlir_arg_types)

    # Use distinct proxy objects for runtime parameters so names like `dim` resolve to block
    # SSA values, not type_args constants. Then make_shape(dim, 16) gets a dynamic `?` dim
    # while make_shape(4, 8, 16) keeps static dimensions from literals.
    class _ArgProxy:
        __slots__ = ()

    proxies = [_ArgProxy() for _ in runtime_arg_names]
    call_args_for_fn = list(call_args)
    idx = 0
    for i, name in enumerate(arg_names):
        if name in runtime_arg_names:
            call_args_for_fn[i] = proxies[idx]
            idx += 1
    call_args_for_fn = tuple(call_args_for_fn)
    arg_bindings = {
        id(proxy): value for proxy, value in zip(proxies, entry.arguments, strict=False)
    }
    category_bindings: dict[int, str] = {}
    for name, proxy, value in zip(
        runtime_arg_names, proxies, entry.arguments, strict=False
    ):
        category = _category_from_type_like(ctx, arg_types.get(name))
        if category is None:
            continue
        category_bindings[id(proxy)] = category
        category_bindings[id(value)] = category

    tensor_host_by_value: dict[Any, Any] = {}
    for pos, name in enumerate(arg_names):
        if name in constexpr_names:
            continue
        v = call_args[pos]
        if isinstance(v, Tensor):
            j = runtime_arg_names.index(name)
            tensor_host_by_value[entry.arguments[j]] = v

    with mlir_ir.InsertionPoint(entry):
        with runtime_mod._frontend_emission(
            arg_bindings=arg_bindings,
            category_bindings=category_bindings,
            tensor_host_by_value=tensor_host_by_value,
            module=module,
        ) as emit_state:
            try:
                fn(*call_args_for_fn)
            except runtime_mod.TlaCoreAPIError:
                raise
            except TlaLoweringError:
                raise
            except Exception as exc:
                raise UnsupportedExecutionLowering(
                    f"Execution-mode lowering failed while running `{fn.__name__}`: {exc}"
                ) from exc
        _set_module_has_vector_region_attr(module, emit_state.has_vector_region)
        _set_func_exec_units_attr(func_op, emit_state.exec_units)
        mlir_ir.Operation.create("tla.return", loc=fn_loc)


def _set_func_exec_units_attr(func_op: mlir_ir.Operation, exec_units: set[str]) -> None:
    attr_value = runtime_mod._frontend_exec_units_attr_value(exec_units)
    if attr_value is None or attr_value == "vector":
        return
    func_op.attributes["tla.exec_units"] = mlir_ir.StringAttr.get(attr_value)
    if attr_value == "cube_vector":
        func_op.attributes["tla.mixed_split"] = mlir_ir.UnitAttr.get(func_op.context)


def _set_module_has_vector_region_attr(
    module: mlir_ir.Module, has_vector_region: bool
) -> None:
    if not has_vector_region:
        return
    module.operation.attributes["tla.has_vector_region"] = mlir_ir.UnitAttr.get(
        module.context
    )


def _set_module_exec_units_attr(module: mlir_ir.Module) -> None:
    exec_units: set[str] = set()
    for op in module.body.operations:
        try:
            attr = op.attributes["tla.exec_units"]
        except KeyError:
            continue
        value = str(mlir_ir.StringAttr(attr).value)
        if value in {"cube", "cube_vector"}:
            exec_units.add("cube")
        if value in {"vector", "cube_vector"}:
            exec_units.add("vector")
    attr_value = runtime_mod._frontend_exec_units_attr_value(exec_units)
    if attr_value is None or attr_value == "vector":
        return
    if attr_value == "cube_vector":
        _validate_mixed_kernel_target_contract()
    module.operation.attributes["tla.module_exec_units"] = mlir_ir.StringAttr.get(
        attr_value
    )


def _get_default_mixed_kernel_module_attr_inputs() -> MixedKernelModuleAttrInputs:
    return MixedKernelModuleAttrInputs(
        target_name="Ascend950PR_9589",
        module_core_type="MIX",
        target_system_spec=_DEFAULT_MIXED_KERNEL_TARGET_SYSTEM_SPEC,
    )


def _validate_mixed_kernel_target_contract() -> None:
    build_mixed_kernel_module_attrs(_get_default_mixed_kernel_module_attr_inputs())


def _coerce_location(
    ctx: mlir_ir.Context, location: DSLLocation | None
) -> mlir_ir.Location:
    if location is None:
        return mlir_ir.Location.unknown(ctx)
    if location.lineno <= 0:
        return mlir_ir.Location.unknown(ctx)
    file_loc = mlir_ir.Location.file(
        location.filename,
        int(location.lineno),
        int(location.col_offset),
        ctx,
    )
    return mlir_ir.Location.name(location.function_name, childLoc=file_loc, context=ctx)


def _coerce_type(ctx: mlir_ir.Context, type_like: Any) -> mlir_ir.Type:
    from . import _tla_type_bridge

    if isinstance(type_like, mlir_ir.Type):
        return type_like
    if type_like is None:
        return _tla_type_bridge.value_type_get(ctx)
    if isinstance(type_like, str):
        with ctx:
            return _elem_token_to_mlir_type(type_like)
    raise TypeError(
        "execution lowering expected mlir.ir.Type, Tla element token, or None; "
        f"got {type(type_like).__name__}"
    )


def _category_from_type_like(ctx: mlir_ir.Context, type_like: Any) -> str | None:
    from . import _tla_type_bridge

    if type_like is None:
        return "value"
    try:
        ty = _coerce_type(ctx, type_like)
    except Exception:
        return None
    if isinstance(ty, mlir_ir.IndexType):
        return "index"
    return _tla_type_bridge.tla_type_category(ty)


def _resolve_execution_arg_types(
    *,
    fn: Any,
    arg_names: Sequence[str],
    arg_values: Mapping[str, Any] | None,
    ctx: mlir_ir.Context,
) -> Mapping[str, Any]:
    resolved: dict[str, Any] = {}
    if arg_values is not None:
        for name, value in arg_values.items():
            mlir_types_getter = getattr(value, "__get_mlir_types__", None)
            if callable(mlir_types_getter):
                resolved_types = mlir_types_getter(ctx)
                if resolved_types:
                    resolved[name] = resolved_types[0]
                    continue
            if isinstance(value, bool):
                resolved[name] = "i1"
            elif isinstance(value, int):
                resolved[name] = "index"
            elif isinstance(value, float):
                resolved[name] = "f32"
    return resolved


def _load_execution_dialects(ctx: mlir_ir.Context) -> None:
    _tla_type_bridge.load_tla_dialect(ctx)
    for dialect in ("arith", "scf"):
        ctx.dialects[dialect]


def _is_constexpr_annotation(annotation: Any) -> bool:
    if annotation is inspect._empty:
        return False
    if Constexpr.is_constexpr_annotation(annotation):
        return True
    if isinstance(annotation, str):
        compact = annotation.replace(" ", "")
        return (
            compact == "Constexpr"
            or compact.startswith("Constexpr[")
            or compact == "tla.Constexpr"
            or compact.startswith("tla.Constexpr[")
        )
    return False


__all__ = [
    "TlaLoweringError",
    "LoweredTlaIR",
    "UnsupportedExecutionLowering",
    "lower_jit_to_tlair_by_execution",
    "lower_jit_to_tlair_module_by_execution",
]
