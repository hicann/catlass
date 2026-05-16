"""Allocator helpers extracted from the Tla core API facade."""

from __future__ import annotations

from typing import Any

from mlir import ir as mlir_ir  # type: ignore[assignment]

from ..address_space import AddressSpace
from ..base_dsl.arch import get_localmem_capacity_bytes
from ..base_dsl.typing import Int8, Pointer
from ..execution_lowering import TlaLoweringError
from ..types import PtrType


def _core_api() -> Any:
    from .. import core_api as _tla_core_api

    return _tla_core_api


class LocalmemAllocator:
    """Allocator object compatible with the HQ-style frontend surface."""

    __slots__ = ()

    @staticmethod
    def capacity_in_bytes(mem_scope: AddressSpace | None = None) -> int:
        core_api = _core_api()
        if mem_scope is None:
            return get_localmem_capacity_bytes("cbuf")
        core_api._require_pointer_addrspace(
            "utils.LocalmemAllocator.capacity_in_bytes", mem_scope, 0
        )
        return get_localmem_capacity_bytes(str(core_api._token(mem_scope)))

    def allocate(
        self,
        size_bytes: Any,
        byte_alignment: int,
        mem_scope: AddressSpace,
    ) -> Pointer:
        """Emit static ``tla.alloc_ptr {size_bytes = N : i64} -> !tla.ptr<i8, …>`` (scratch ``alloc_ptr``-style).

        ``size_bytes`` must be a compile-time constant in ``[1, 2**63-1]`` (static only).
        Address space and byte alignment appear **only** on the result ``!tla.ptr`` type;
        ``mem_scope`` must be ``tla.AddressSpace`` (any member; unsupported scopes may fail at capacity / lowering).
        """
        core_api = _core_api()
        core_api._require_index(
            "utils.LocalmemAllocator.allocate", "size_bytes", size_bytes, 0
        )
        if (
            isinstance(byte_alignment, bool)
            or not isinstance(byte_alignment, int)
            or byte_alignment <= 0
        ):
            core_api._op_error(
                "utils.LocalmemAllocator.allocate",
                "invalid argument 'byte_alignment' (position 1): expected positive_int, "
                f"got {core_api._type_name(byte_alignment)}",
            )
        core_api._require_pointer_addrspace(
            "utils.LocalmemAllocator.allocate", mem_scope, 2
        )
        core_api._require_frontend_state("utils.LocalmemAllocator.allocate")
        mlir_loc = core_api._capture_user_loc()
        scope_key = str(mem_scope)

        size_const = core_api._const_int_value(size_bytes)
        if size_const is None:
            raise TlaLoweringError(
                "LocalmemAllocator.allocate requires a static size_bytes "
                "(compile-time constant); dynamic sizes are not supported."
            )
        n = int(size_const)
        if n <= 0 or n > 9_223_372_036_854_775_807:
            raise TlaLoweringError(
                "LocalmemAllocator.allocate size_bytes must be in [1, 2**63-1] "
                f"for tla.alloc_ptr {{size_bytes : i64}}; got {n}"
            )

        ctx = mlir_loc.context if mlir_loc is not None else mlir_ir.Context()
        int8_ty = Int8.mlir_type(ctx)
        ptr_ty = PtrType.get(int8_ty, scope_key, byte_alignment, context=ctx)
        i64_ty = mlir_ir.IntegerType.get_signless(64, context=ctx)
        op = mlir_ir.Operation.create(
            "tla.alloc_ptr",
            operands=[],
            results=[ptr_ty],
            attributes={
                "size_bytes": mlir_ir.IntegerAttr.get(i64_ty, n),
            },
            loc=mlir_loc,
        )

        return core_api._Pointer(op.results[0], alloc_size_bytes=n)


__all__ = ["LocalmemAllocator"]
