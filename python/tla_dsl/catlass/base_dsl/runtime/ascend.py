"""Ascend NPU runtime helpers for kernel load and launch (PyACL / AscendCL).

Module-level ``load_binary`` / ``launch_kernel``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from ...execution import TlaRuntimeUnavailableError

# acl_rt.h — PyACL does not export these macros; pass numeric literals.
# ACL_ERROR_INVALID_PARAM. The runtime returns it for a launch whose parameters
# it will not accept -- among them a dynamic UB size larger than the binary's own
# accounting allows, which is the one case a caller can act on.
ACL_ERROR_INVALID_PARAM = 107000

_ACL_RT_BINARY_LOAD_OPT_MAGIC = 2
_ACL_RT_BINARY_MAGIC_ELF_AICORE = 0x43554245
_ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE = 0x41415246


@dataclass
class _LoadedKernel:
    binary_handle: int
    function_handle: int
    kernel_binary_path: Path


def load_acl() -> ModuleType:
    """Import the Ascend ``acl`` Python package (early availability check)."""
    try:
        import acl
    except ImportError as exc:
        raise TlaRuntimeUnavailableError(
            "Failed to import `acl`. Ensure the Ascend Python runtime is installed."
        ) from exc
    return acl


def check_acl_errors(
    ret: object,
    op_name: str,
    *,
    error_cls: type[Exception] | None = None,
) -> None:
    """Raise if an ACL / AscendCL call did not return 0."""
    if int(ret) == 0:
        return
    cls = error_cls if error_cls is not None else TlaRuntimeUnavailableError
    error = cls(f"{op_name} failed with ret={int(ret)}")
    # The numeric code, so a caller that handles one specific failure can test
    # for it rather than grepping the message.
    error.ret = int(ret)  # type: ignore[attr-defined]
    raise error


def _acl_status(result: object) -> int:
    """Normalize PyACL return value to a status code (``ret`` or ``(value, ret)``)."""
    if isinstance(result, tuple):
        return int(result[-1])
    return int(result)


def _binary_load_options_for_mode(kernel_mode: str) -> list[dict[str, int]]:
    """Build PyACL ``binary_load_from_data`` options for ``kernel_mode``.

    Must use ``binary_load_from_data`` (not FromFile): FromFile rejects
    ``ACL_RT_BINARY_LOAD_OPT_MAGIC`` with ``ACL_ERROR_INVALID_PARAM``. See
    ``scripts/aclrt_poc/mre_binary_load_magic.cpp``.

    - aiv: VECTOR_CORE magic → AIV slot
    - aic: AICORE magic → AIC slot
    - mix: empty options; fat mix ELF already exposes both aic+aiv addrs
    """
    mode = (kernel_mode or "").strip().lower()
    if mode == "aiv":
        return [
            {
                "type": _ACL_RT_BINARY_LOAD_OPT_MAGIC,
                "value": _ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE,
            }
        ]
    if mode == "aic":
        return [
            {
                "type": _ACL_RT_BINARY_LOAD_OPT_MAGIC,
                "value": _ACL_RT_BINARY_MAGIC_ELF_AICORE,
            }
        ]
    if mode == "mix":
        return []
    return [
        {
            "type": _ACL_RT_BINARY_LOAD_OPT_MAGIC,
            "value": _ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE,
        }
    ]


def _register_kernel_binary(
    *, kernel_binary_path: Path, entrypoint: str, kernel_mode: str
) -> _LoadedKernel:
    """Load ``kernel.o`` via ``acl.rt.binary_load_from_data`` + mode magic."""
    import acl

    raw = kernel_binary_path.read_bytes()
    host_buf = (ctypes.c_char * len(raw)).from_buffer_copy(raw)
    opts = _binary_load_options_for_mode(kernel_mode)
    binary_handle, load_ret = acl.rt.binary_load_from_data(
        ctypes.addressof(host_buf), len(raw), opts
    )
    check_acl_errors(load_ret, "acl.rt.binary_load_from_data")
    function_handle, get_ret = acl.rt.binary_get_function(
        int(binary_handle), entrypoint
    )
    if int(get_ret) != 0:
        try:
            acl.rt.binary_unload(int(binary_handle))
        except Exception:
            pass
        check_acl_errors(get_ret, "acl.rt.binary_get_function")
    return _LoadedKernel(
        binary_handle=int(binary_handle),
        function_handle=int(function_handle),
        kernel_binary_path=kernel_binary_path,
    )


def load_binary(
    *, entrypoint: str, kernel_mode: str, kernel_binary_path: Path
) -> tuple[int, int]:
    """Load ``kernel.o`` in the current context and return its handles."""
    resolved = kernel_binary_path.resolve()
    loaded = _register_kernel_binary(
        kernel_binary_path=resolved,
        entrypoint=entrypoint,
        kernel_mode=kernel_mode,
    )
    return loaded.binary_handle, loaded.function_handle


# ``ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE`` from ``acl_rt.h``. PyACL marshals
# ``cfg`` as a list of ``{"id", "value"}`` dicts, so both go in as plain ints.
_ACL_LAUNCH_ATTR_DYN_UBUF_SIZE = 2


def _launch_config(dyn_ubuf_bytes: int) -> list[dict[str, int]]:
    """Ask the launch for ``dyn_ubuf_bytes`` of UB beyond what the binary declares.

    The size of the launch-sized region, not where it ends: the runtime adds it
    to the static footprint recorded in the binary. An empty list leaves the
    launch untouched, which is what a kernel whose UB the compiler placed in
    full wants -- and on SIMT it also leaves the Data Cache its full extent,
    since the cache takes whatever UB goes unclaimed.
    """
    if dyn_ubuf_bytes <= 0:
        return []
    return [{"id": _ACL_LAUNCH_ATTR_DYN_UBUF_SIZE, "value": int(dyn_ubuf_bytes)}]


def launch_kernel(
    *,
    function_handle: int,
    stream: int,
    block_num: int,
    payload: bytes,
    dyn_ubuf_bytes: int = 0,
    uses_scalar_print: bool = False,
    uses_tensor_print: bool = False,
    is_mixed: bool = False,
    print_tensor_position: str | None = None,
    device_id: int | None = None,
) -> None:
    """Launch via PyACL ``kernel_args_*`` + ``launch_kernel_with_config`` (Host args)."""
    import acl

    block_num = int(block_num)
    cfg = _launch_config(dyn_ubuf_bytes)

    def _acl_launch(launch_payload: bytes) -> None:
        launch_payload = launch_payload or (b"\x00" * ctypes.sizeof(ctypes.c_uint64))
        host_buf = (ctypes.c_uint8 * len(launch_payload)).from_buffer_copy(
            launch_payload
        )
        args_handle, init_ret = acl.rt.kernel_args_init(int(function_handle))
        check_acl_errors(init_ret, "acl.rt.kernel_args_init")
        check_acl_errors(
            _acl_status(
                acl.rt.kernel_args_append(
                    args_handle, ctypes.addressof(host_buf), len(launch_payload)
                )
            ),
            "acl.rt.kernel_args_append",
        )
        check_acl_errors(
            _acl_status(acl.rt.kernel_args_finalize(args_handle)),
            "acl.rt.kernel_args_finalize",
        )
        # PyACL: cfg must be a list (``None`` fails parse); reserve as ``0``.
        check_acl_errors(
            _acl_status(
                acl.rt.launch_kernel_with_config(
                    int(function_handle),
                    int(block_num),
                    int(stream),
                    cfg,
                    args_handle,
                    0,
                )
            ),
            "acl.rt.launch_kernel_with_config",
        )

    uses_print_fifo = uses_scalar_print or uses_tensor_print
    if uses_print_fifo:
        from .ascend_debug_fifo import launch_with_debug_fifo

        launch_with_debug_fifo(
            launch_kernel=_acl_launch,
            payload=payload,
            block_num=block_num,
            stream=int(stream),
            uses_scalar_print=uses_scalar_print,
            uses_tensor_print=uses_tensor_print,
            is_mixed=is_mixed,
            print_tensor_position=print_tensor_position,
            device=int(device_id or 0),
        )
        return

    _acl_launch(payload)


__all__ = [
    "load_acl",
    "check_acl_errors",
    "load_binary",
    "launch_kernel",
]
