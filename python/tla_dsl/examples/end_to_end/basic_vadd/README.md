# Basic VADD End-to-End Example

This directory contains a small vector-add kernel and a host harness that mirrors
the `basic_mmad` example.

Files:

- `basic_vadd.py`: kernel, compile-only path, torch_npu run path, and TLAIR dump path.
- `dump_ir.py`: writes raw TLA MLIR, lowered MLIR, and pass trace artifacts.
- `dump_mlir.sh`: convenience wrapper for refreshing MLIR artifacts.

From `python/tla_dsl`:

```bash
python examples/end_to_end/basic_vadd/basic_vadd.py --dump-tlair
python examples/end_to_end/basic_vadd/basic_vadd.py --build-only
python examples/end_to_end/basic_vadd/basic_vadd.py --run --device 2 --force-recompile
```

The runtime path uses `torch` and `torch_npu` tensors on NPU, wraps them through
`catlass.runtime.from_dlpack`, compiles with `arch_scope="aiv.c310"`, launches the
kernel, and compares `z` with `x + y`.
