# AGENTS.md — DSL (python/tla_dsl)

## Test Guidelines

### Execution

- pytest: `pytest python/tla_dsl/tests/`
- lit: `ninja check-tla-lit` (in build directory)

### Layout

Rule: assertion contains `hivm.*` → lit; otherwise → pytest.

**pytest** (`tests/`):

```
tests/
├── conftest.py / _bootstrap.py        # shared infra only
├── api/                               # op API preconditions, generated op surface
├── frontend/                          # language layer: control flow / decorators / AST / constexpr
├── types/                             # numeric types, tensor types, dlpack
├── ops/
│   ├── vector/{simd,simt}/
│   ├── scalar/
│   ├── tensor/
│   ├── load-store/
│   ├── sync/
│   ├── extern/
│   └── print/
├── compiler/                          # compile/execute chain, kernel ABI
├── runtime/                           # JIT, execution args, stream, launch ABI
├── arch/                              # arch metadata, capacity queries
└── infra/                             # build scripts, API doc generation, pretest bootstrap
```

**lit** (`lit/`):

```
lit/
├── sync/{event,cross-core,mutex}/
├── copy/
├── mmad/
├── vector/{simd,simt}/
├── control-flow/
├── scalar/
├── tensor/
└── print/
```

### Naming

- **pytest** = snake_case: `tests/<area>[/<sub>]/test_<feature>.py`
- **lit** = kebab-case: positive → `topic-case.mlir`; errors → `topic-diagnostics.test`
  (multi-case → `BEGIN/END` blocks)
