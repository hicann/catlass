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
├── frontend/                          # frontend-layer tests
│   └── ops/                           # Tla.td op tests; subdirs mirror lit topics
│       ├── vector/{simd,simt}/
│       ├── scalar/
│       ├── tensor/
│       ├── load-store/
│       ├── sync/
│       └── print/
├── fixtures/                          # shared test data
└── test_*.py                          # flat cases; further topical grouping in follow-up PRs
```

**lit** (`lit/`, op-lowering topics directly under lit/, no extra grouping layer):

```
lit/
├── sync/                              # sync ops; cross-core/ + mutex/ subtopics
├── copy/
├── mmad/
├── vector/{simd,simt}/
├── scalar/
├── tensor/
├── control-flow/
├── print/
├── extern/
├── arch/                              # block/ub capacity limits
└── kernel/                            # kernel-level structure & ABI, regression template
```

### Naming

- **pytest** = snake_case: `tests/<area>[/<sub>]/test_<feature>.py`
- **lit** = kebab-case: positive → `topic-case.mlir`; errors → `topic-diagnostics.test`
  (multi-case → `BEGIN/END` blocks)
