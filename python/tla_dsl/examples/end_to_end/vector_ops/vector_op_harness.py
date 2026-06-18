from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

import catlass as tla

DEFAULT_SHAPES = (31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 500, 1024, 2000)
DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"


@dataclass(frozen=True)
class DTypeSpec:
    name: str
    tla_dtype: Any
    torch_dtype_name: str
    default_sentinel: float | int
    element_bytes: int
    register_elements: int | None = None


REG_BYTES = 256

DTYPES: dict[str, DTypeSpec] = {
    "f32": DTypeSpec("f32", tla.Float32, "float32", -7.0, 4),
    "f16": DTypeSpec("f16", tla.Float16, "float16", -7.0, 2),
    "bf16": DTypeSpec("bf16", tla.BFloat16, "bfloat16", -7.0, 2),
    "i8": DTypeSpec("i8", tla.Int8, "int8", -101, 1),
    "i16": DTypeSpec("i16", tla.Int16, "int16", -7, 2),
    "i32": DTypeSpec("i32", tla.Int32, "int32", -7, 4),
}


@dataclass(frozen=True)
class VectorKernelConfig:
    vector_elements: int
    lanes: int
    loops: int
    tla_dtype: Any
    torch_dtype: Any
    default_sentinel: float | int
    element_bytes: int


@dataclass(frozen=True)
class DirectVectorOpConfig:
    description: str
    kernel: Any
    all_dtypes: tuple[str, ...]
    operator_specs: Callable[[], dict[str, dict[str, Any]]]
    set_kernel_config: Callable[
        [str, str, tuple[int, ...] | None], tuple[type[Any], Any, float | int]
    ]
    compile_only_type_args: Callable[[str, str, tuple[int, ...] | None], tuple[Any, ...]]
    get_vector_elements: Callable[[], int]
    get_kernel_shape: Callable[[], tuple[int, ...]]
    make_inputs: Callable[[argparse.Namespace, str, Any], tuple[Any, ...]]
    expected: Callable[[str, tuple[Any, ...]], Any]
    unsupported_case: Callable[[str, str], bool]
    print_skip: Callable[[str, str, tuple[int, ...]], None]
    script_path: Path
    env_compile_jobs: str
    float_dtypes: frozenset[str]
    output_count: int = 1


def shape_num_elements(shape: tuple[int, ...]) -> int:
    elements = 1
    for dim in shape:
        elements *= dim
    return elements


def shape_label(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def dtype_config(dtype_name: str, all_dtypes: tuple[str, ...]) -> DTypeSpec:
    if dtype_name not in all_dtypes or dtype_name not in DTYPES:
        raise SystemExit(
            f"unsupported dtype={dtype_name!r}; expected one of: {', '.join(all_dtypes)}"
        )
    return DTYPES[dtype_name]


def torch_dtype(spec: DTypeSpec, torch: Any | None = None) -> Any:
    if torch is None:
        try:
            import torch as torch_mod
        except ImportError:
            return None
        torch = torch_mod
    return getattr(torch, spec.torch_dtype_name)


def vector_kernel_config(
    dtype_name: str, shape: tuple[int, ...] | None, all_dtypes: tuple[str, ...]
) -> VectorKernelConfig:
    spec = dtype_config(dtype_name, all_dtypes)
    vector_elements = shape_num_elements(shape) if shape is not None else 400
    lanes = spec.register_elements or (REG_BYTES // spec.element_bytes)
    return VectorKernelConfig(
        vector_elements=vector_elements,
        lanes=lanes,
        loops=(vector_elements + lanes - 1) // lanes,
        tla_dtype=spec.tla_dtype,
        torch_dtype=torch_dtype(spec),
        default_sentinel=spec.default_sentinel,
        element_bytes=spec.element_bytes,
    )


def make_type_args(
    tla_dtype: Any, kernel_shape: tuple[int, ...], tensor_count: int
) -> tuple[Any, ...]:
    from catlass import runtime as runtime_mod

    with runtime_mod._eager_capture():
        tensor_shape = tla.make_shape(*kernel_shape)
        tensor_coord = tla.make_coord(*(0 for _ in kernel_shape))
        tensor_stride = tla.make_stride(1)
        return tuple(
            tla.Tensor(
                tensor_shape,
                tla_dtype,
                origin_shape=tensor_shape,
                coord=tensor_coord,
                stride=tensor_stride,
                layout_tag=tla.arch.RowMajor,
            )
            for _ in range(tensor_count)
        )


def parse_shape(value: str, *, script_name: str) -> tuple[int, ...]:
    text = value.strip().lower().replace(",", "x")
    if not text:
        raise argparse.ArgumentTypeError("shape must not be empty")
    try:
        dims = tuple(int(part) for part in text.split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid shape {value!r}; expected N or AxB"
        ) from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(f"shape dimensions must be positive: {value!r}")
    if len(dims) != 1:
        raise argparse.ArgumentTypeError(
            f"{script_name} currently supports flat 1D vector shapes only"
        )
    return dims


def _parse_compile_jobs(value: str) -> int | str:
    if value == "all":
        return value
    try:
        jobs = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "compile jobs must be a positive integer, 0 for CPU count, or 'all'"
        ) from exc
    if jobs < 0:
        raise argparse.ArgumentTypeError(
            "compile jobs must be a positive integer, 0 for CPU count, or 'all'"
        )
    return jobs


def _available_cpu_cores() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    cpu_count = os.cpu_count() or 0
    return list(range(cpu_count))


def _resolve_compile_jobs(args: argparse.Namespace, case_count: int) -> int:
    requested = args.compile_jobs
    if requested == "all":
        return case_count
    jobs = (
        int(requested)
        if requested
        else len(_available_cpu_cores()) or os.cpu_count() or 1
    )
    return max(1, min(jobs, case_count))


def runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aiv.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }


def make_tla_tensor(
    dev_buf: Any, tla_dtype: type[Any], kernel_shape: tuple[int, ...]
) -> Any:
    from catlass import runtime as runtime_mod

    contiguous = dev_buf.contiguous()
    with runtime_mod._eager_capture():
        tensor_shape = tla.make_shape(*kernel_shape)
        tensor = tla.Tensor(
            tensor_shape,
            tla_dtype,
            origin_shape=tensor_shape,
            coord=tla.make_coord(*(0 for _ in kernel_shape)),
            stride=tla.make_stride(1),
            data_ptr=int(contiguous.data_ptr()),
        )
    tensor._external_binding = True
    return tensor


def require_torch_npu(device_id: int, script_name: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"{script_name} --run requires PyTorch.") from exc
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"{script_name} --run requires torch_npu.") from exc
    torch.npu.set_device(device_id)
    return torch


class DirectVectorOpHarness:
    def __init__(self, config: DirectVectorOpConfig) -> None:
        self.config = config
        self.script_name = config.script_path.stem

    def _runtime_kwargs(self, args: argparse.Namespace) -> dict[str, Any]:
        return runtime_kwargs(args)

    def dump_tlair(self, args: argparse.Namespace) -> str:
        if self.config.unsupported_case(args.op, args.dtype):
            self.config.print_skip(args.op, args.dtype, args.shape)
            return ""
        return self.config.kernel.dump_mlir(
            type_args=self.config.compile_only_type_args(args.op, args.dtype, args.shape)
        )

    def build_only(self, args: argparse.Namespace) -> int:
        if self.config.unsupported_case(args.op, args.dtype):
            self.config.print_skip(args.op, args.dtype, args.shape)
            return 0
        artifact = tla.compile(
            self.config.kernel,
            *self.config.compile_only_type_args(args.op, args.dtype, args.shape),
            **self._runtime_kwargs(args),
        )
        print("compile_ok=True")
        print(f"kernel.o path={artifact.kernel_binary_path}")
        return 0

    def _case_cache_dir(
        self, args: argparse.Namespace, op_name: str, dtype_name: str, shape: tuple[int, ...]
    ) -> Path:
        return Path(args.cache_dir).expanduser() / op_name / f"{dtype_name}_{shape_label(shape)}"

    def _case_args(
        self, args: argparse.Namespace, dtype_name: str, shape: tuple[int, ...]
    ) -> argparse.Namespace:
        case_args = argparse.Namespace(**vars(args))
        case_args.dtype = dtype_name
        case_args.shape = shape
        case_args.cache_dir = str(self._case_cache_dir(args, args.op, dtype_name, shape))
        return case_args

    def _precompile_case_command(
        self, args: argparse.Namespace, dtype_name: str, shape: tuple[int, ...], index: int
    ) -> list[str]:
        command = [
            sys.executable,
            str(self.config.script_path),
            args.op,
            "--build-only",
            "--dtype",
            dtype_name,
            "--shape",
            shape_label(shape),
            "--cache-dir",
            str(self._case_cache_dir(args, args.op, dtype_name, shape)),
            "--force-recompile",
        ]
        taskset = shutil.which("taskset")
        cores = _available_cpu_cores()
        if taskset is None or not cores:
            return command
        return [taskset, "-c", str(cores[index % len(cores)]), *command]

    @staticmethod
    def _print_process_output(process: subprocess.CompletedProcess[str]) -> None:
        if process.stdout:
            print(process.stdout, end="" if process.stdout.endswith("\n") else "\n")
        if process.stderr:
            print(process.stderr, end="" if process.stderr.endswith("\n") else "\n")

    def precompile_sweep(self, args: argparse.Namespace) -> int:
        cases = []
        skipped = 0
        for dtype_name in args.dtypes:
            for shape in args.shapes:
                if self.config.unsupported_case(args.op, dtype_name):
                    skipped += 1
                    print(
                        f"===== PRECOMPILE SKIP {args.op} dtype={dtype_name} "
                        f"shape={shape_label(shape)} ====="
                    )
                    self.config.print_skip(args.op, dtype_name, shape)
                    continue
                cases.append((dtype_name, shape))
        if not cases:
            print(
                f"{args.op} precompile summary: "
                f"passed=0 failed=0 skipped={skipped} total={skipped}"
            )
            return 0
        jobs = _resolve_compile_jobs(args, len(cases))
        print("precompile_sweep_enabled=True")
        print(f"precompile_sweep_jobs={jobs}")

        def run_case(
            index: int, dtype_name: str, shape: tuple[int, ...]
        ) -> tuple[str, tuple[int, ...], subprocess.CompletedProcess[str]]:
            process = subprocess.run(
                self._precompile_case_command(args, dtype_name, shape, index),
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            return dtype_name, shape, process

        failed = 0
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(run_case, index, dtype_name, shape)
                for index, (dtype_name, shape) in enumerate(cases)
            ]
            for future in concurrent.futures.as_completed(futures):
                dtype_name, shape, process = future.result()
                label = f"{args.op} dtype={dtype_name} shape={shape_label(shape)}"
                if process.returncode == 0:
                    print(f"===== PRECOMPILE PASS {label} =====")
                    continue
                failed += 1
                print(f"===== PRECOMPILE FAIL {label} rc={process.returncode} =====")
                self._print_process_output(process)
                if args.fail_fast:
                    break

        print(f"timing.sweep_precompile_seconds={time.perf_counter() - start:.6f}")
        print(
            f"{args.op} precompile summary: "
            f"passed={len(cases) - failed} failed={failed} "
            f"skipped={skipped} total={len(cases) + skipped}"
        )
        return 0 if failed == 0 else 1

    def _run_single_case(
        self, args: argparse.Namespace, dtype_name: str, torch: Any
    ) -> int:
        if self.config.unsupported_case(args.op, dtype_name):
            self.config.print_skip(args.op, dtype_name, args.shape)
            return 0
        tla_dtype, torch_dt, default_sentinel = self.config.set_kernel_config(
            args.op, dtype_name, args.shape
        )
        inputs = self.config.make_inputs(args, dtype_name, torch)
        device = "npu"
        sentinel = args.sentinel if args.sentinel is not None else default_sentinel
        outputs = tuple(
            torch.full(
                (self.config.get_vector_elements(),),
                sentinel,
                dtype=torch_dt,
                device=device,
            )
            for _ in range(self.config.output_count)
        )
        expected = self.config.expected(args.op, inputs)
        expected_outputs = expected if isinstance(expected, tuple) else (expected,)
        if len(expected_outputs) != self.config.output_count:
            raise SystemExit(
                f"expected() returned {len(expected_outputs)} outputs, "
                f"but output_count={self.config.output_count}"
            )

        tla_inputs = tuple(
            make_tla_tensor(input_tensor, tla_dtype, self.config.get_kernel_shape())
            for input_tensor in inputs
        )
        tla_outputs = tuple(
            make_tla_tensor(output, tla_dtype, self.config.get_kernel_shape())
            for output in outputs
        )

        artifact = tla.compile(
            self.config.kernel,
            *tla_inputs,
            *tla_outputs,
            **self._runtime_kwargs(args),
        )
        artifact(*tla_inputs, *tla_outputs, block=args.block)

        torch.npu.synchronize()
        first_mismatch: dict[str, Any] | None = None
        output_matches = []
        for output_index, (actual, expected_output) in enumerate(
            zip(outputs, expected_outputs, strict=True)
        ):
            if dtype_name in self.config.float_dtypes:
                expected_match = torch.isclose(
                    actual, expected_output, rtol=0.0, atol=args.atol
                )
            else:
                expected_match = actual.eq(expected_output)
            output_matches.append(bool(expected_match.all()))
            mismatch = expected_match.logical_not().nonzero(as_tuple=False)
            if first_mismatch is None and mismatch.numel():
                index = int(mismatch[0].item())
                first_mismatch = {
                    "output": output_index,
                    "index": index,
                    "actual": actual[index].item(),
                    "expected": expected_output[index].item(),
                }

        print(
            f"compile_ok=True host=torch_npu op={args.op} dtype={dtype_name} "
            f"shape={shape_label(args.shape)} layout=row"
        )
        print(f"kernel.o path={artifact.kernel_binary_path}")
        print("launch_ok=True")
        print(f"outputs equal expected {args.op}? {all(output_matches)}")
        print(f"first mismatch={first_mismatch}")
        return 0 if first_mismatch is None else 1

    def sweep(self, args: argparse.Namespace) -> int:
        if args.precompile_sweep:
            precompile_rc = self.precompile_sweep(args)
            if precompile_rc != 0:
                return precompile_rc
            args = argparse.Namespace(**vars(args))
            args.no_cache = False
            args.force_recompile = False

        tla.initialize(device=args.device)
        try:
            torch = require_torch_npu(args.device, self.script_name)
            total = 0
            passed = 0
            failed = 0
            skipped = 0
            start = time.perf_counter()
            for dtype_name in args.dtypes:
                for shape in args.shapes:
                    total += 1
                    case_args = self._case_args(args, dtype_name, shape)
                    if self.config.unsupported_case(args.op, dtype_name):
                        skipped += 1
                        print(
                            f"===== SKIP {args.op} dtype={dtype_name} "
                            f"shape={shape_label(shape)} ====="
                        )
                        self.config.print_skip(args.op, dtype_name, shape)
                        continue
                    print(
                        f"===== START {args.op} dtype={dtype_name} "
                        f"shape={shape_label(shape)} ====="
                    )
                    rc = self._run_single_case(case_args, dtype_name, torch)
                    if rc == 0:
                        passed += 1
                        print(
                            f"===== PASS {args.op} dtype={dtype_name} "
                            f"shape={shape_label(shape)} ====="
                        )
                    else:
                        failed += 1
                        print(
                            f"===== FAIL {args.op} dtype={dtype_name} "
                            f"shape={shape_label(shape)} rc={rc} ====="
                        )
                        if args.fail_fast:
                            break
                if failed and args.fail_fast:
                    break
            print(
                f"{args.op} summary: passed={passed} failed={failed} "
                f"skipped={skipped} total={total}"
            )
            print(f"timing.sweep_total_seconds={time.perf_counter() - start:.6f}")
            return 0 if failed == 0 else 1
        finally:
            tla.finalize()

    def run(self, args: argparse.Namespace) -> int:
        tla.initialize(device=args.device)
        try:
            torch = require_torch_npu(args.device, self.script_name)
            failed = 0
            dtypes = self.config.all_dtypes if args.all_dtypes else (args.dtype,)
            for dtype_name in dtypes:
                print("---", f"op={args.op}", f"dtype={dtype_name}", "---")
                failed += self._run_single_case(args, dtype_name, torch)
            return 0 if failed == 0 else 1
        finally:
            tla.finalize()

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self.config.description)
        parser.add_argument("op", choices=tuple(sorted(self.config.operator_specs())))
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument("--build-only", action="store_true")
        mode.add_argument("--run", action="store_true")
        mode.add_argument("--sweep", action="store_true")
        parser.add_argument("--device", type=int, default=2)
        parser.add_argument("--block", type=int, default=1)
        parser.add_argument("--dtype", choices=self.config.all_dtypes, default="f32")
        parser.add_argument(
            "--shape",
            type=lambda value: parse_shape(value, script_name=self.script_name),
            default=(self.config.get_vector_elements(),),
        )
        parser.add_argument(
            "--shapes",
            type=lambda value: parse_shape(value, script_name=self.script_name),
            nargs="+",
            help="Flat vector shapes for --sweep.",
        )
        parser.add_argument(
            "--sizes",
            type=int,
            nargs="+",
            help="Alias for --shapes with one-dimensional vector sizes.",
        )
        parser.add_argument(
            "--dtypes",
            choices=self.config.all_dtypes,
            nargs="+",
            default=self.config.all_dtypes,
            help="Operand dtypes for --sweep.",
        )
        parser.add_argument("--all-dtypes", action="store_true")
        parser.add_argument("--sentinel", type=float, default=None)
        parser.add_argument("--atol", type=float, default=None)
        parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
        parser.add_argument("--force-recompile", action="store_true")
        parser.add_argument("--no-cache", action="store_true")
        parser.add_argument("--dump-tlair", action="store_true")
        parser.add_argument("--fail-fast", action="store_true")
        parser.add_argument(
            "--precompile-sweep",
            action="store_true",
            default=True,
            help="For --sweep, compile all cases in parallel subprocesses before running.",
        )
        parser.add_argument(
            "--no-precompile-sweep",
            action="store_false",
            dest="precompile_sweep",
        )
        parser.add_argument(
            "--compile-jobs",
            type=_parse_compile_jobs,
            default=_parse_compile_jobs(os.environ.get(self.config.env_compile_jobs, "all")),
        )
        return parser

    def main(self) -> int:
        args = self._build_parser().parse_args()
        if args.atol is None:
            args.atol = self.config.operator_specs()[args.op]["default_atol"]
        if args.sizes is not None:
            args.shapes = tuple((size,) for size in args.sizes)
        elif args.shapes is None:
            args.shapes = tuple((size,) for size in DEFAULT_SHAPES)
        if args.dump_tlair:
            if args.all_dtypes:
                raise SystemExit("--dump-tlair requires a single dtype.")
            dumped = self.dump_tlair(args)
            if dumped:
                print(dumped)
            return 0
        if args.sweep:
            return self.sweep(args)
        if args.build_only:
            if args.all_dtypes:
                raise SystemExit("--build-only requires a single dtype.")
            return self.build_only(args)
        return self.run(args)
