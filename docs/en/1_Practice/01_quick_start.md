# Template Library Quick Start

Before learning about template library development, complete the basic environment setup and source code download by referring to the following steps. Ensure that the NPU driver and CANN software (`Ascend-cann-toolkit` and `Ascend-cann-ops`) are installed, and then try to build and run a basic template library sample.

## Installing CANN

See [CANN Quick Installation](https://www.hiascend.com/cann/download), select the stable version (only CANN 8.5.0 and later are supported), download the corresponding package based on your product series, CPU architecture, and OS, and complete the installation by following the commands provided on the page.

## Installing Other Basic Dependencies

Install the basic dependencies one by one by referring to [Required Software and Hardware](../../../README_en.md#-software-and-hardware-requirements) according to your actual OS environment.

### Environment Verification

After installing the CANN package, verify that the environment and driver work properly.

- **Check the NPU device**

    ```bash
    # Run npu-smi. If the device information is displayed properly, the driver is working.
    npu-smi info
    ```

- **Check the CANN version**

    ```bash
    # View the CANN toolkit package version (default installation path).
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info

    # View the CANN ops package version (default installation path).
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_ops_install.info
    ```

    `${arch}` can be queried using `uname -m`, for example, aarch64 or x86_64.
- **Check other basic dependencies**

    ```bash
    gcc --version
    cmake --version
    ```

### Environment Variable Configuration

Run the appropriate command to make the environment variables take effect.

```bash
# Default installation path (using the root user as an example; for a non-root user, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Custom installation path
# source ${install_path}/cann/set_env.sh
```

### Downloading the Source Code

Download the CATLASS code repository to the local PC.

```bash
# Download the project source code, using the master branch as an example.
git clone https://gitcode.com/cann/catlass.git
```

## Build and Execution

> The template library provides a set of reusable templates and basic components for matrix multiplication operator development. Operator samples can be found [here](../../../examples).

- Samples whose names contain `ascend950` are supported only on Ascend 950PR&950DT series products.
- Other samples are supported only on Atlas A2/A3 series products, and the architecture tag `ArchTag` in the code is uniformly `Arch::AtlasA2`.

### Building the Sample

Go to the root directory of the project and run the following build command:

```bash
bash scripts/build.sh [options] <target>
```

- `options`: Optional build options. Currently supported options include:
  - `--clean`: Clears the previous build and output directories (the default paths are `/build` and `/output`, respectively).
  - `--debug`: Compiles in debug mode.
  - `--msdebug`: Enables [msDebug](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0062.html). For details, see [Using msDebug in a CATLASS Sample Project](./evaluation/msdebug.md).
  - `--simulator`: Enables the simulator mode. After this option is enabled, the code will not run on an actual NPU. For details, see [CATLASS Sample Simulation](./evaluation/performance_tools.md#example-with-msprof-op-simulator).
  - `--enable_profiling`: Enables the profiling tool. For details, see [Tuning Performance in a CATLASS Sample Project](./evaluation/performance_tools.md#profiling-overview).
  - `--enable_print`: Enables the compiler's print functionality. For details, see [Device-Side Printing Based on `cce::printf`](./evaluation/print.md).
  - `--enable_ascendc_dump`: Enables `Ascend C` operator debugging APIs. For details, see [Using Ascend C Operator Debugging APIs in a CATLASS Sample Project](./evaluation/ascendc_dump.md).
  - `-DCATLASS_ARCH`: Specifies the NPU architecture. Currently, `2201` (Atlas A2/A3 series products) and `3510` (Ascend 950PR&950DT series products) are supported. If not specified, the default is `2201`.
  - `-D<option>`: Passes other build options to CMake.

- `target`: Specifies the operator sample to build. You can specify a specific sample name or one of the following:
  - `catlass_examples`: Builds all samples in the repository.
  - `python_extension`: Builds the Pybind extension. For details, see [Python-based CATLASS Sample Calls](../../../examples/python_extension/README.md).
  - `torch_library`: Builds the Torch extension. For details, see [Python-based CATLASS Sample Calls](../../../examples/python_extension/README.md).
  - `mstuner_catlass`: Builds msTuner_CATLASS. For details, see [`mstuner_catlass` Usage Guide](../../../tools/tuner/README.md).

Take the [basic_matmul](../../../examples/00_basic_matmul/README.md) sample as an example. Run the following command (for other samples that support the Ascend 950PR&950DT series products, add `-DCATLASS_ARCH=3510`):

```bash
# Build the operator component.
bash scripts/build.sh 00_basic_matmul
```

If the following prompt appears, the build succeeded.

```text
"[INFO] Target "{target}" built successfully."
```

### Executing Operators

The operator build output is located in the `output/bin` directory. Switch to this directory to run the operator sample program.
Take the [basic_matmul] sample as an example. You can run the operator by executing the following commands:

```bash
# Switch to the build output directory.
cd output/bin
# ./00_basic_matmul m n k [deviceId]
./00_basic_matmul 256 512 1024 0
```

- `256`, `512`, and `1024` are the dimensions of the matrix multiplication along the M-axis, N-axis, and K-axis, respectively (left/right matrix data is randomly generated).
- `deviceId` (optional, default value: 0) specifies the NPU ID.

After executing the operator sample, the following result indicates that the computation meets the precision expectations (in this sample, the left and right matrices of the matmul are filled with random numbers, and the ground truth is based on CPU computation).

```text
Compare success.
```

To start complete development of your first innovative operator, see the [Innovative Example Development Guide](./10_innovative_example_development_guide.md). To try host-side operator template assembly, see the [Host-Side Code Assembly Guide](./02_host_example_assembly.md).
