# BigMatmul Example Readme

## Code Organization

```
├── 39_big_matmul_tla
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── big_matmul_tla.cpp # Main file
```

## Remarks

This test case utilizes the L2 tiling and staggered core allocation scheduler. In large-case scenarios, this scheduler can improve the L2 cache hit ratio and reduce inter-core address conflicts.

## Example

- After obtaining the code, compile the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```
# Compiling a specified case
bash scripts/build.sh 39_big_matmul_tla
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./39_big_matmul_tla 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```
Compare success.
```
