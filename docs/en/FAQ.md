# FAQ

## How do I specify the target hardware architecture for CATLASS?

At the first community meeting in March 2026, we officially confirmed that the CATLASS community mainline would add support for the next-generation Ascend hardware, the Ascend 950PR&950DT series products. Official support starts with CATLASS 1.5.0. To distinguish the underlying interface implementations across platforms, this support introduces a compilation macro that must be set in the corresponding build commands.

- New macro: `CATLASS_ARCH`, which specifies the target architecture. Its value is listed in the `__NPU_ARCH__` column of [SIMD BuiltIn Keywords](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/en/guide/programming_guide/language_extension/SIMD-BuiltIn_keyword.md).
  - `Atlas A2/A3 series products`: `2201`
  - `Ascend 950PR&950DT series products`: `3510`

- Usage by scenario:
  - `bisheng` command line: `bisheng ... -DCATLASS_ARCH=2201 ...`
  - `cmake`: `add_compile_definitions(CATLASS_ARCH=2201)`
  - `msopgen/aclnn` projects:
    - Old syntax (CANN < 9.0.0): `add_ops_compile_options(ALL OPTIONS -DCATLASS_ARCH=2201 ...)`
    - New syntax (CANN >= 9.0.0): `npu_op_kernel_options(ascendc_kernels ALL OPTIONS -DCATLASS_ARCH=2201)` (in an msopgen project, the first parameter defaults to `ascendc_kernels` and can be adjusted as needed)
  - CATLASS source repository: `bash scripts/build.sh -DCATLASS_ARCH=2201 ...`
  - Code reference: [examples/CMakeLists.txt](https://gitcode.com/cann/catlass/blob/master/examples/CMakeLists.txt)
