# 模板库快速上手

在学习各类模板库开发知识之前，请您先参考下面步骤完成基础环境搭建和源码下载，确保已安装NPU驱动和CANN软件（`Ascend-cann-toolkit`和`Ascend-cann-ops`）等，并尝试进行最基础的模板库示例编译和运行。

## 安装CANN

请参考《[CANN快速安装](https://www.hiascend.com/cann/download)》，选择`稳定版本`（仅支持CANN 8.5.0及后续版本），并根据产品系列、CPU架构、操作系统等下载对应包，最后参考页面提供的命令完成安装。

## 安装其他基础依赖

请根据您的实际操作系统环境，参考[README-软硬件配套说明](../../../README.md#-软硬件配套说明)逐个安装基础依赖。

### 环境验证

安装完CANN包后，需验证环境和驱动是否正常。

- **检查NPU设备**

    ```bash
    # 运行npu-smi，若能正常显示设备信息，则驱动正常
    npu-smi info
    ```

- **检查CANN版本**

    ```bash
    # 查看CANN toolkit包版本信息（默认路径安装）
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info

    # 查看CANN ops包版本信息（默认路径安装）
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_ops_install.info
    ```

    其中`${arch}`可通过`uname -m`查询当前架构，如aarch64、x86_64。
- **检查其他基础依赖**

    ```bash
    gcc --version
    cmake --version
    ```

### 环境变量配置

按需选择合适的命令使环境变量生效。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

### 下载源码

将CATLASS代码仓下载到本地。

```bash
# 下载项目源码，以master分支为例
git clone https://gitcode.com/cann/catlass.git
```

## 编译执行

> 模板库提供了一套可复用的模板、基础组件，用于矩阵乘法算子开发，算子样例可见[样例目录](../../../examples)。

- 样例名称里带`ascend950`字样的仅支持Ascend 950PR&950DT 系列产品。
- 其他样例仅支持Atlas A2/A3 系列产品，代码中架构标签`ArchTag`统一使用`Arch::AtlasA2`。

### 样例编译

进入项目根目录，执行下述编译指令：

```bash
bash scripts/build.sh [options] <target>
```

- `options`：可选编译选项，当前支持的选项包括：
  - `--clean`：清理此前编译及输出目录（默认路径分别为`/build`，`/output`）。
  - `--debug`：以Debug模式进行编译。
  - `--msdebug`：启用[msDebug](https://www.hiascend.com/document/redirect/CannCommunityToolMsdebug)工具，详见[在CATLASS样例工程使用msDebug](./evaluation/msdebug.md)。
  - `--simulator`：启用仿真器模式，该选项启用后将不在实际NPU上执行，详见[CATLASS样例仿真](./evaluation/performance_tools.md#msprof-op-simulator使用示例--msprof-op-simulator-example-)。
  - `--enable_profiling`：启用Profiling工具，详见[CATLASS样例性能调优](./evaluation/performance_tools.md#profiling简介--profiling-introduction-)。
  - `--enable_print`：启用编译器的打印功能，详见[基于`cce::printf`进行设备侧打印](./evaluation/print.md)。
  - `--enable_ascendc_dump`：启用`AscendC`相关算子调测API，详见[CATLASS样例使用AscendC算子调测API](./evaluation/ascendc_dump.md)。
  - `-DCATLASS_ARCH`：指明NPU架构，当前支持`2201`（Atlas A2/A3 系列产品）和`3510`（Ascend 950PR&950DT 系列产品），不指明则默认为`2201`。
  - `-D<option>`：给CMake传递其他的编译选项。

- `target`：要编译的算子样例，可指定为特定的样例名，也可指定为：
  - `catlass_examples`：对当前仓样例进行全量编译。
  - `python_extension`：编译pybind扩展，详见基于[Python调用CATLASS样例](../../../examples/python_extension/README.md)。
  - `torch_library`：编译torch扩展，详见[Python调用CATLASS样例](../../../examples/python_extension/README.md)。
  - `mstuner_catlass`：编译msTuner_CATLASS工具，详见[`mstuner_catlass`使用说明](../../../tools/tuner/README.md)。

以[00_basic_matmul](../../../examples/00_basic_matmul/README.md)样例编译过程为例，执行下述指令（若为其他支持Ascend 950PR&950DT 系列产品的样例，需要添加`-DCATLASS_ARCH=3510`。）：

```bash
# 编译算子组件
bash scripts/build.sh 00_basic_matmul
```

若有下述提示信息，则编译成功。

```text
"[INFO] Target "{target}" built successfully."
```

### 算子执行

算子编译产物在`output/bin`路径下，切换至该目录下可运行算子样例程序。
以[00_basic_matmul](../../../examples/00_basic_matmul/README.md)样例为例，可通过下述指令执行该算子：

```bash
# 切换至编译产物目录
cd output/bin
# ./00_basic_matmul m n k [deviceId]
./00_basic_matmul 256 512 1024 0
```

- `256`，`512`，`1024`分别为矩阵乘法在m轴、n轴、k轴的维度（左/右矩阵数据随机生成）。
- `deviceId`可选（默认为0），指定NPU卡的ID号。

执行该算子样例后，如出现下述结果则表明其计算符合精度预期（该样例中Matmul的左、右矩阵使用随机数填充，真值以CPU计算为准。）

```text
Compare success.
```

请进一步参考[创新样例开发流程指南](./10_innovative_example_development_guide.md)开始第一个创新算子的完整开发，或参考[Host侧代码组装指南](./02_host_example_assembly.md)开始Host侧算子模板组装体验。
