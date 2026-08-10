# CATLASS DSL 环境变量一览

本文说明运行与编译 CATLASS DSL 时可设置的环境变量。  
未特别说明时，变量名均以 `CATLASS_DSL_` 开头。

## 开关类变量

部分变量是开关。下面这些写法都表示「开」（不区分大小写）：

`1`、`true`、`yes`、`on`

下面这些写法都表示「关」：

`0`、`false`、`no`、`off`

下文表格中的「默认值」即未设置该变量时采用的取值。

---

## 编译与运行

### `CATLASS_DSL_CACHE`

是否使用编译缓存（相同配置下不再重复编译）。

| 取值 | 含义 |
| --- | --- |
| 默认值：`1` | 开启缓存 |
| `0` | 每次重新编译，不复用已有结果 |

---

### `CATLASS_DSL_CACHE_DIR`

编译缓存存放的目录。

| 取值 | 含义 |
| --- | --- |
| 默认值：`$HOME/.cache/catlass` | 若已设置 `XDG_CACHE_HOME`，则为 `$XDG_CACHE_HOME/catlass` |
| 某个目录路径 | 缓存写到该目录 |

关闭缓存时，不会长期使用该目录。

---

### `CATLASS_DSL_FORCE_RECOMPILE`

是否强制重新编译（即使已有缓存）。

| 取值 | 含义 |
| --- | --- |
| 默认值：`0` | 不强制，可命中缓存 |
| `1` | 强制重新编译 |

---

### `CATLASS_DSL_PRINT_IR`

是否在编译过程中打印中间表示（IR），便于排查编译问题。

| 取值 | 含义 |
| --- | --- |
| 默认值：`0` | 不打印 |
| `1` | 打印编译过程中的中间 IR |

---

### `CATLASS_DSL_KEEP`

编译成功后，把哪些结果**额外拷贝**到便于查看的目录（见 `CATLASS_DSL_DUMP_DIR`）。  
可写多个，用英文逗号分隔，不区分大小写。不认识的名字会被忽略。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 不额外拷贝；结果仍在缓存目录里 |
| `ir` | 拷贝编译后的中间 IR 文件（`.mlir`） |
| `ir-debug` | 拷贝更早阶段、便于调试的 IR（`.tlair.mlir`）；若有过程 dump，也会一并拷贝 |
| `kernel` | 拷贝设备侧编译产物（`.o`） |
| `all` | 上面三种都要 |

示例：`ir,kernel` 或 `all`。

---

### `CATLASS_DSL_DUMP_DIR`

`KEEP` 导出文件的目标目录。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 开启 `KEEP` 时导出到当前工作目录 |
| 某个目录路径 | 导出到该目录（不存在会自动创建） |

说明：仅设置本变量、未开启 `KEEP` 时，不会自动导出 `ir` / `kernel` 等；若同时开启 `PRINT_IR`，过程中的 dump 文件仍会写到本目录。

---

### `CATLASS_DSL_NPU_DEVICE`

在未使用 `torch_npu` 当前设备时，指定在哪张 NPU 上跑。

| 取值 | 含义 |
| --- | --- |
| 默认值：`0` | 使用设备号 `0` |
| 非负整数（如 `0`、`1`） | 使用该设备号 |

若环境里已有可用的 `torch` / `torch_npu`，优先用 torch 当前设备，本变量不生效。

---

## 构建环境

### `CATLASS_DSL_PREBUILT_ASCENDNPU_IR`

本机已编译好的 AscendNPU-IR 工程根目录（用于构建 DSL、跑测试）。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | `tests/run_dsl_test.sh` 等脚本会依次回退到 `CATLASS_DSL_ASCENDNPU_IR_ROOT`、工作区旁路 `AscendNPU-IR`、仓库内 `python/tla_dsl/3rdparty/AscendNPU-IR` |
| 某个目录路径 | 指向该 IR 工程根目录（需已完成构建） |

请指向工程根目录，而不是其中的安装子目录。

---

### `MLIR_TBLGEN_INCLUDE_DIR`

生成/编译工具链所需的 MLIR 头文件目录。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 由构建脚本根据 `CATLASS_DSL_PREBUILT_ASCENDNPU_IR` 自动设置 |
| 某个目录路径 | 手动指定该目录 |

普通业务开发通常不必改。

---

### `CATLASS_DSL_CATLASS_INCLUDE_DIR`

查找 Catlass 头文件时使用的目录（构建相关）。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 由 CMake 根据 `CATLASS_INCLUDE_DIR` 或仓库布局自动推导 |
| 某个目录路径 | 手动指定头文件所在目录 |

普通业务开发通常不必改。

---

## 其它常用环境变量

### `ASCEND_HOME_PATH`

昇腾 CANN 工具包安装路径（由 `set_env.sh` 设置）。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 依赖系统 `PATH` 查找编译器等工具 |
| 某个目录路径 | 在该安装树下查找编译工具 |

---

### `XDG_CACHE_HOME`

在未指定 `CATLASS_DSL_CACHE_DIR` 时，影响默认缓存目录。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 缓存目录为 `$HOME/.cache/catlass` |
| 某个目录路径 | 默认缓存为该路径下的 `catlass` 子目录 |

---

### `PYTHONPATH` / `LD_LIBRARY_PATH`

Python 模块搜索路径与动态库搜索路径。按系统惯例用 `:` 拼接多个目录即可。具体应包含哪些路径，见环境准备文档。
