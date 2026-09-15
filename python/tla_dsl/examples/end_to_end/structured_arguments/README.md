# 结构化 kernel 参数

这些示例展示 Python 结构化参数如何进入设备 kernel，并检查实际计算结果。
需要已构建的 TLA DSL 和支持 `--npu-arch 3510` 的 NPU 环境。

| 示例 | 内容 |
|---|---|
| `mixed_memref_arguments.py` | 普通 Tensor 与 Dynamic-GM 混合、嵌套参数、多描述符及替换输入 |
| `scalar_subclass_arguments.py` | 标量子类、枚举、bool、Int64 通过普通、嵌套和 dataclass 参数传递 |
| `static_tensor_interfaces.py` | Tensor 在容器中保留设备侧接口，顶层 Constexpr 在启动时省略 |

在当前目录运行，例如：

```bash
python mixed_memref_arguments.py --device 0
```

设备编号由 `--device` 指定，默认 0。示例包含重复启动和输入更新检查。
这些示例也由仓库现有 DSL battery 执行；在仓库根目录可仅运行此组：

```bash
python -m pytest -q tests/dsl_battery --run-battery --device 0 -k structured-args
```
