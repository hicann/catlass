# 混合 GM 参数

本示例将 Dynamic-GM Tensor、普通静态 Tensor 和标量作为独立顶层参数传入 kernel，
不使用结构化参数。覆盖动态参数在前、静态参数在前、插入标量及多个动态参数，
分别使用 FP16/FP32 检查计算结果、重复启动、编译 key 稳定性和地址重绑定。
输入使用独立缓冲和非对称计算；另覆盖 rank-2 静态 Tensor 的非零索引、
多维 stride 和完整输出检查。RowMajor 输入保持 contiguous，遵循现有 DLPack 接口约定。
编译 key 相同不单独作为缓存命中的证明。
rank-3/4 字段投影由 host 打包单测覆盖，不作为此 RowMajor 真机示例的验证结论。

在已安装 DSL、PyTorch 和 torch_npu 的环境中运行：

```bash
python mixed_memref_arguments.py --device 0
```

当前支持的工具链在 GM memref 的 shape、stride 或 offset 无法从类型中静态确定时，
使用完整描述符调用约定，同一入口中的静态 GM memref 也展开为描述符。
因此 rank-R 静态 Tensor 的 host 参数包含 `allocated/aligned/offset`、
R 个 size 和 R 个 stride，而不是单个指针。
shape、stride 和 offset 全部静态且支持 bare pointer 转换的入口仍沿用 pointer ABI。
普通 rank-R 描述符与 TLA Dynamic-GM 的 13 字段参数组分别处理；后者还包含两个
originShape 参数。IR 中的动态 offset 可以在启动时取 0，不代表必须使用非零偏移。
这些 ABI 规则不改变 `from_dlpack` 的现有布局接入约束。

本示例通过 `mixed-memref-arguments` 用例接入 `tests/dsl_battery`，设备编号由测试入口提供。
