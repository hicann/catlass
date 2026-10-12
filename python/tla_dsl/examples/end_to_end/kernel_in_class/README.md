# 以类实例组织 kernel

这些示例是 [基础 VADD 端到端示例](../basic_vadd/README.md)的类版本：编译期配置、kernel 与编译入口封装在同一个类里，再交给 `tla.compile`。

| 示例 | 内容 |
|---|---|
| `vector_add_with_static_length.py` | 向量长度 `n_ele` 作为实例属性，在 kernel 编排时折叠为常量；不同 `n_ele` 各得一份编译产物 |

写法与约束见[编译与启动](../../../docs/zh/kernel_development/core_concepts/compile_and_launch.md)。
