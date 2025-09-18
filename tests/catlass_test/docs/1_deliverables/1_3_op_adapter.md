
# 模板适配器编写
完成算子模板的编写后，我们就需要编写对应的模板适配器。
模板适配器与算子模板是一对多的关系，一个适配器可以为多个模板所用，例如`basic_matmul`/`padding_matmul`/`optimized_matmul`等可以共享一个`matmul`适配器。
但如果我们引入并不通用的算子，比如`attention`类算子，那就需要为其单独进行适配。

# 基础数据结构

为了快速获取模板中的using代码串，以及方便结构体传参，catlass_test实现了一些catlass中的数据结构。

- `GemmCoord`: 一个包装`m`/`n`/`k`的结构体，用于存放`problemShape`，借助ctypes实现

```py
class GemmCoord(Structure):
    _fields_ = [
        ("m", c_uint),
        ("n", c_uint),
        ("k", c_uint),
    ]
```

- `GemmShape`: 一个包装`m`/`n`/`k`的**静态**结构体，用于存放`TileShape`，可通过调用成员快速获取一些预置的字符串

```py
class GemmShape:
    def __init__(self, m: int, n: int, k: int) -> None:
        self.m = m
        self.n = n
        self.k = k

    def get_alias(self, name: str) -> str:
        return f"using {name} = GemmShape<{self.m}, {self.n}, {self.k}>;\n"

    def get_str(self):
        return f"{self.m}_{self.n}_{self.k}"
```

# 基类

通常对于不同的case，我们都需要单独进行参数的提取，然后进行编译，因此，适配器的实例被称为case。Case基类的定义如下：
