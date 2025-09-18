# 接口设计原则

catlass_test的最终呈现是一个类torch的接口，使用者可以直接传入torch.Tensor来进行计算。

- 对于`torch`中可以找到的算子，比如`basic_matmul`功能上对应`torch.mm`，接口原则上要求兼容对应的`torch`接口，即变量名和类型都要一致。若需要增加参数，需要保证存在对应的默认值。

```py
# torch
def mm(input: torch.Tensor, 
        mat2: torch.Tensor, 
        out_dtype: torch.dtype) -> torch.Tensor:
    pass

# catlass_test
def basic_matmul(input: torch.Tensor, 
        mat2: torch.Tensor, 
        out_dtype: torch.dtype
        # 增加的参数需要保证存在对应的默认值
        custom_param_1: bool = False
        ) -> torch.Tensor:
    pass

```

- 对于`torch`中不存在的算子，应当尽量贴近`torch`算子的参数命名规范。
