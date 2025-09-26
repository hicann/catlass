# 快速体验catlass_test

catlass_test是对catlass算子的python封装，不同于example中的pybind封装或者Python DSL，你可以用最少的代码接入并调起catlass算子，从而快速进行体验和测试。

## 用catlass_test调用算子

```py
import torch
import catlass_test
a = torch.ones([4, 4], dtype=torch.float16).npu()
b = torch.ones([4, 4], dtype=torch.float16).npu()
# catlass_test interface calling
c_catlass_test = catlass_test.basic_matmul(a, b)
print(c_catlass_test)
>>> tensor([[4., 4., 4., 4.],
        [4., 4., 4., 4.],
        [4., 4., 4., 4.],
        [4., 4., 4., 4.]], device='npu:0', dtype=torch.float16)
```

## 将自定义算子`custom_matmul`接入catlass_test

```cpp
// custom_matmul.hpp
template <ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>
inline int32_t CustomMatmul(GemmCoord problemShape, uint8_t *deviceA, uint8_t *deviceB, uint8_t *deviceC, aclrtStream stream){
    // catlass API combinations
}
```

```py
# custom_matmul.py
import torch
from catlass_test.adapter import MatmulAdapter
def custom_matmul(input: torch.Tensor, mat2: torch.Tensor)->torch.Tensor:
    case = MatmulCase(
        "custom_matmul.hpp"
        {"A": input, "B": mat2},
    )
    case.run()
    return case.get_tensor("C")
a = torch.ones([4, 4], dtype=torch.float16).npu()
b = torch.ones([4, 4], dtype=torch.float16).npu()
c = custom_matmul(a, b)
print(c)
>>> tensor([[4., 4., 4., 4.],
        [4., 4., 4., 4.],
        [4., 4., 4., 4.],
        [4., 4., 4., 4.]], device='npu:0', dtype=torch.float16)
```
