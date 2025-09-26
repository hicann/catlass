# 工作原理

```mermaid
graph LR
A[torch.Tensor]-->B[模板参数]-->C[模板代码替换]-->D[动态链接库]-->F[ctypes调用]
A-->E[运行时参数]-->F
```
