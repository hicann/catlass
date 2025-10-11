# MatmulGelu Example Readme
## 代码组织
```
├── 27_matmul_gelu
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── matmul_gelu.cpp # 主文件
```
## 功能介绍

执行以下函数的计算

Gelu:
$$
out = Gelu(a × b)

$$
其中Gelu的公式为：
$$
Gelu(x) =0.5∗x∗(1+Tanh(\sqrt {2/π}∗(x+0.044715∗x^3 )))
$$
Sigmoid：
$$
Sigmoid(x)=\frac{1}{1+e^{-x}}
$$
Tanh:
$$
\begin{align}Tanh的实现为：\\Tanh(x) &= \frac{(e^x - e^{-x})}{(e^x + e^{-x})}\\Tanh(x) &= \frac{(e^x - e^{-x})\times e^{-x}}{(e^x + e^{-x}) \times e^{-x} }\\Tanh(x) &= \frac{1 - e^{-2x} }{1 + e^{-2x}}\\Tanh(x) &= 1 - 2\times \frac{e^{-2x}}{1 + e^{-2x}}\\Tanh(x) &= 1 - 2\times (1 - \frac{1}{1 + e^{-2x}})\\Tanh(x) &= 1 - 2\times (1 - Sigmoid(2x))\\因此可以化简为：Tanh(x) &= 2\times Sigmoid(2x) - 1\end{align}

$$
最后Gelu的计算形式为：
$$
\begin{align}Gelu(x) &=0.5∗x∗(1+Tanh(\sqrt {2/π}∗(x+0.044715∗x^3 )))\\把Z&=\sqrt {2/π}∗(x+0.044715∗x^3)\\Gelu(x) &=0.5∗x∗ (1 + 2\times Sigmoid(2Z) - 1)\\Gelu(x) &=x∗Sigmod( 2Z )))\\展开Z可得Gelu(x) &=x∗Sigmod(\sqrt {8/π}∗(x+0.044715∗x^3 ))\\其中\sqrt {8/π}可近似为 1.595769，可得：\\Gelu(x) &=x∗Sigmod( 1.595769∗(x+0.044715∗x^3 ))\\展开Sigmoid，可得:\\Gelu(x) &= \frac{x}{1+e^{-1.595769∗(x+0.044715∗x^3 )}}\\\end{align}
$$

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 27_matmul_gelu
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./27_matmul_gelu 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```