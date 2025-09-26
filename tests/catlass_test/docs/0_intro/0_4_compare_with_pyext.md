# 与`example/python_extension`的区别

catlass代码仓中，`python_extension`也是python接口，二者并非重复工作。以下是二者的比较：

|          | catlass_test                                 | python_extension                               |
| -------- | -------------------------------------------- | ---------------------------------------------- |
| 技术路线 | python->ctypes-><<<>>>                       | python->pybind/torchscript->shared_lib-><<<>>> |
| 语言     | 框架为python，模板为C++，调试方便            | 几乎全为C++，调试较难                          |
| 泛化能力 | 可通过运行时判断动态生成kernel               | 必须提前编写特化分支，会形成**箭头形代码**     |
| 接入难度 | 容易，可以使用内置适配器，或者继承内置适配器 | 较难，需要熟悉`libtorch`/`c10`/`pybind`的API   |
| 运行性能 | 理论较差，适合进行测试                       | 理论较好，适合接入正式业务                     |
| 兼容性   | 几乎为纯python代码，可以打包为whl复用        | 使用C扩展，与环境绑定，更换环境需重新编译      |
