# 简单的运行时编译

以一段简单的代码为例：

```cpp
#include <cstdint>
#include <acl/acl.h>

struct Tensor {
    aclDataType dtype;
    uint8_t *addr;
    uint64_t dataLen;
};

template <typename Element>
void demo_template(uint8_t *addr, uint64_t dataLen) {
    // do some operation with type Element...
}
```

其中Tensor结构体可以看成是外部组件传递进来的变量。`dtype`是运行时参数，而`Element`是编译期参数。运行时用例的`dtype`是多种多样的，传统的做法是预判可能的情况，然后编写条件分支：

```cpp
void demo(Tensor tensor) {
    if (tensor.dtype == ACL_FLOAT16) {
        demo_template<half>(tensor.addr, tensor.dataLen);
    } else if (tensor.dtype == ACL_BF16) {
        demo_template<bfloat16_t>(tensor.addr, tensor.dataLen);
    }
}
```

catlass常用于开发matmul类算子，一般有两个输入一个输出；常用的dtype有`half` `bfloat16_t` `int8_t`等。如果我们为三个变量适配三种情况，那么代码将会膨胀到惊人的大小：

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202509158292652/29527330/26c71d669e16491f856c5fbdd156cc05.png)

由此可看出，这种调用方法代码繁琐，根据可变模板参数的数量`n`和值域大小`m`，分支复杂度可达到`O(m^n)`，使用这种方式搭建测试，随着特性支持的增加，代码会越来越难以维护。

采用动态编译，仅维护一个模板是一种不错的方案。若使用模板运行时编译的方式，代码类似如下：

```cpp
#ifndef DTYPE1
#define DTYPE1 half
#endif
#ifndef DTYPE2
#define DTYPE2 half
#endif
#ifndef DTYPE3
#define DTYPE3 half
#endif
void demo(Tensor tensor1,Tensor tensor2, Tensor tensor3) {
    demo_template<DTYPE1, DTYPE2, DTYPE3>(tensor1.addr, tensor1.dataLen, tensor2.addr, tensor2.dataLen, tensor3.addr, tensor3.dataLen);
}
```

编译：

```bash
bisheng -xcce --cce-aicore-arch=dav-c220 -DDTYPE1=half -DDTYPE2=half -DDTYPE3=half demo.cpp -o --shared libdemo_half_half_half.so
bisheng -xcce --cce-aicore-arch=dav-c220 -DDTYPE1=half -DDTYPE2=half -DDTYPE3=bfloat16_t demo.cpp -o --shared libdemo_half_half_bfloat16_t.so
bisheng -xcce --cce-aicore-arch=dav-c220 -DDTYPE1=half -DDTYPE2=half -DDTYPE3=int8_t demo.cpp -o --shared libdemo_half_half_int8_t.so
# ...
bisheng -xcce --cce-aicore-arch=dav-c220 -DDTYPE1=int8_t -DDTYPE2=int8_t -DDTYPE3=int8_t demo.cpp -o --shared libdemo_int8_t_int8_t_int8_t.so
```

这些编译命令可以通过外层框架轻松地生成。在运行时通过提取输入数据，生成编译命令并编译，然后使用`ctypes`、`dlopen`等手段调用，也可以增加cache等特性，将维护代码从130+行减少到1行，极大地提升了测试效率。
