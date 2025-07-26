# 在CATLASS样例工程进行设备侧打印

提供设备侧打印函数`cce::printf`，用法与C标准库的printf一致。

- 支持`cube/vector/mix`算子
- 支持格式化字符串
- 支持打印常见整型与浮点数、指针、字符

  - ⚠️ **注意** 这个功能将在社区版未来的CANN 8.3开始支持，商用最新版现已支持。

# 使用示例

下面以对`09_splitk_matmul`为例，进行`设备侧打印`的使用说明。

## 插入打印代码

在想进行调试的代码段增加打印代码。

```diff
extern "C" __global__ __aicore__ void(...)
{
    // ...
    uint32_t tileLen;
    if (taskPerAiv > COMPUTE_LENGTH) {
        tileLen = COMPUTE_LENGTH;
    } else {
        tileLen = taskPerAiv;
    }
+   cce::printf("tileLen:%d\n", tileLen);
    // ...
}
```

## 编译运行

1. 基于[快速上手](../../README.md#快速上手)，打开工具的编译开关`--enable_print`， 使能设备侧打印特性编译算子样例。

```bash
bash scripts/build.sh --enable_print 09_splitk_matmul
```

2. 切换到可执行文件的编译目录`output/bin`下，直接执行算子样例程序。

```bash
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID（可选）
msdebug ./09_basic_matmul 256 512 1024 0
```

- ⚠ 注意事项
  - 目前`设备侧打印`仅支持打印`GM`和`SB(Scalar Buffer)`上的数值。

## 输出示例

输出结果

```bash
./09_splitk_matmul 256 512 1024 0
-----------------------------------------------------------------------------
---------------------------------HiIPU Print---------------------------------
-----------------------------------------------------------------------------
==> Logical Block 0
=> Physical Block

=> Physical Block
tileLen:2752

=> Physical Block
tileLen:2752

==> Logical Block 1
=> Physical Block

=> Physical Block
tileLen:2752

=> Physical Block
tileLen:2752

... # 此处省略

==> Logical Block 23
=> Physical Block

=> Physical Block
tileLen:2752

=> Physical Block
tileLen:2752

Compare success.
```
