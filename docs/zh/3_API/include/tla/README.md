# TLA API

TLA 提供编译期整数、异质 tuple、Layout 与 Tensor 视图等基础接口。本页只覆盖算子开发中最常用的公共 API；模块分层和依赖请参见 [TLA 模块架构](../../../2_Design/02_tla/04_module_architecture.md)。

## 头文件选择

| 需求 | 头文件 |
| --- | --- |
| 编译期常量 `Int`、`C`、`_` | `tla/integral/integral_constant.hpp` |
| tuple 基础类型、`MakeTuple`、`get` | `tla/tuple.hpp`（兼容入口，转发 `tla/tuple/tuple.hpp`） |
| 编译期整数运算符 | `tla/integral/integral_math.hpp` |
| tuple 变换、选择、折叠 | `tla/tuple/tuple_algorithms.hpp` |
| 整数 tuple、算术、`product`、`ceil_div` | `tla/tuple/tuple_math.hpp` |
| stride 或坐标/线性索引换算 | `tla/stride.hpp` |
| Layout 与 LayoutTag 工厂 | `tla/layout.hpp` |
| Tensor、切片及 tile 视图 | `tla/tensor.hpp` |

`tla/layout.hpp` 和 `tla/tensor.hpp` 仍是布局和 Tensor 的公共入口；`tla/tuple.hpp` 保留为 tuple 基础类型入口，但不包含全部 tuple 算法。直接使用底层接口时，应显式包含其所属头文件，不依赖间接包含。

对于仍直接引用以下旧路径的代码，请按所用功能选择当前头文件；这些旧路径不提供转发头：

| 旧路径 | 当前头文件（均相对于 `tla/`） |
| --- | --- |
| `tla/type_traits.hpp` | `utils/type_traits.hpp` |
| `tla/numeric/integral_constant.hpp` | 常量类型：`integral/integral_constant.hpp`；常量运算：`integral/integral_math.hpp` |
| `tla/numeric/integer_sequence.hpp` | `integral/integral_sequence.hpp` |
| `tla/numeric/math.hpp` | `min`、`max`：`utils/math.hpp`；原 `tla::add(a, b)` 使用 `a + b` |
| `tla/int_tuple.hpp` | tuple 算术：`tuple/tuple_math.hpp`；坐标/stride 换算：`stride.hpp` |

以下示例默认使用 `tla` 命名空间。

## 编译期整数与 tuple

### 常用构造与查询

| API | 说明 |
| --- | --- |
| `Int<N>{}`、`C<N>{}` | 编译期常量。`C` 自动推导常量值的类型。 |
| `constant<T, v>`、`bool_constant<b>` | 分别是 `integral_constant<T, v>`、`Bool<b>` 的别名。 |
| `make_tuple(a, b, ...)` / `MakeTuple(a, b, ...)` | 构造 `tla::tuple`。 |
| `get<I>(t)` / `get<I, J, ...>(t)` | 取 tuple 元素或递归子元素。 |
| `rank(t)`、`depth(t)` | 顶层元素个数、最大嵌套深度；对应值版本为 `rank_v<T>`、`depth_v<T>`。 |
| `select<I...>(t)`、`take<B, E>(t)` | 按索引选择，或取半开区间 `[B, E)`。 |
| `transform(t, f)`、`transform(t0, t1, f)`、`fold(f, init, t)` | 逐元素变换、从左到右折叠。 |
| `transform_apply(t, f, g)`、`transform_apply(t0, t1, f, g)` | 逐元素调用 `f`，将各结果传给 `g`。 |

```cpp
auto shape = make_tuple(Int<2>{}, 4, 8);
auto first = get<0>(shape);             // Int<2>
auto tail = take<1, 3>(shape);          // (4, 8)
auto doubled = transform(shape, [](auto x) { return x * 2; });
```

`transform` 通过 const 引用访问输入，按值构造输出 tuple；`transform_apply` 直接访问输入元素，保留输入的 const / 引用属性及 `f` 到 `g` 的中间引用。两者都不预先复制输入 tuple，公开返回类型均为 `auto`，不保留最终结果的引用。非空类型元素的 `get<I>(t)` 对可变左值、const 左值和非 const 右值分别返回 `T&`、`T const&` 和 `T&&`；空类型元素按值返回。

### 常量求值与可修改的结果

包含 `tla/integral/integral_math.hpp` 后，常量之间的运算保留编译期结果；部分常量与运行时整数的混合运算也会静态简化。例如，`Int<0>{} * n`（`n` 为运行时整数）返回 `C<0>`，而不是普通整数。这里需要关注的是返回类型，不是该表达式的数值结果。

如果结果仅用于后续表达式，可以保留 `auto`，继续传播静态信息；如果需要赋新值或进行 `+=`、`++` 等修改，应先转换为取值范围合适的普通整数类型：

```cpp
#include "tla/integral/integral_math.hpp"

int n = 128;
auto staticOffset = tla::Int<0>{} * n;  // tla::C<0>，数值为 0
// staticOffset += 16;                 // 不支持：编译期常量不能用作累加变量
int offset = static_cast<int>(staticOffset);
offset += 16;                         // 普通整数，可以累加
```

这一原则同样适用于 tuple 聚合或布局坐标计算中传播出的静态结果：不要假定 `auto` 推导出的返回类型一定是可修改的普通整数。

双常量逻辑运算以及能静态确定结果的常量 / 运行时整数混合逻辑运算，返回 `C<true>` 或 `C<false>`。选中重载的 `&&` / `||` **不会短路**，两侧操作数都会求值；需要短路保护时，应将两侧显式转为 bool，例如 `bool(flag) && bool(check())`，或使用条件语句。普通 bool 逻辑运算仍保持短路。

### 整数 tuple 运算

`tuple_math.hpp` 支持 tuple 与标量广播，及逐元素 `+`、`-`、`*`、`/`。高频的聚合和 tiling 接口如下：

| API | 说明 |
| --- | --- |
| `product(x)` / `size(x)` | 递归相乘；空 tuple 的积为 1。 |
| `ceil_div(a, b)` | 向上整除；支持 tuple 结构。 |
| `round_up(a, b)` | 向上对齐到 `b` 的倍数；支持 tuple 广播。 |
| `inner_product(a, b)` | 递归点积。 |
| `evenly_divides(a, b)` | 判断是否可整除。 |

```cpp
auto tile = make_tuple(64, 128);
auto grid = ceil_div(make_tuple(130, 300), tile);  // (3, 3)
auto elements = product(tile);                      // 8192
```

## stride 与坐标换算

| API | 说明 |
| --- | --- |
| `compact_row_major(shape[, start])` | 生成行优先紧凑 stride；最末维 stride 为 `start`（默认 1）。 |
| `compact_col_major(shape[, start])` | 生成列优先紧凑 stride；首维 stride 为 `start`。 |
| `compact_order(shape, order)` | 按静态 `order` 生成紧凑 stride。 |
| `crd2idx(coord, shape, stride)` / `crd2offset(...)` | 坐标到线性元素偏移；后者是前者的兼容别名。 |
| `idx2crd(index, shape, stride)` | 线性索引到坐标。 |

```cpp
auto shape = make_tuple(2, 4);
auto stride = compact_row_major(shape);             // (4, 1)
auto offset = crd2offset(make_tuple(1, 2), shape, stride);  // 6
```

## Layout

### 构造与访问

`Shape`、`Stride`、`Coord` 分别是 `tla::tuple` 的语义别名；推荐用对应工厂构造。

```cpp
auto layout = MakeLayout(MakeShape(2, 4), MakeStride(4, 1));
auto offset = layout(MakeCoord(1, 2));  // 6

auto shape = layout.shape();
auto stride = layout.stride();
auto origin = layout.originShape();
```

`MakeLayout(shape, stride)` 自动从 `shape` 推导 `originShape`；三参数重载可以显式保留逻辑有效范围。这两种调用对应同参数的小写 `make_layout` 入口；单参数及 LayoutTag 工厂不能统一按大小写互换。对于 CATLASS LayoutTag，使用：

```cpp
auto layout = MakeLayout<float, Catlass::layout::RowMajor>(rows, cols);
```

### 子布局与 tile 布局

| API | 说明 |
| --- | --- |
| `get<I...>(layout)` | 递归取一个子布局。 |
| `take<B, E>(layout)` | 取 `[B, E)` 维，并同时投影 shape、stride、originShape。 |
| `select<I...>(layout)` | 按维度索引投影三者。 |
| `coshape(layout)` / `cosize(layout)` | 计算布局余域形状/元素数。 |
| `GetTileLayout(layout, tileShape, coord)` | 创建 tile 的 Layout，并按元素坐标裁剪 tail tile 的 `originShape`。 |

`GetTileLayout` 要求 `tileShape` 与 `coord` 是一层 tuple。深度为 1 的布局可用任意 rank；分形布局当前仅支持 rank 2。

## Tensor 视图

### 构造、访问和切片

`MakeTensor` 绑定已有 AscendC Tensor 与 TLA Layout，不进行拷贝：

```cpp
auto tensor = MakeTensor(builtinTensor, layout, Catlass::Arch::PositionGM{});
auto sub = MakeTensor(builtinTensor, layout, MakeCoord(1, 0), Catlass::Arch::PositionGM{});
```

| API | 说明 |
| --- | --- |
| `tensor.data()` / `layout()` / `coord()` | 返回底层 Tensor、布局、当前视图起点。 |
| `shape()` / `stride()` / `originShape()` | 转发到布局。 |
| `tensor(coord)` 或 `tensor(i, j, ...)` | 无 `_` 时按 `coord() + coord` 访问底层元素。 |
| `tensor(i, _, ...)` | 用 `_` 保留对应维度，返回子 Tensor 视图。 |

切片参数必须是一层 tuple，rank 必须等于 Tensor rank。固定维度的索引不会做运行时边界检查；调用方必须保证其位于 `originShape` 的有效范围内。

### tile 与相似 Tensor

| API | 说明 |
| --- | --- |
| `GetTile(tensor, coord, shape)` | `coord` 是元素坐标，返回 tail 已裁剪的 tile 视图。 |
| `TileView(tensor, tileCoord, tileShape)` | `tileCoord` 是 tile 坐标，等价于以 `tileCoord ⊙ tileShape` 调用 `GetTile`。 |
| `MakeTensorLike<LayoutTag>(...)` | 以参考 Tensor 的 `originShape` 创建目标 Layout 的视图；支持 rank 1/2，不传 `layoutBase` 时一维输入要求 `VectorLayout`，生成步长为 1 的布局。 |

`GetTile` 和 `TileView` 的 rank 必须与输入 Tensor 一致，且都只创建视图，不移动数据。
