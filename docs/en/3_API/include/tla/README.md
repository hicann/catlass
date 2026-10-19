# TLA API

TLA provides compile-time integers, heterogeneous tuples, layouts, and Tensor views. This page documents the most frequently used public APIs. For module layering and dependencies, see [TLA Module Architecture](../../../2_Design/02_tla/04_module_architecture.md).

## Header selection

| Need | Header |
| --- | --- |
| Static constants `Int`, `C`, `_` | `tla/integral/integral_constant.hpp` |
| Basic tuple types, `MakeTuple`, `get` | `tla/tuple.hpp` (compatibility entry point forwarding to `tla/tuple/tuple.hpp`) |
| Compile-time integer operators | `tla/integral/integral_math.hpp` |
| Tuple transformation, selection, and folding | `tla/tuple/tuple_algorithms.hpp` |
| Integer tuples, arithmetic, `product`, `ceil_div` | `tla/tuple/tuple_math.hpp` |
| Strides or coordinate/index conversion | `tla/stride.hpp` |
| Layouts and LayoutTag factories | `tla/layout.hpp` |
| Tensors, slicing, and tile views | `tla/tensor.hpp` |

`tla/layout.hpp` and `tla/tensor.hpp` remain the public entry points for layouts and Tensors. `tla/tuple.hpp` remains an entry point for basic tuple types, but does not include all tuple algorithms. When using lower-level APIs directly, explicitly include the header that provides them instead of relying on transitive includes.

For code that still directly includes the following old paths, select the current headers by functionality. These old paths do not have forwarding headers:

| Old path | Current headers (relative to `tla/`) |
| --- | --- |
| `tla/type_traits.hpp` | `utils/type_traits.hpp` |
| `tla/numeric/integral_constant.hpp` | Constant types: `integral/integral_constant.hpp`; constant operations: `integral/integral_math.hpp` |
| `tla/numeric/integer_sequence.hpp` | `integral/integral_sequence.hpp` |
| `tla/numeric/math.hpp` | `min`, `max`: `utils/math.hpp`; use `a + b` in place of `tla::add(a, b)` |
| `tla/int_tuple.hpp` | Tuple arithmetic: `tuple/tuple_math.hpp`; coordinate/stride conversion: `stride.hpp` |

The examples below assume the `tla` namespace.

## Compile-time integers and tuples

### Construction and inspection

| API | Description |
| --- | --- |
| `Int<N>{}`, `C<N>{}` | Compile-time constants. `C` deduces the value type. |
| `constant<T, v>`, `bool_constant<b>` | Aliases for `integral_constant<T, v>` and `Bool<b>`, respectively. |
| `make_tuple(a, b, ...)` / `MakeTuple(a, b, ...)` | Create a `tla::tuple`. |
| `get<I>(t)` / `get<I, J, ...>(t)` | Get an element or a recursively nested element. |
| `rank(t)`, `depth(t)` | Top-level size and maximum nesting depth; `rank_v<T>` and `depth_v<T>` are value forms. |
| `select<I...>(t)`, `take<B, E>(t)` | Select indices or the half-open range `[B, E)`. |
| `transform(t, f)`, `transform(t0, t1, f)`, `fold(f, init, t)` | Element-wise transformation and left fold. |
| `transform_apply(t, f, g)`, `transform_apply(t0, t1, f, g)` | Apply `f` to each element and pass the results to `g`. |

```cpp
auto shape = make_tuple(Int<2>{}, 4, 8);
auto first = get<0>(shape);             // Int<2>
auto tail = take<1, 3>(shape);          // (4, 8)
auto doubled = transform(shape, [](auto x) { return x * 2; });
```

`transform` accesses inputs through const references and constructs the output tuple by value. `transform_apply` accesses input elements directly, preserving their const/reference properties and intermediate references passed from `f` to `g`. Neither operation copies the input tuples beforehand. Both public return types are `auto`, so references in the final result are not preserved. For non-empty element types, `get<I>(t)` returns `T&`, `T const&`, or `T&&` for mutable lvalues, const lvalues, or non-const rvalues, respectively. Empty element types are returned by value.

### Constant evaluation and mutable results

With `tla/integral/integral_math.hpp` included, operations between constants preserve compile-time results. Some operations mixing a constant with a runtime integer also simplify statically. For example, `Int<0>{} * n`, where `n` is a runtime integer, returns `C<0>` rather than an ordinary integer. The distinction here is the return type, not the numerical result of this expression.

If the result is only used in subsequent expressions, keep `auto` to propagate static information. If it needs reassignment or modification through `+=`, `++`, or similar operations, first convert it to an ordinary integer type with an appropriate value range:

```cpp
#include "tla/integral/integral_math.hpp"

int n = 128;
auto staticOffset = tla::Int<0>{} * n;  // tla::C<0>, with value 0
// staticOffset += 16;                 // Unsupported: a static constant is not an accumulator
int offset = static_cast<int>(staticOffset);
offset += 16;                         // An ordinary integer supports accumulation
```

The same rule applies to static results propagated through tuple aggregation or layout coordinate calculations: do not assume that a return type deduced with `auto` is an ordinary mutable integer.

Logical operations between constants, and constant/runtime-integer logical operations whose results can be determined statically, return `C<true>` or `C<false>`. Overloaded `&&` and `||` **do not short-circuit**: both operands are evaluated. If short-circuit protection is required, explicitly convert both operands to bool, as in `bool(flag) && bool(check())`, or use a conditional statement. Ordinary bool logical operations still short-circuit.

### Integer-tuple arithmetic

`tuple_math.hpp` supports element-wise `+`, `-`, `*`, and `/`, with scalar broadcasting.

| API | Description |
| --- | --- |
| `product(x)` / `size(x)` | Recursive product; an empty tuple evaluates to 1. |
| `ceil_div(a, b)` | Ceiling division, including tuple structures. |
| `round_up(a, b)` | Round up to a multiple, with tuple broadcasting. |
| `inner_product(a, b)` | Recursive dot product. |
| `evenly_divides(a, b)` | Check divisibility. |

```cpp
auto tile = make_tuple(64, 128);
auto grid = ceil_div(make_tuple(130, 300), tile);  // (3, 3)
auto elements = product(tile);                      // 8192
```

## Strides and coordinates

| API | Description |
| --- | --- |
| `compact_row_major(shape[, start])` | Build compact row-major strides; the last stride is `start` (default: 1). |
| `compact_col_major(shape[, start])` | Build compact column-major strides; the first stride is `start`. |
| `compact_order(shape, order)` | Build compact strides from a static `order`. |
| `crd2idx(coord, shape, stride)` / `crd2offset(...)` | Convert a coordinate to an element offset; `crd2offset` is a compatibility alias. |
| `idx2crd(index, shape, stride)` | Convert a linear index to a coordinate. |

```cpp
auto shape = make_tuple(2, 4);
auto stride = compact_row_major(shape);             // (4, 1)
auto offset = crd2offset(make_tuple(1, 2), shape, stride);  // 6
```

## Layout

### Construction and access

`Shape`, `Stride`, and `Coord` are semantic aliases of `tla::tuple`; construct them with their factory functions.

```cpp
auto layout = MakeLayout(MakeShape(2, 4), MakeStride(4, 1));
auto offset = layout(MakeCoord(1, 2));  // 6

auto shape = layout.shape();
auto stride = layout.stride();
auto origin = layout.originShape();
```

`MakeLayout(shape, stride)` derives `originShape` from `shape`; the three-argument overload accepts an explicit logical valid range. These two forms have lowercase `make_layout` entry points with the same arguments. Single-argument and LayoutTag factories are not generally interchangeable by changing case. For a CATLASS LayoutTag, use:

```cpp
auto layout = MakeLayout<float, Catlass::layout::RowMajor>(rows, cols);
```

### Sublayouts and tiles

| API | Description |
| --- | --- |
| `get<I...>(layout)` | Extract a recursively nested sublayout. |
| `take<B, E>(layout)` | Project dimensions `[B, E)` across shape, stride, and originShape. |
| `select<I...>(layout)` | Project selected dimensions across all three components. |
| `coshape(layout)` / `cosize(layout)` | Compute codomain shape or element count. |
| `GetTileLayout(layout, tileShape, coord)` | Create a tile layout and crop `originShape` for tail tiles. |

`tileShape` and `coord` must be flat tuples. Depth-1 layouts support any rank; fractal layouts currently support rank 2 only.

## Tensor views

### Construction, access, and slicing

`MakeTensor` binds an existing AscendC Tensor to a TLA Layout and does not copy data:

```cpp
auto tensor = MakeTensor(builtinTensor, layout, Catlass::Arch::PositionGM{});
auto sub = MakeTensor(builtinTensor, layout, MakeCoord(1, 0), Catlass::Arch::PositionGM{});
```

| API | Description |
| --- | --- |
| `tensor.data()` / `layout()` / `coord()` | Return the underlying Tensor, layout, and view origin. |
| `shape()` / `stride()` / `originShape()` | Forward to the Layout. |
| `tensor(coord)` or `tensor(i, j, ...)` | Without `_`, access using `coord() + coord`. |
| `tensor(i, _, ...)` | Keep the `_` dimensions and return a sub-Tensor view. |

Slice arguments must be flat tuples whose rank matches the Tensor rank. Fixed indices are not runtime bounds-checked; callers must keep them within `originShape`.

### Tiles and similar Tensors

| API | Description |
| --- | --- |
| `GetTile(tensor, coord, shape)` | `coord` is an element coordinate; returns a tail-cropped tile view. |
| `TileView(tensor, tileCoord, tileShape)` | `tileCoord` is a tile coordinate; equivalent to `GetTile` at `tileCoord ⊙ tileShape`. |
| `MakeTensorLike<LayoutTag>(...)` | Create a view from a rank-1/2 reference Tensor's `originShape`. Without `layoutBase`, rank-1 inputs require `VectorLayout` and produce a unit-stride layout. |

`GetTile` and `TileView` require ranks matching the input Tensor. Both create views only and do not move data.
