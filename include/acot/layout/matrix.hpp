/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_LAYOUT_MATRIX_HPP
#define ACOT_LAYOUT_MATRIX_HPP

#include "acot/acot.hpp"
#include "acot/coord.hpp"
#include "acot/detail/alignment.hpp"
#include "acot/matrix_coord.hpp"

namespace acot::layout
{

    /// Mapping function for row-major matrices
    // Coord<2, T>表示二维坐标 (T x, T y)
    struct RowMajor
    {
    public:
        /// 逻辑维度（二维矩阵）
        static constexpr int RANK = 2;

        /// 索引类型（如行数、列数）
        using Index = uint32_t;

        /// 长索引类型（用于大范围偏移）
        using LongIndex = int64_t;

        /// 形状类型（存储行、列数）
        // 示例：Coord<2, T>, 坐标(T row, T col)
        using Shape = Coord<RANK, Index>;

        /// 步长类型（存储行、列步长）
        // 示例：Coord<2, T>, 坐标(T 行步长(lda), T 列步长(1))
        using Stride = Coord<RANK, LongIndex>;

    public:
        /// Constructor
        ACOT_HOST_DEVICE
        RowMajor(Index rows = 0, Index cols = 0)
            : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(cols), LongIndex(1))) {}

        /// Constructor
        ACOT_HOST_DEVICE
        RowMajor(Index rows, Index cols, LongIndex ldm)
            : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(ldm, LongIndex(1))) {}

        /// Ctor, 直接用shape和stride进行构造
        ACOT_HOST_DEVICE
        RowMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

        // 静态构造方法，创建内存对齐的布局，将lda对齐到BYTE_PER_C0 / sizeof(Element)的倍数
        template <class Element>
        ACOT_HOST_DEVICE static RowMajor MakeLayoutInUb(MatrixCoord const &shape)
        {
            return RowMajor(shape.row(), shape.column(), RoundUp<BYTE_PER_C0 / sizeof(Element)>(shape.column()));
        }

        // 偏移计算
        // 公式：offset = row * stride[0] + column * stride[1]
        // 列方向连续，步长为stride[1]=1, 行步长stride[0]=cols(默认)
        ACOT_HOST_DEVICE
        LongIndex GetOffset(MatrixCoord const &coord) const
        {
            return LongIndex(coord.row()) * stride_[0] + LongIndex(coord.column());
        }

        // 分块布局
        // 生成子矩阵的布局，步长和原矩阵保持一致
        ACOT_HOST_DEVICE
        RowMajor GetTileLayout(MatrixCoord const &tileShape) const
        {
            return RowMajor(MakeCoord(tileShape.row(), tileShape.column()),
                            stride());
        }

        /// Returns the shape of the layout
        // 返回常量的shape，不可修改
        ACOT_HOST_DEVICE
        Shape shape() const
        {
            return shape_;
        }

        /// Returns the shape of the layout
        // 返回shape的引用，可修改
        ACOT_HOST_DEVICE
        Shape &shape()
        {
            return shape_;
        }

        /// Returns the shape of the layout
        // 返回常量的shape中的单个维度，如行或列，不可修改
        ACOT_HOST_DEVICE
        typename Shape::Index shape(int idx) const
        {
            return shape_[idx];
        }

        /// Returns the shape of the layout
        // 返回shape中单个维度的引用，如行或列，可修改
        ACOT_HOST_DEVICE
        typename Shape::Index &shape(int idx)
        {
            return shape_[idx];
        }

        /// Returns the stride of the layout
        // 返回常量的stride，不可修改
        ACOT_HOST_DEVICE
        Stride stride() const
        {
            return stride_;
        }

        /// Returns the stride of the layout
        // 返回stride的引用，可修改
        ACOT_HOST_DEVICE
        Stride &stride()
        {
            return stride_;
        }

        /// Returns the stride of the layout
        // 返回常量的stride中的单个维度，如行步长或列步长，不可修改
        ACOT_HOST_DEVICE
        typename Stride::Index stride(int idx) const
        {
            return stride_[idx];
        }

        /// Returns the stride of the layout
        // 返回stride的引用中的单个维度，如行步长或列步长，不可修改
        ACOT_HOST_DEVICE
        typename Stride::Index &stride(int idx)
        {
            return stride_[idx];
        }

    private:
        //
        // Data members
        //

        /// Shape data member
        Shape shape_;

        /// Stride data member
        Stride stride_;
    };

    /// Mapping function for col-major matrices
    struct ColumnMajor
    {
    public:
        /// Logical rank of tensor
        static constexpr int RANK = 2;

        /// Index type used for coordinates
        using Index = uint32_t;

        /// Long index type used for offsets
        using LongIndex = int64_t;

        /// Logical coordinate
        using Shape = Coord<RANK, Index>;

        /// Stride vector
        using Stride = Coord<RANK, LongIndex>;

    public:
        // Methods

        /// Constructor
        ACOT_HOST_DEVICE
        ColumnMajor(Index rows = 0, Index cols = 0)
            : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), LongIndex(rows))) {}

        /// Constructor
        ACOT_HOST_DEVICE
        ColumnMajor(Index rows, Index cols, LongIndex ldm)
            : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), ldm)) {}

        /// Ctor
        ACOT_HOST_DEVICE
        ColumnMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

        /// Returns the offset of a coordinate in linear memory.
        /// Assumes coordinate has convention (row, column)
        ACOT_HOST_DEVICE
        LongIndex GetOffset(MatrixCoord const &coord) const
        {
            return LongIndex(coord.row()) + LongIndex(coord.column()) * stride_[1];
        }

        /// Returns the layout of a tile.
        ACOT_HOST_DEVICE
        ColumnMajor GetTileLayout(MatrixCoord const &tileShape) const
        {
            return ColumnMajor(MakeCoord(tileShape.row(), tileShape.column()),
                               stride());
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        Shape shape() const
        {
            return shape_;
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        Shape &shape()
        {
            return shape_;
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        typename Shape::Index shape(int idx) const
        {
            return shape_[idx];
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        typename Shape::Index &shape(int idx)
        {
            return shape_[idx];
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        Stride stride() const
        {
            return stride_;
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        Stride &stride()
        {
            return stride_;
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        typename Stride::Index stride(int idx) const
        {
            return stride_[idx];
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        typename Stride::Index &stride(int idx)
        {
            return stride_[idx];
        }

    private:
        //
        // Data members
        //

        /// Shape data member
        Shape shape_;

        /// Stride data member
        Stride stride_;
    };

    /// Mapping function for nZ matrices which is col-major inside fractal and row-major between fractal
    struct nZ
    {
    public:
        /// Logical rank of tensor
        // 4个维度
        static constexpr int RANK = 4;

        /// Index type used for coordinates
        // 坐标的索引
        using Index = uint32_t;

        /// Long index type used for offsets
        // 偏移量的索引
        using LongIndex = int64_t;

        /// Logical rank of orgshape
        // 原始的矩阵是二维的
        static constexpr int ORG_SHAPE_RANK = 2;

        /// Logical coordinate
        // Coord<2, Index>类型，保存原始矩阵的行数和列数
        using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

        /// Logical coordinate
        // 保存分形布局后的4个维度
        using Shape = Coord<RANK, Index>;

        /// Stride vector
        // 保存各个维度之间的步长
        using Stride = Coord<RANK, LongIndex>;

    public:
        // Methods

        /// Constructor
        ACOT_HOST_DEVICE
        nZ(Index orgRows = 0, /// Number of rows of origin matrices 原始矩阵行
           Index orgCols = 0, /// Number of cols of origin matrices 原始矩阵列

           // shape的四维
           Index rowsInFractal = 0, /// Number of rows inside the fractal 分形块内的行数
           Index rowsByFractal = 0, /// number of rows by the fractal 行方向的分形块数量
           Index colsInFractal = 0, /// number of cols inside the fractal 分形块内的列数
           Index colsByFractal = 0, /// number of cols by the fractal 列方向的分形块数量

           // stride的四维，其实就是分型内的行步长、列步长， 分形间的行步长、列步长
           LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
           LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
           LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
           LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
            : orgShape_(MakeCoord(orgRows, orgCols)),
              shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
              stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal))
        {
        }

        /// Ctor
        ACOT_HOST_DEVICE
        nZ(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

        /// Make the layout of a coordinate (row, column)
        template <class Element>
        ACOT_HOST_DEVICE static nZ MakeLayout(Index orgRows, Index orgCols)
        {
            static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
            static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
            Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
            Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
            return nZ(orgRows,
                      orgCols,

                      ELE_NUM_PER_C0, // 32B/sizeof(element)
                      rowsRound / ELE_NUM_PER_C0,
                      C0_NUM_PER_FRACTAL, // 16
                      colsRound / C0_NUM_PER_FRACTAL,

                      1,                          // 分形内行步长，nZ中分形内是列主序，因此相邻行之间是连续的，距离为1
                      colsRound * ELE_NUM_PER_C0, // 分形间行步长，就大Z两个起始地址之间的距离
                      ELE_NUM_PER_C0,             // 分形内列步长，就是32B/sizeof(element)
                      ELE_NUM_PER_FRACTAL);       // 分形间的列步长，实际上就是一个基块(512B)的元素数
        }

        /// Returns the offset of a coordinate in linear memory.
        /// Assumes coordinate has convention (row, column)
        // 将一个原始矩阵的坐标(x, y)转换到四维空间，计算线性偏移。就是算原始矩阵(x, y)在nZ分型后具体的线性偏移量是多少
        // 公式：offset = (行坐标 / rowsInFractal) * strideRowsByFractal + (列坐标 / colsInFractal) * strideColsByFractal;
        ACOT_HOST_DEVICE
        LongIndex GetOffset(MatrixCoord const &coord) const
        {
            return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3];
        }

        /// Returns the origin shape of the layout
        ACOT_HOST_DEVICE
        typename OrgShape::Index orgShape(int idx) const
        {
            return orgShape_[idx];
        }

        /// Returns the origin shape of the layout
        ACOT_HOST_DEVICE
        typename OrgShape::Index &orgShape(int idx)
        {
            return orgShape_[idx];
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        Shape shape() const
        {
            return shape_;
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        Shape &shape()
        {
            return shape_;
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        typename Shape::Index shape(int idx) const
        {
            return shape_[idx];
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        typename Shape::Index &shape(int idx)
        {
            return shape_[idx];
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        Stride stride() const
        {
            return stride_;
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        Stride &stride()
        {
            return stride_;
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        typename Stride::Index stride(int idx) const
        {
            return stride_[idx];
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        typename Stride::Index &stride(int idx)
        {
            return stride_[idx];
        }

    private:
        /// Origin Shape data member
        OrgShape orgShape_;

        /// Shape data member
        Shape shape_;

        /// Stride data member
        Stride stride_;
    };

    /// Mapping function for zN matrices which is row-major inside fractal and col-major between fractal
    struct zN
    {
    public:
        /// Logical rank of tensor
        static constexpr int RANK = 4;

        /// Index type used for coordinates
        using Index = uint32_t;

        /// Long index type used for offsets
        using LongIndex = int64_t;

        /// Logical rank of orgshape
        static constexpr int ORG_SHAPE_RANK = 2;

        /// Logical coordinate
        using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

        /// Logical coordinate
        using Shape = Coord<RANK, Index>;

        /// Stride vector
        using Stride = Coord<RANK, LongIndex>;

    public:
        // Methods

        /// Constructor
        ACOT_HOST_DEVICE
        zN(Index orgRows = 0,                 /// Number of rows of origin matrices
           Index orgCols = 0,                 /// Number of cols of origin matrices
           Index rowsInFractal = 0,           /// Number of rows inside the fractal
           Index rowsByFractal = 0,           /// number of rows by the fractal
           Index colsInFractal = 0,           /// number of cols inside the fractal
           Index colsByFractal = 0,           /// number of cols by the fractal
           LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
           LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
           LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
           LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
            : orgShape_(MakeCoord(orgRows, orgCols)),
              shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
              stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal))
        {
        }

        /// Ctor
        ACOT_HOST_DEVICE
        zN(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

        /// Make the layout of a coordinate (row, column)
        template <class Element>
        ACOT_HOST_DEVICE static zN MakeLayout(Index orgRows, Index orgCols)
        {
            static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
            static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
            Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
            Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
            return zN(orgRows,
                      orgCols,

                      C0_NUM_PER_FRACTAL,
                      rowsRound / C0_NUM_PER_FRACTAL,
                      ELE_NUM_PER_C0,
                      colsRound / ELE_NUM_PER_C0,

                      ELE_NUM_PER_C0,
                      ELE_NUM_PER_FRACTAL,
                      1,
                      rowsRound * ELE_NUM_PER_C0);
        }

        ACOT_HOST_DEVICE
        static zN MakeLayoutInL0C(MatrixCoord const &shape)
        {
            return zN(shape.row(),
                      shape.column(),
                      C0_NUM_PER_FRACTAL,
                      CeilDiv<C0_NUM_PER_FRACTAL>(shape.row()),
                      C0_NUM_PER_FRACTAL,
                      CeilDiv<C0_NUM_PER_FRACTAL>(shape.column()),
                      C0_NUM_PER_FRACTAL,
                      C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL,
                      1,
                      RoundUp<C0_NUM_PER_FRACTAL>(shape.row()) * C0_NUM_PER_FRACTAL);
        }

        /// Returns the offset of a coordinate in linear memory.
        /// Assumes coordinate has convention (row, column)
        ACOT_HOST_DEVICE
        LongIndex GetOffset(MatrixCoord const &coord) const
        {
            return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3];
        }

        /// Returns the layout of a tile.
        ACOT_HOST_DEVICE
        zN GetTileLayout(MatrixCoord const &tileShape) const
        {
            return zN(MakeCoord(tileShape.row(), tileShape.column()),
                      MakeCoord(shape(0), CeilDiv(tileShape.row(), shape(0)),
                                shape(2), CeilDiv(tileShape.column(), shape(2))),
                      stride());
        }

        /// Returns the origin shape of the layout
        ACOT_HOST_DEVICE
        typename OrgShape::Index orgShape(int idx) const
        {
            return orgShape_[idx];
        }

        /// Returns the origin shape of the layout
        ACOT_HOST_DEVICE
        typename OrgShape::Index &orgShape(int idx)
        {
            return orgShape_[idx];
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        Shape shape() const
        {
            return shape_;
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        Shape &shape()
        {
            return shape_;
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        typename Shape::Index shape(int idx) const
        {
            return shape_[idx];
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        typename Shape::Index &shape(int idx)
        {
            return shape_[idx];
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        Stride stride() const
        {
            return stride_;
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        Stride &stride()
        {
            return stride_;
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        typename Stride::Index stride(int idx) const
        {
            return stride_[idx];
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        typename Stride::Index &stride(int idx)
        {
            return stride_[idx];
        }

    private:
        /// Origin Shape data member
        OrgShape orgShape_;

        /// Shape data member
        Shape shape_;

        /// Stride data member
        Stride stride_;
    };

    /// Mapping function for zN matrices which is row-major inside fractal and row-major between fractal
    // 分形内是行优先，分形间也是行优先
    struct zZ
    {
    public:
        /// Logical rank of tensor
        static constexpr int RANK = 4;

        /// Index type used for coordinates
        using Index = uint32_t;

        /// Long index type used for offsets
        using LongIndex = int64_t;

        /// Logical rank of orgshape
        static constexpr int ORG_SHAPE_RANK = 2;

        /// Logical coordinate
        using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

        /// Logical coordinate
        using Shape = Coord<RANK, Index>;

        /// Stride vector
        using Stride = Coord<RANK, LongIndex>;

    public:
        // Methods

        /// Constructor
        ACOT_HOST_DEVICE
        zZ(Index orgRows = 0, /// Number of rows of origin matrices
           Index orgCols = 0, /// Number of cols of origin matrices

           Index rowsInFractal = 0, /// Number of rows inside the fractal
           Index rowsByFractal = 0, /// number of rows by the fractal
           Index colsInFractal = 0, /// number of cols inside the fractal
           Index colsByFractal = 0, /// number of cols by the fractal

           LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
           LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
           LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
           LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
            : orgShape_(MakeCoord(orgRows, orgCols)),
              shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
              stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal))
        {
        }

        /// Ctor
        ACOT_HOST_DEVICE
        zZ(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

        /// Make the layout of a coordinate (row, column)
        // 主要是关注这里，zZ的分形内结构和分形间结构，这些都是固定的
        template <class Element>
        ACOT_HOST_DEVICE static zZ MakeLayout(Index orgRows, Index orgCols)
        {
            static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
            static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
            Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows); // 行方向对齐16
            Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);     // 列方向对齐32B/sizeof(element)
            return zZ(orgRows,
                      orgCols,

                      C0_NUM_PER_FRACTAL,             // 分形内行数，16
                      rowsRound / C0_NUM_PER_FRACTAL, // 分形间行方向的分形数
                      ELE_NUM_PER_C0,                 // 分形内列数，32B/sizeof(element)
                      colsRound / ELE_NUM_PER_C0,     // 分形间列方向的分形数

                      ELE_NUM_PER_C0,                 // 分形内的行步长，就是分型内的列数
                      colsRound * C0_NUM_PER_FRACTAL, // 分形间的行步长，行方向上相邻两个分形的起始地址之间的距离，就是16乘以zZ矩阵的列数
                      1,                              // 分形内的列步长，因为分形内是行优先，因此为1
                      ELE_NUM_PER_FRACTAL);           // 分形间的列步长，就是一个分形的元素数
        }

        /// Returns the offset of a coordinate in linear memory.
        /// Assumes coordinate has convention (row, column)
        ACOT_HOST_DEVICE
        LongIndex GetOffset(MatrixCoord const &coord) const
        {
            return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3];
        }

        /// Returns the origin shape of the layout
        ACOT_HOST_DEVICE
        typename OrgShape::Index orgShape(int idx) const
        {
            return orgShape_[idx];
        }

        /// Returns the origin shape of the layout
        ACOT_HOST_DEVICE
        typename OrgShape::Index &orgShape(int idx)
        {
            return orgShape_[idx];
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        Shape shape() const
        {
            return shape_;
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        Shape &shape()
        {
            return shape_;
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        typename Shape::Index shape(int idx) const
        {
            return shape_[idx];
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        typename Shape::Index &shape(int idx)
        {
            return shape_[idx];
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        Stride stride() const
        {
            return stride_;
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        Stride &stride()
        {
            return stride_;
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        typename Stride::Index stride(int idx) const
        {
            return stride_[idx];
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        typename Stride::Index &stride(int idx)
        {
            return stride_[idx];
        }

    private:
        /// Origin Shape data member
        OrgShape orgShape_;

        /// Shape data member
        Shape shape_;

        /// Stride data member
        Stride stride_;
    };

    /// Mapping function for nN matrices which is column-major inside fractal and column-major between fractal
    // 分形内是列优先，分形间也是列优先
    struct nN
    {
    public:
        /// Logical rank of tensor
        static constexpr int RANK = 4;

        /// Index type used for coordinates
        using Index = uint32_t;

        /// Long index type used for offsets
        using LongIndex = int64_t;

        /// Logical rank of orgshape
        static constexpr int ORG_SHAPE_RANK = 2;

        /// Logical coordinate
        using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

        /// Logical coordinate
        using Shape = Coord<RANK, Index>;

        /// Stride vector
        using Stride = Coord<RANK, LongIndex>;

    public:
        // Methods

        /// Constructor
        ACOT_HOST_DEVICE
        nN(Index orgRows = 0, /// Number of rows of origin matrices
           Index orgCols = 0, /// Number of cols of origin matrices

           Index rowsInFractal = 0, /// Number of rows inside the fractal
           Index rowsByFractal = 0, /// number of rows by the fractal
           Index colsInFractal = 0, /// number of cols inside the fractal
           Index colsByFractal = 0, /// number of cols by the fractal

           LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
           LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
           LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
           LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
            : orgShape_(MakeCoord(orgRows, orgCols)),
              shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
              stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal))
        {
        }

        /// Ctor
        ACOT_HOST_DEVICE
        nN(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

        /// Make the layout of a coordinate (row, column)
        // 主要是关注这里，nN的分形内结构和分形间结构，这些都是固定的
        template <class Element>
        ACOT_HOST_DEVICE static nN MakeLayout(Index orgRows, Index orgCols)
        {
            static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
            static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
            Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);     // 行方向对齐32B/sizeof(element)
            Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols); // 列方向对齐16
            return nN(orgRows,
                      orgCols,

                      ELE_NUM_PER_C0,                 // 分形内行数，32B/sizeof(element)
                      rowsRound / ELE_NUM_PER_C0,     // 分形间行方向的分形数
                      C0_NUM_PER_FRACTAL,             // 分形内列数，16
                      colsRound / C0_NUM_PER_FRACTAL, // 分形间列方向的分形数

                      1,                               // 分形内的行步长，因为分形内是列优先，所以是1
                      ELE_NUM_PER_FRACTAL,             // 分形间的行步长，就是一个分形的元素数
                      ELE_NUM_PER_C0,                  // 分形内的列步长
                      rowsRound * C0_NUM_PER_FRACTAL); // 分形间的列步长，列方向上相邻两个分形的起始地址之间的距离，就是分形数乘以nN矩阵的列数
        }

        /// Returns the offset of a coordinate in linear memory.
        /// Assumes coordinate has convention (row, column) 这里的算法, nN 和zZ 应该是一样的
        ACOT_HOST_DEVICE
        LongIndex GetOffset(MatrixCoord const &coord) const
        {
            return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3];
        }

        /// Returns the origin shape of the layout
        ACOT_HOST_DEVICE
        typename OrgShape::Index orgShape(int idx) const
        {
            return orgShape_[idx];
        }

        /// Returns the origin shape of the layout
        ACOT_HOST_DEVICE
        typename OrgShape::Index &orgShape(int idx)
        {
            return orgShape_[idx];
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        Shape shape() const
        {
            return shape_;
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        Shape &shape()
        {
            return shape_;
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        typename Shape::Index shape(int idx) const
        {
            return shape_[idx];
        }

        /// Returns the shape of the layout
        ACOT_HOST_DEVICE
        typename Shape::Index &shape(int idx)
        {
            return shape_[idx];
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        Stride stride() const
        {
            return stride_;
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        Stride &stride()
        {
            return stride_;
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        typename Stride::Index stride(int idx) const
        {
            return stride_[idx];
        }

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        typename Stride::Index &stride(int idx)
        {
            return stride_[idx];
        }

    private:
        /// Origin Shape data member
        OrgShape orgShape_;

        /// Shape data member
        Shape shape_;

        /// Stride data member
        Stride stride_;
    };

} // namespace acot::layout

#endif // ACOT_LAYOUT_MATRIX_HPP