/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_LAYOUT_CONV2D_HPP
#define CATLASS_LAYOUT_CONV2D_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/conv2d_coord.hpp"

namespace Catlass::layout {

/// Mapping function for input map
struct Fmap { // (Batch, Cin1, Hi, Wi, C0)
    /// Logical rank of tensor
    static constexpr int RANK = 5;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Ctor
    CATLASS_HOST_DEVICE
    Fmap(Index batch = 0, Index cin1 = 0, Index hi = 0, Index wi = 0, Index c0 = 0)
        : shape_(MakeCoord(batch, cin1, hi, wi, c0))
    {
        LongIndex strideBatch = cin1 * hi * wi * c0;
        LongIndex strideCin1 = hi * wi * c0;
        LongIndex strideHi = wi * c0;
        LongIndex strideWi = c0;
        LongIndex strideC0 = 1;
        stride_ = MakeCoord(strideBatch, strideCin1, strideHi, strideWi, strideC0);
    }

    CATLASS_HOST_DEVICE
    Fmap(Index batch, Index cin1, Index hi, Index wi, Index c0,
        LongIndex strideBatch, LongIndex strideCin1, LongIndex strideHi, LongIndex strideWi, LongIndex strideC0)
        : shape_(MakeCoord(batch, cin1, hi, wi, c0)), 
        stride_(MakeCoord(strideBatch, strideCin1, strideHi, strideWi, strideC0)) {}

    CATLASS_HOST_DEVICE
    Fmap(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (batch, cin1, hi, wi, c0)
    template <class Element>
    CATLASS_HOST_DEVICE
    static Fmap MakeLayout(Index batch, Index cin1, Index hi, Index wi, Index c0) {
        return Fmap(batch, cin1, hi, wi, c0);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (batch, cin1, hi, wi, c0)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(FmapCoord const &coord) const {
        return LongIndex(coord.batch()) * stride_[0] +
            LongIndex(coord.cin1()) * stride_[1] + 
            LongIndex(coord.hi()) * stride_[2] +
            LongIndex(coord.wi()) * stride_[3] +
            LongIndex(coord.c0()) * stride_[4]; 
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    Fmap GetTileLayout(FmapCoord const &tileShape) const {
        return Fmap(tileShape, stride());
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape &shape() {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index &shape(int idx) {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride &stride() {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index &stride(int idx) {
        return stride_[idx];
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    size_t Capacity() {
        return static_cast<size_t>(shape_[0]) * stride_[0];
    }

private:
    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for filter map
struct Filter { // (Cin1, Kh, Kw, Cout, C0)
    /// Logical rank of tensor
    static constexpr int RANK = 5;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Ctor
    CATLASS_HOST_DEVICE
    Filter(Index cin1 = 0, Index kh = 0, Index kw = 0, Index cout = 0, Index c0 = 0)
        : shape_(MakeCoord(cin1, kh, kw, cout, c0))
    {
        LongIndex strideCin1 = kh * kw * cout * c0;
        LongIndex strideKh = kw * cout * c0;
        LongIndex strideKw = cout * c0;
        LongIndex strideCout = c0;
        LongIndex strideC0 = 1;
        stride_ = MakeCoord(strideCin1, strideKh, strideKw, strideCout, strideC0);
    }

    CATLASS_HOST_DEVICE
    Filter(Index cin1, Index kh, Index kw, Index cout, Index c0,
        LongIndex strideCin1, LongIndex strideKh, LongIndex strideKw,
        LongIndex strideCout, LongIndex strideC0)
        : shape_(MakeCoord(cin1, kh, kw, cout, c0)), 
        stride_(MakeCoord(strideCin1, strideKh, strideKw, strideCout, strideC0)) {}

    CATLASS_HOST_DEVICE
    Filter(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (cin1, hi, wi, c0)
    template <class Element>
    CATLASS_HOST_DEVICE
    static Filter MakeLayout(Index cin1, Index kh, Index kw, Index cout, Index c0) {
        return Filter(cin1, kh, kw, cout, c0);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (cin1, kh, kw, cout, c0)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(FilterCoord const &coord) const {
        return LongIndex(coord.cin1()) * stride_[0] + 
            LongIndex(coord.kh()) * stride_[1] +
            LongIndex(coord.kw()) * stride_[2] +
            LongIndex(coord.cout()) * stride_[3] +
            LongIndex(coord.c0()) * stride_[4]; 
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    Filter GetTileLayout(FilterCoord const &tileShape) const {
        return Filter(tileShape, stride());
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape &shape() {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index &shape(int idx) {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride &stride() {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index &stride(int idx) {
        return stride_[idx];
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    size_t Capacity() {
        return static_cast<size_t>(shape_[0]) * stride_[0];
    }

private:
    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for output map
struct Output { // (Batch, Cout1, Ho, Wo, C0)
    /// Logical rank of tensor
    static constexpr int RANK = 5;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Ctor
    CATLASS_HOST_DEVICE
    Output(Index batch = 0, Index cout1 = 0, Index ho = 0, Index wo = 0, Index c0 = 0)
        : shape_(MakeCoord(batch, cout1, ho, wo, c0))
    {
        LongIndex strideBatch = cout1 * ho * wo * c0;
        LongIndex strideCout1 = ho * wo * c0;
        LongIndex strideHo = wo * c0;
        LongIndex strideWo = c0;
        LongIndex strideC0 = 1;
        stride_ = MakeCoord(strideBatch, strideCout1, strideHo, strideWo, strideC0);
    }

    CATLASS_HOST_DEVICE
    Output(Index batch, Index cout1, Index ho, Index wo, Index c0,
        LongIndex strideBatch, LongIndex strideCout1, LongIndex strideHo, LongIndex strideWo, LongIndex strideC0)
        : shape_(MakeCoord(batch, cout1, ho, wo, c0)), 
        stride_(MakeCoord(strideBatch, strideCout1, strideHo, strideWo, strideC0)) {}

    CATLASS_HOST_DEVICE
    Output(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (batch, cout1, ho, wo, c0)
    template <class Element>
    CATLASS_HOST_DEVICE
    static Output MakeLayout(Index batch = 0, Index cout1 = 0, Index ho = 0, Index wo = 0, Index c0 = 0) {
        return Output(batch, cout1, ho, wo, c0);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (batch, cout1, ho, wo, c0)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(OutputCoord const &coord) const {
        return LongIndex(coord.batch()) * stride_[0] + 
            LongIndex(coord.cout1()) * stride_[1] + 
            LongIndex(coord.ho()) * stride_[2] +
            LongIndex(coord.wo()) * stride_[3] +
            LongIndex(coord.c0()) * stride_[4]; 
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    Output GetTileLayout(OutputCoord const &tileShape) const {
        return Output(tileShape, stride());
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape &shape() {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index &shape(int idx) {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride &stride() {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index &stride(int idx) {
        return stride_[idx];
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    size_t Capacity() {
        return static_cast<size_t>(shape_[0]) * stride_[0];
    }

private:
    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

}  // namespace Catlass::layout

#endif  // CATLASS_LAYOUT_CONV2D_HPP