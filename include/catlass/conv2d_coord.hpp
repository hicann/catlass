/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV2D_COORD_HPP
#define CATLASS_CONV2D_COORD_HPP

namespace Catlass {

template <
    uint32_t Ho_ = 1,
    uint32_t Wo_ = 1,
    uint32_t Cin1_ = 1
>
struct Conv2dFmapL1Shape { // (Ho, Wo, Cin1)
    static constexpr uint32_t Ho = Ho_;
    static constexpr uint32_t Wo = Wo_;
    static constexpr uint32_t Cin1 = Cin1_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord() {
        return MakeCoord(Ho, Wo, Cin1);
    }
};

template <
    uint32_t Cout_ = 16,
    uint32_t Cin1_ = 1
>
struct Conv2dFilterL1Shape { // (Cout, Cin1)
    static constexpr uint32_t Cout = Cout_;
    static constexpr uint32_t Cin1 = Cin1_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<2> ToCoord() {
        return MakeCoord(Cout, Cin1);
    }
};

template <
    uint32_t M_ = 16,
    uint32_t N_ = 16,
    uint32_t K_ = 16
>
struct Conv2dL0Shape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord() {
        return MakeCoord(M, N, K);
    }
};

template <
    uint32_t Batch_ = 1,
    uint32_t Cin1_ = 1,
    uint32_t Hi_ = 1,
    uint32_t Wi_ = 1,
    uint32_t C0_ = 16
>
struct FmapShape { // (Batch, Cin1, Hi, Wi, C0)
    static constexpr uint32_t Batch = Batch_;
    static constexpr uint32_t Cin1 = Cin1_;
    static constexpr uint32_t Hi = Hi_;
    static constexpr uint32_t Wi = Wi_;
    static constexpr uint32_t C0 = C0_;

    static constexpr int64_t Cin = Cin1 * C0;
    static constexpr int64_t HWi = Hi * Wi;

    static constexpr int64_t COUNT = Batch * Cin * HWi;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<5> ToCoord() {
        return MakeCoord(Batch, Cin1, Hi, Wi, C0);
    }
};

struct FmapCoord : public Coord<5, uint32_t> { // (Batch, Cin1, Hi, Wi, C0)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t BATCH_INDEX = 0;
    static constexpr uint32_t CIN1_INDEX = 1;
    static constexpr uint32_t HI_INDEX = 2;
    static constexpr uint32_t WI_INDEX = 3;
    static constexpr uint32_t C0_INDEX = 4;

    /// Default ctor
    CATLASS_HOST_DEVICE
    FmapCoord() {}

    /// Constructs from Coord<5>
    CATLASS_HOST_DEVICE
    FmapCoord(Coord<5, Index> const &coord) : Base(coord) {}

    /// Helper to construct from Cin1, Hi, Wi, C0
    CATLASS_HOST_DEVICE
    FmapCoord(Index batch, Index cin1, Index hi, Index wi, Index c0)
    : Base(MakeCoord(batch, cin1, hi, wi, c0)) {}

    CATLASS_HOST_DEVICE
    FmapCoord(LongIndex batch, LongIndex cin1, LongIndex hi, LongIndex wi, LongIndex c0)
    : Base(MakeCoord(Index(batch), Index(cin1), Index(hi), Index(wi), Index(c0))) {}

    CATLASS_HOST_DEVICE
    Index const &cin1() const {
        return this->At(CIN1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cin1() {
        return this->At(CIN1_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &batch() const {
        return this->At(BATCH_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &batch() {
        return this->At(BATCH_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &hi() const {
        return this->At(HI_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &hi() {
        return this->At(HI_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &wi() const {
        return this->At(WI_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &wi() {
        return this->At(WI_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &c0() const {
        return this->At(C0_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &c0() {
        return this->At(C0_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    FmapCoord operator+(Base const &b) const {
        return FmapCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    FmapCoord &operator+=(Base const &b) {
        Base::operator+=(b);
        return *this;
    }
};

template <
    uint32_t Cin1_ = 1,
    uint32_t Kh_ = 1,
    uint32_t Kw_ = 1,
    uint32_t Cout_ = 16,
    uint32_t C0_ = 16
>
struct FilterShape { // (Cin1, Kh, Kw, Cout, C0)
    static constexpr uint32_t Cin1 = Cin1_;
    static constexpr uint32_t Kh = Kh_;
    static constexpr uint32_t Kw = Kw_;
    static constexpr uint32_t Cout = Cout_;
    static constexpr uint32_t C0 = C0_;

    static constexpr int64_t Cin = Cin1 * C0;
    static constexpr int64_t Khw = Kh * Kw;

    static constexpr int64_t COUNT = Cin * Khw * Cout;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<5> ToCoord() {
        return MakeCoord(Cin1, Kh, Kw, Cout, C0);
    }
};

struct FilterCoord : public Coord<5, uint32_t> { // (Cin1, Kh, Kw, Cout, C0)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t CIN1_INDEX = 0;
    static constexpr uint32_t KH_INDEX = 1;
    static constexpr uint32_t KW_INDEX = 2;
    static constexpr uint32_t COUT_INDEX = 3;
    static constexpr uint32_t C0_INDEX = 4;

    /// Default ctor
    CATLASS_HOST_DEVICE
    FilterCoord() {}

    /// Constructs from Coord<5>
    CATLASS_HOST_DEVICE
    FilterCoord(Coord<5, Index> const &coord) : Base(coord) {}

    /// Helper to construct from Cin1, Kh, Kw, Cout, C0
    CATLASS_HOST_DEVICE
    FilterCoord(Index cin1, Index kh, Index kw, Index cout, Index c0)
    : Base(MakeCoord(cin1, kh, kw, cout, c0)) {}

    CATLASS_HOST_DEVICE
    FilterCoord(LongIndex cin1, LongIndex kh, LongIndex kw, LongIndex cout, LongIndex c0)
    : Base(MakeCoord(Index(cin1), Index(kh), Index(kw), Index(cout), Index(c0))) {}

    CATLASS_HOST_DEVICE
    Index const &cin1() const {
        return this->At(CIN1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cin1() {
        return this->At(CIN1_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &kh() const {
        return this->At(KH_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &kh() {
        return this->At(KH_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &kw() const {
        return this->At(KW_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &kw() {
        return this->At(KW_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cout() const {
        return this->At(COUT_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cout() {
        return this->At(COUT_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &c0() const {
        return this->At(C0_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &c0() {
        return this->At(C0_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    FilterCoord operator+(Base const &b) const {
        return FilterCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    FilterCoord &operator+=(Base const &b) {
        Base::operator+=(b);
        return *this;
    }
};

template <
    uint32_t Batch_ = 1,
    uint32_t Cout1_ = 1,
    uint32_t Ho_ = 1,
    uint32_t Wo_ = 1,
    uint32_t C0_ = 16
>
struct OutputShape { // (Batch, Cout1, Ho, Wo, C0)
    static constexpr uint32_t Batch = Batch_;
    static constexpr uint32_t Cout1 = Cout1_;
    static constexpr uint32_t Ho = Ho_;
    static constexpr uint32_t Wo = Wo_;
    static constexpr uint32_t C0 = C0_;

    static constexpr int64_t Cout = Cout1 * C0;
    static constexpr int64_t HWo = Ho * Wo;

    static constexpr int64_t COUNT = Batch * Cout * HWo;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<5> ToCoord() {
        return MakeCoord(Batch, Cout1, Ho, Wo, C0);
    }
};

struct OutputCoord : public Coord<5, uint32_t> { // (Batch, Cout1, Ho, Wo, C0)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;
    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t BATCH_INDEX = 0;
    static constexpr uint32_t COUT1_INDEX = 1;
    static constexpr uint32_t HO_INDEX = 2;
    static constexpr uint32_t WO_INDEX = 3;
    static constexpr uint32_t C0_INDEX = 4;

    /// Default ctor
    CATLASS_HOST_DEVICE
    OutputCoord() {}

    /// Constructs from Coord<5>
    CATLASS_HOST_DEVICE
    OutputCoord(Coord<5, Index> const &coord) : Base(coord) {}

    /// Helper to construct from Batch, Cout1, Ho, Wo, C0
    CATLASS_HOST_DEVICE
    OutputCoord(Index batch, Index cout1, Index ho, Index wo, Index c0)
    : Base(MakeCoord(batch, cout1, ho, wo, c0)) {}

    CATLASS_HOST_DEVICE
    OutputCoord(LongIndex batch, LongIndex cout1, LongIndex ho, LongIndex wo, LongIndex c0)
    : Base(MakeCoord(Index(batch), Index(cout1), Index(ho), Index(wo), Index(c0))) {}

    CATLASS_HOST_DEVICE
    Index const &batch() const {
        return this->At(BATCH_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &batch() {
        return this->At(BATCH_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cout1() const {
        return this->At(COUT1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cout1() {
        return this->At(COUT1_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &ho() const {
        return this->At(HO_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &ho() {
        return this->At(HO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &wo() const {
        return this->At(WO_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &wo() {
        return this->At(WO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &c0() const {
        return this->At(C0_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &c0() {
        return this->At(C0_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    OutputCoord operator+(Base const &b) const {
        return OutputCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    OutputCoord &operator+=(Base const &b) {
        Base::operator+=(b);
        return *this;
    }
};

template <
    uint32_t Ho_ = 1,
    uint32_t Wo_ = 1,
    uint32_t Cout_ = 16
>
struct HoWoCoutShape { // (Ho, Wo, Cout)
    static constexpr uint32_t Ho = Ho_;
    static constexpr uint32_t Wo = Wo_;
    static constexpr uint32_t Cout = Cout_;

    static constexpr int64_t HWo = Ho * Wo;
    static constexpr int64_t COUNT = HWo * Cout;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord() {
        return MakeCoord(Ho, Wo, Cout);
    }
};

struct HoWoCoutCoord : public Coord<3, uint32_t> { // (Ho, Wo, Cout)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=3
    using Base = Coord<3, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t HO_INDEX = 0;
    static constexpr uint32_t WO_INDEX = 1;
    static constexpr uint32_t COUT_INDEX = 2;

    /// Default ctor
    CATLASS_HOST_DEVICE
    HoWoCoutCoord() {}

    /// Constructs from Coord<3>
    CATLASS_HOST_DEVICE
    HoWoCoutCoord(Coord<3, Index> const &coord) : Base(coord) {}

    /// Helper to construct from Ho, Wo, Cout
    CATLASS_HOST_DEVICE
    HoWoCoutCoord(Index ho, Index wo, Index cout)
    : Base(MakeCoord(ho, wo, cout)) {}

    CATLASS_HOST_DEVICE
    HoWoCoutCoord(LongIndex ho, LongIndex wo, LongIndex cout)
    : Base(MakeCoord(Index(ho), Index(wo), Index(cout))) {}

    CATLASS_HOST_DEVICE
    Index const &ho() const {
        return this->At(HO_INDEX);
    }
    
    CATLASS_HOST_DEVICE
    Index &ho() {
        return this->At(HO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &wo() const {
        return this->At(WO_INDEX);
    }
    
    CATLASS_HOST_DEVICE
    Index &wo() {
        return this->At(WO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index howo() const {
        return this->At(HO_INDEX) * this->At(WO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cout() const {
        return this->At(COUT_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &cout() {
        return this->At(COUT_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    HoWoCoutCoord operator+(Base const &b) const {
        return HoWoCoutCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    HoWoCoutCoord &operator+=(Base const &b) {
        Base::operator+=(b);
        return *this;
    }
};

template <
    uint32_t Batch_ = 1,
    uint32_t H_ = 1,
    uint32_t W_ = 1,
    uint32_t Cout_ = 16,
    uint32_t Cin1_ = 1
>
struct Conv2d5HdShape { // (Batch, H, W, Cout, Cin1)
    static constexpr uint32_t Batch = Batch_;
    static constexpr uint32_t H = H_;
    static constexpr uint32_t W = W_;
    static constexpr uint32_t Cout = Cout_;
    static constexpr uint32_t Cin1 = Cin1_;

    static constexpr int64_t HW = H * W;
    
    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<5> ToCoord() {
        return MakeCoord(Batch, H, W, Cout, Cin1);
    }
};

struct Conv2d5HdCoord : public Coord<5, uint32_t> { // (Batch, H, W, Cout, Cin1)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t BATCH_INDEX = 0;
    static constexpr uint32_t H_INDEX = 1;
    static constexpr uint32_t W_INDEX = 2;
    static constexpr uint32_t COUT_INDEX = 3;
    static constexpr uint32_t CIN1_INDEX = 4;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv2d5HdCoord() {}

    /// Constructs from Coord<5>
    CATLASS_HOST_DEVICE
    Conv2d5HdCoord(Coord<5, Index> const &coord) : Base(coord) {}

    /// Helper to construct from Batch, H, W, Cout, Cin1
    CATLASS_HOST_DEVICE
    Conv2d5HdCoord(Index batch, Index h, Index w, Index cout, Index cin1)
    : Base(MakeCoord(batch, h, w, cout, cin1)) {}

    CATLASS_HOST_DEVICE
    Conv2d5HdCoord(LongIndex batch, LongIndex h, LongIndex w, LongIndex cout, LongIndex cin1)
    : Base(MakeCoord(Index(batch), Index(h), Index(w), Index(cout), Index(cin1))) {}

    CATLASS_HOST_DEVICE
    Index const &batch() const {
        return this->At(BATCH_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &batch() {
        return this->At(BATCH_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &h() const {
        return this->At(H_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &h() {
        return this->At(H_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &w() const {
        return this->At(W_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &w() {
        return this->At(W_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cout() const {
        return this->At(COUT_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cout() {
        return this->At(COUT_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cin1() const {
        return this->At(CIN1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cin1() {
        return this->At(CIN1_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    Conv2d5HdCoord operator+(Base const &b) const {
        return Conv2d5HdCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    Conv2d5HdCoord &operator+=(Base const &b) {
        Base::operator+=(b);
        return *this;
    }

    CATLASS_HOST_DEVICE
    auto GetCoordHoWoCout() const {
        return this->GetCoordByAxis<H_INDEX, W_INDEX, COUT_INDEX>();
    }
};

class Conv2dConfigs {
public:
    typedef uint8_t ShortIndex;
    typedef uint32_t Index;
    static constexpr uint32_t C0 = 16;
    using Ks = Coord<2, ShortIndex>;
    using Pads = Coord<4, ShortIndex>;
    using Strides = Coord<2, ShortIndex>;
    using Dilations = Coord<2, ShortIndex>;
private:
    Ks ks;
    Pads pads;
    Strides strides;
    Dilations dilations;
public:
    Conv2dConfigs(ShortIndex kh = 0, ShortIndex kw = 0,
        ShortIndex padLeft = 0, ShortIndex padRight = 0, ShortIndex padTop = 0, ShortIndex padBottom = 0,
        ShortIndex strideH = 0, ShortIndex strideW = 0, ShortIndex dilationH = 0, ShortIndex dilationW = 0)
    : ks(MakeCoord(kh, kw)), pads(MakeCoord(padLeft, padRight, padTop, padBottom)), 
      strides(MakeCoord(strideH, strideW)), dilations(MakeCoord(dilationH, dilationW)) {}

    CATLASS_HOST_DEVICE
    ShortIndex const &kh() const {
        return this->ks[0];
    }
    CATLASS_HOST_DEVICE
    ShortIndex &kh() {
        return this->ks[0];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &kw() const {
        return this->ks[1];
    }
    CATLASS_HOST_DEVICE
    ShortIndex &kw() {
        return this->ks[1];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padLeft() const {
        return this->pads[0];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padLeft() {
        return this->pads[0];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padRight() const {
        return this->pads[1];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padRight() {
        return this->pads[1];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padTop() const {
        return this->pads[2];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padTop() {
        return this->pads[2];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padBottom() const {
        return this->pads[3];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padBottom() {
        return this->pads[3];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &strideH() const {
        return this->strides[0];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &strideH() {
        return this->strides[0];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &strideW() const {
        return this->strides[1];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &strideW() {
        return this->strides[1];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &dilationH() const {
        return this->dilations[0];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &dilationH() {
        return this->dilations[0];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &dilationW() const {
        return this->dilations[1];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &dilationW() {
        return this->dilations[1];
    }
};

struct Conv2dParams {
public:
    typedef uint8_t ShortIndex;
    typedef uint32_t Index;
    static constexpr uint32_t C0 = 16;
    using Pads = Coord<4, ShortIndex>;
    using Strides = Coord<2, ShortIndex>;
    using Dilations = Coord<2, ShortIndex>;
private:
    // Batch, Hi, Wi, Cin, Cout, Kh, Kw
    FmapCoord fmapShape; // {Batch, Cin1, Hi, Wi, C0}
    FilterCoord filterShape; // {Cin1, Kh, Kw, Cout, C0}
    OutputCoord outputShape; // {Batch, Cout1, Ho, Wo, C0}
    Conv2dConfigs configs; // {Ks, Pads, Strides, Dilations}
    Conv2d5HdCoord postIm2colShape; // {Batch, Ho, Wo, Cout, Cin1}
public:
    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv2dParams() {}

    CATLASS_HOST_DEVICE
    Conv2dParams(Index batch, Index hi, Index wi, Index cin, Index cout,
        ShortIndex kh, ShortIndex kw,
        ShortIndex padLeft, ShortIndex padRight, ShortIndex padTop, ShortIndex padBottom,
        ShortIndex strideH, ShortIndex strideW,
        ShortIndex dilationH, ShortIndex dilationW)
    : fmapShape(MakeCoord(batch, CeilDiv(cin, C0), hi, wi, C0)),
      filterShape(MakeCoord(CeilDiv(cin, C0), (Index)kh, (Index)kw, cout, C0)),
      configs(kh, kw, padLeft, padRight, padTop, padBottom, strideH, strideW, dilationH, dilationW)
    {
        Index cout1 = CeilDiv(cout, C0);
        Index ho = (hi + padTop + padBottom - dilationH * (kh - 1) - 1) / strideH + 1;
        Index wo = (wi + padLeft + padRight - dilationW * (kw - 1) - 1) / strideW + 1;
        outputShape = MakeCoord(batch, cout1, ho, wo, C0);
        postIm2colShape = MakeCoord(batch, ho, wo, cout, filterShape.cin1());
    }

    CATLASS_HOST_DEVICE
    static Conv2dParams MakeConv2dParams(
        const Index* dataSizes, // [Batch, Hi, Wi, Cin, Cout]
        const ShortIndex* filterSizes, // [Kh, Kw]
        const ShortIndex* pads, // [padLeft, padRight, padTop, padBottom]
        const ShortIndex* strides, // [strideH, strideW]
        const ShortIndex* dilations) // [dilationH, dilationW]
    {
        return Conv2dParams(
            dataSizes[0], dataSizes[1], dataSizes[2], dataSizes[3], dataSizes[4],
            filterSizes[0], filterSizes[1],
            pads[0], pads[1], pads[2], pads[3],
            strides[0], strides[1], strides[2], strides[3]);
    }

    CATLASS_HOST_DEVICE
    Conv2dConfigs const &getConv2dConfigs() const {
        return this->configs;
    }

    CATLASS_HOST_DEVICE
    OutputCoord const &getOutputShape() const {
        return this->outputShape;
    }

    CATLASS_HOST_DEVICE
    Conv2d5HdCoord const &getPostIm2colShape() const {
        return this->postIm2colShape;
    }

    CATLASS_HOST_DEVICE
    Index const &batch() const {
        return this->fmapShape.batch();
    }
    CATLASS_HOST_DEVICE
    Index &batch() {
        return this->fmapShape.batch();
    }

    CATLASS_HOST_DEVICE
    Index const &hi() const {
        return this->fmapShape.hi();
    }
    CATLASS_HOST_DEVICE
    Index &hi() {
        return this->fmapShape.hi();
    }

    CATLASS_HOST_DEVICE
    Index const &wi() const {
        return this->fmapShape.wi();
    }
    CATLASS_HOST_DEVICE
    Index &wi() {
        return this->fmapShape.wi();
    }

    CATLASS_HOST_DEVICE
    Index cin() const {
        return this->fmapShape.cin1() * C0;
    }

    CATLASS_HOST_DEVICE
    Index const &cin1() const {
        return this->fmapShape.cin1();
    }
    CATLASS_HOST_DEVICE
    Index &cin1() {
        return this->fmapShape.cin1();
    }

    CATLASS_HOST_DEVICE
    Index const &cout() const {
        return this->filterShape.cout();
    }
    CATLASS_HOST_DEVICE
    Index &cout() {
        return this->filterShape.cout();
    }

    CATLASS_HOST_DEVICE
    Index const &cout1() const {
        return this->outputShape.cout1();
    }
    CATLASS_HOST_DEVICE
    Index &cout1() {
        return this->outputShape.cout1();
    }

    CATLASS_HOST_DEVICE
    Index coutRound() const {
        return (this->filterShape.cout() + C0 - 1) / C0 * C0;
    }

    CATLASS_HOST_DEVICE
    Index const &ho() const {
        return this->outputShape.ho();
    }
    CATLASS_HOST_DEVICE
    Index &ho() {
        return this->outputShape.ho();
    }

    CATLASS_HOST_DEVICE
    Index const &wo() const {
        return this->outputShape.wo();
    }
    CATLASS_HOST_DEVICE
    Index &wo() {
        return this->outputShape.wo();
    }

    CATLASS_HOST_DEVICE
    Index howo() const {
        return this->outputShape.ho() * this->outputShape.wo();
    }

    CATLASS_HOST_DEVICE
    Index howoRound() const {
        return (this->howo() + C0 - 1) / C0 * C0;
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &kh() const {
        return this->configs.kh();
    }
    CATLASS_HOST_DEVICE
    ShortIndex &kh() {
        return this->configs.kh();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &kw() const {
        return this->configs.kw();
    }
    CATLASS_HOST_DEVICE
    ShortIndex &kw() {
        return this->configs.kw();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padTop() const {
        return this->configs.padTop();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padTop() {
        return this->configs.padTop();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padBottom() const {
        return this->configs.padBottom();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padBottom() {
        return this->configs.padBottom();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padLeft() const {
        return this->configs.padLeft();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padLeft() {
        return this->configs.padLeft();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padRight() const {
        return this->configs.padRight();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padRight() {
        return this->configs.padRight();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &strideH() const {
        return this->configs.strideH();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &strideH() {
        return this->configs.strideH();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &strideW() const {
        return this->configs.strideW();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &strideW() {
        return this->configs.strideW();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &dilationH() const {
        return this->configs.dilationH();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &dilationH() {
        return this->configs.dilationH();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &dilationW() const {
        return this->configs.dilationW();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &dilationW() {
        return this->configs.dilationW();
    }
};

} // namespace Catlass

#endif // CATLASS_CONV2D_COORD_HPP
