/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_COORD_HPP
#define CATLASS_CONV_COORD_HPP

#include "catlass/coord.hpp"

namespace Catlass {

/// Shape of conv3d operation
struct Conv3dParams {
public:
    typedef uint32_t Index;
    static constexpr uint32_t N0 = 16;
    using OrgFmapShape = Coord<5, Index>;
    using OrgFilterShape = Coord<5, Index>;
    using OrgOutShape = Coord<5, Index>;
    using Cin1N1Cout1C0 = Coord<4, Index>;
    using Strides = Coord<3, Index>;
    using Pads = Coord<3, Index>;
    using Dilations = Coord<3, Index>;
private:
    OrgFmapShape orgFmapShape_;
    OrgFilterShape orgFilterShape_;
    OrgOutShape orgOutShape_;
    Strides strides_;
    Pads pads_;
    Dilations dilations_;
    Cin1N1Cout1C0 cin1n1cout1c0_;
public:
    CATLASS_HOST_DEVICE
    Conv3dParams(
        Index N = 1,
        Index Cin = 1,
        Index Di = 1,
        Index Hi = 1,
        Index Wi = 1,
        Index C0 = 16,
        Index Cin1 = 1,
        Index Cout = 1,
        Index Kd = 1,
        Index Kh = 1,
        Index Kw = 1,
        Index N1 = 1,
        Index Do = 1,
        Index Ho = 1,
        Index Wo = 1,
        Index Cout1 = 1,
        Index padHead = 0,
        Index padTop = 0,
        Index padLeft = 0,
        Index strideD = 1,
        Index strideH = 1,
        Index strideW = 1,
        Index dilationD = 1,
        Index dilationH = 1,
        Index dilationW = 1
    )
    : orgFmapShape_(MakeCoord(N, Cin, Di, Hi, Wi)),
      orgFilterShape_(MakeCoord(Cout, Cin, Kd, Kh, Kw)),
      orgOutShape_(MakeCoord(N, Cout, Do, Ho, Wo)),
      cin1n1cout1c0_(MakeCoord(Cin1, N1, Cout1, C0)),
      pads_(MakeCoord(padHead, padTop, padLeft)),
      strides_(MakeCoord(strideD, strideH, strideW)),
      dilations_(MakeCoord(dilationD, dilationH, dilationW))
    {}

    template <class Element>
    CATLASS_HOST_DEVICE
    static Conv3dParams MakeConvCoord(
        const uint32_t* fmapOriShape,
        const uint32_t* filterOriShape,
        const uint32_t* paddings,
        const uint32_t* strides,
        const uint32_t* dilations)
    {
        constexpr uint32_t C0 = BYTE_PER_C0 / sizeof(Element);
        return Conv3dParams(
            fmapOriShape[0],
            fmapOriShape[1],
            fmapOriShape[2],
            fmapOriShape[3],
            fmapOriShape[4],
            C0, //C0
            (fmapOriShape[1] + C0 - 1) / C0,  //cin1
            filterOriShape[0],
            filterOriShape[2],
            filterOriShape[3],
            filterOriShape[4],
            (filterOriShape[0] + N0 - 1) / N0,   //N1
            (fmapOriShape[2] + paddings[0] * 2 - dilations[0] * (filterOriShape[2] - 1) - 1) / strides[0] + 1,  //Do
            (fmapOriShape[3] + paddings[1] * 2 - dilations[1] * (filterOriShape[3] - 1) - 1) / strides[1] + 1,  //Ho
            (fmapOriShape[4] + paddings[2] * 2 - dilations[2] * (filterOriShape[4] - 1) - 1) / strides[2] + 1,  //Wo
            (filterOriShape[0] + C0 - 1) / C0,   //Cout1
            paddings[0],
            paddings[1],
            paddings[2],
            strides[0],
            strides[1],
            strides[2],
            dilations[0],
            dilations[1],
            dilations[2]
        );
    }
    

    //fmapShape
    CATLASS_HOST_DEVICE
    Index const &n() const { return orgFmapShape_[0]; }
    CATLASS_HOST_DEVICE
    Index const &cin() const { return orgFmapShape_[1]; }
    CATLASS_HOST_DEVICE
    Index const &di() const { return orgFmapShape_[2]; }
    CATLASS_HOST_DEVICE
    Index const &hi() const { return orgFmapShape_[3]; }
    CATLASS_HOST_DEVICE
    Index const &wi() const { return orgFmapShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const &cin1() const { return cin1n1cout1c0_[0]; }
    CATLASS_HOST_DEVICE
    Index const &cin0() const { return cin1n1cout1c0_[3]; }

    //filterShape
    CATLASS_HOST_DEVICE
    Index const &kd() const { return orgFilterShape_[2]; }
    CATLASS_HOST_DEVICE
    Index const &kh() const { return orgFilterShape_[3]; }
    CATLASS_HOST_DEVICE
    Index const &kw() const { return orgFilterShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const khkw() const { return orgFilterShape_[3] * orgFilterShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const kdc1khkw() const { return orgFilterShape_[2] * cin1n1cout1c0_[0] * orgFilterShape_[3] * orgFilterShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const &n1() const { return cin1n1cout1c0_[1]; }
    CATLASS_HOST_DEVICE
    Index const &n0() const { return this->N0; }

    //outShape
    CATLASS_HOST_DEVICE
    Index const &cout() const { return orgOutShape_[1]; }
    CATLASS_HOST_DEVICE
    Index const &dout() const { return orgOutShape_[2]; }
    CATLASS_HOST_DEVICE
    Index const &ho() const { return orgOutShape_[3]; }
    CATLASS_HOST_DEVICE
    Index const &wo() const { return orgOutShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const &cout1() const { return cin1n1cout1c0_[2]; }
    CATLASS_HOST_DEVICE
    Index const &cout0() const { return cin1n1cout1c0_[3]; }

    /// paddings
    CATLASS_HOST_DEVICE
    Index const &padhead() const { return pads_[0]; }
    CATLASS_HOST_DEVICE
    Index const &padtail() const { return pads_[0]; }
    CATLASS_HOST_DEVICE
    Index const &padtop() const { return pads_[1]; }
    CATLASS_HOST_DEVICE
    Index const &padbottom() const { return pads_[1]; }
    CATLASS_HOST_DEVICE
    Index const &padleft() const { return pads_[2]; }
    CATLASS_HOST_DEVICE
    Index const &padright() const { return pads_[2]; }

    /// strideSize
    CATLASS_HOST_DEVICE
    Index const &sD() const { return strides_[0]; }
    CATLASS_HOST_DEVICE
    Index const &sH() const { return strides_[1]; }
    CATLASS_HOST_DEVICE
    Index const &sW() const { return strides_[2]; }

    /// dilationSize
    CATLASS_HOST_DEVICE
    Index const &dD() const { return dilations_[0]; }
    CATLASS_HOST_DEVICE
    Index const dilatedKernelD() const { return 1 + (orgFilterShape_[2] - 1) * dilations_[0]; }
    CATLASS_HOST_DEVICE
    Index const &dH() const { return dilations_[1]; }
    CATLASS_HOST_DEVICE
    Index const dilatedKernelH() const { return 1 + (orgFilterShape_[3] - 1) * dilations_[1]; }
    CATLASS_HOST_DEVICE
    Index const &dW() const { return dilations_[2]; }
    CATLASS_HOST_DEVICE
    Index const dilatedKernelW() const { return 1 + (orgFilterShape_[4] - 1) * dilations_[2]; }
    
    ///// used in block
    CATLASS_HOST_DEVICE
    Index const howo() const { return orgOutShape_[3] * orgOutShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const wicin0() const { return orgFmapShape_[4] * cin1n1cout1c0_[3]; }
    CATLASS_HOST_DEVICE
    Index const khkwcin0() const { return orgFilterShape_[3] * orgFilterShape_[4] * cin1n1cout1c0_[3]; }
    CATLASS_HOST_DEVICE
    Index const alignCinKhKwKd() const { return cin1n1cout1c0_[0] * cin1n1cout1c0_[3] * orgFilterShape_[2] * orgFilterShape_[3] * orgFilterShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const kdcin1() const { return cin1n1cout1c0_[0] * orgFilterShape_[2]; }
    CATLASS_HOST_DEVICE
    Index const fmapOneBatchSize() const {return orgFmapShape_[2] * cin1n1cout1c0_[0] * cin1n1cout1c0_[3] * orgFmapShape_[3] * orgFmapShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const outputOneBatchSize() const {return orgOutShape_[2] * cin1n1cout1c0_[2] * cin1n1cout1c0_[3] * orgOutShape_[3] * orgOutShape_[4]; }

};

template <
    uint32_t singleCoreNo_ = 1,
    uint32_t singleCoreDo_ = 1,
    uint32_t singleCoreCo1_ = 1,
    uint32_t singleCoreHoWo_ = 1
>
struct ConvCoreShape {
    static uint32_t const singleCoreNo = singleCoreNo_;
    static uint32_t const singleCoreDo = singleCoreDo_;
    static uint32_t const singleCoreCo1 = singleCoreCo1_;
    static uint32_t const singleCoreHoWo = singleCoreHoWo_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<4> ToCoord() {
        return MakeCoord(singleCoreNo, singleCoreDo, singleCoreCo1, singleCoreHoWo);
    }
};

template<
    uint32_t mAL1_ = 1,
    uint32_t Kd_ = 1,
    uint32_t Ci1_ = 1
>
struct ConvFmapL1Shape {
    static uint32_t const mAL1 = mAL1_;
    static uint32_t const Kd = Kd_;
    static uint32_t const Ci1 = Ci1_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord() {
        return MakeCoord(mAL1, Kd, Ci1);
    }
};

template<
    uint32_t Kd_ = 1,
    uint32_t Ci1_ = 1,
    uint32_t nBL1_ = 1
>
struct ConvFilterL1Shape {
    static uint32_t const Kd = Kd_;
    static uint32_t const Ci1 = Ci1_;
    static uint32_t const nBL1 = nBL1_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord() {
        return MakeCoord(Kd, Ci1, nBL1);
    }
};

template <
    uint32_t mL0_ = 1,
    uint32_t kL0_ = 1,
    uint32_t nL0_ = 1
>
struct ConvL0Shape {
    static uint32_t const mL0 = mL0_;
    static uint32_t const kL0 = kL0_;
    static uint32_t const nL0 = nL0_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord() {
        return MakeCoord(mL0, kL0, nL0);
    }
};

struct Conv3d6HdCoord : public Coord<4, uint32_t> {  // (N, D, C1, HW)
    using Index = uint32_t;

    using Base = Coord<4, Index>;

    static constexpr int N_INDEX = 0;
    static constexpr int D_INDEX = 1;
    static constexpr int C1_INDEX = 2;
    static constexpr int HW_INDEX = 3;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv3d6HdCoord() {}

    CATLASS_HOST_DEVICE
    Conv3d6HdCoord(Index n, Index d, Index c1, Index hw)
    : Base(MakeCoord(n, d, c1, hw)) {}

    CATLASS_HOST_DEVICE
    Index const& n() const {
        return this->At(N_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& n() {
        return this->At(N_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const& d() const {
        return this->At(D_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& d() {
        return this->At(D_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const& c1() const {   //需要包含c0
        return this->At(C1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& c1() {
        return this->At(C1_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const& hw() const {
        return this->At(HW_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& hw() {
        return this->At(HW_INDEX);
    }
};

struct Conv3dFracZ3dCoord : public Coord<2, uint32_t> {  // (kdC1khkw, n1)
    using Index = uint32_t;

    using Base = Coord<2, Index>;

    static constexpr int KDC1KHKW_INDEX = 0;
    static constexpr int N1_INDEX = 0;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv3dFracZ3dCoord() {}

    CATLASS_HOST_DEVICE
    Conv3dFracZ3dCoord(Index kdc1khkw, Index n1)
    : Base(MakeCoord(kdc1khkw, n1)) {}

    CATLASS_HOST_DEVICE
    Index const& kdc1khkw() const {
        return this->At(KDC1KHKW_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& kdc1khkw() {
        return this->At(KDC1KHKW_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const& n1() const {
        return this->At(N1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& n1() {
        return this->At(N1_INDEX);
    }
};
}

#endif  // CATLASS_CONV_COORD_HPP