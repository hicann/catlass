/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_COORD_HPP
#define ACOT_COORD_HPP

#include "acot/acot.hpp"

namespace acot
{

    /// Statically-sized array specifying Coords within a tensor, 定义一个tensor的坐标
    template <
        int RANK_,                 ///< Logical rank of coordinate, 坐标的逻辑维度
        class Index_ = uint32_t,   ///< Index type used for each dimension, 用于每个维度的索引类型，默认为uint32_t 
        class LongIndex_ = int64_t ///< Long index type used for linear offsets, 表示线性偏移的长整型，默认为int64_t 
        >
    struct Coord
    {
    public:
        // Number of elements in Coord, 储存坐标的维度数量
        static const int RANK = RANK_;

        // Index typen used to store elements， 维度的索引类型
        using Index = Index_;

        // Type used to represent linear offsets, 线性偏移的类型
        using LongIndex = LongIndex_;

        // Default ctor initializes uniformly
        // 接受一个Index类型的值value，将该值赋值给所有维度的坐标，默认为0
        ACOT_HOST_DEVICE
        explicit Coord(Index value = Index(0))
        {
            for (int i = 0; i < RANK; ++i)
            {
                idx[i] = value;
            }
        }

        // Constructs from an array of integers
        // 接受一个数组(大小为RANK), 并将元素复制到idx数组中
        ACOT_HOST_DEVICE
        Coord(Index const (&idx_)[RANK])
        {
            for (int i = 0; i < RANK; ++i)
            {
                idx[i] = idx_[i];
            }
        }

        // Constructs from an array of integers
        // 返回idx数组中最小值所在的索引
        ACOT_HOST_DEVICE
        int Argmin() const
        {
            int i = 0;
            for (int j = 1; j < RANK; ++j)
            {
                if (idx[j] < idx[i])
                {
                    i = j;
                }
            }
            return i;
        }

        // Returns the index of the dimension with greatest value
        // 返回 idx 数组中最大值所在的索引
        ACOT_HOST_DEVICE
        int Argmax() const
        {
            int i = 0;
            for (int j = 1; j < RANK; ++j)
            {
                if (idx[j] > idx[i])
                {
                    i = j;
                }
            }
            return i;
        }

        // Returns true if Coord is non-zero
        // 显式转换操作符，将 Coord 转换为 bool。如果 idx 数组中有任一维度非零，则返回 true，否则返回 false。
        ACOT_HOST_DEVICE
        explicit operator bool() const
        {
            for (int i = 0; i < RANK; ++i)
            {
                if (idx[i])
                {
                    return true;
                }
            }
            return false;
        }

        // Return true if Coord is uniformly zero.
        // 逻辑非操作符重载。若 idx 数组中所有元素均为零，则返回 true。
        ACOT_HOST_DEVICE
        bool operator!() const
        {
            for (int i = 0; i < RANK; ++i)
            {
                if (idx[i])
                {
                    return false;
                }
            }
            return true;
        }

        // Element-wise addition
        // 重载+， 执行 Coord 对象的元素级加法
        ACOT_HOST_DEVICE
        Coord operator+(Coord const &b) const
        {
            Coord c;
            for (int i = 0; i < RANK; ++i)
            {
                c.idx[i] = idx[i] + b.idx[i];
            }
            return c;
        }

        // Add a scalar to each element
        // + 操作符的另一个重载，允许将一个标量值加到每个坐标元素上
        ACOT_HOST_DEVICE
        Coord operator+(const Index val) const
        {
            Coord c;
            for (int i = 0; i < RANK; ++i)
            {
                c.idx[i] = idx[i] + val;
            }
            return c;
        }

        // Element-wise subtraction
        // Coord 对象的元素级减法。
        ACOT_HOST_DEVICE
        Coord operator-(Coord const &b) const
        {
            Coord c;
            for (int i = 0; i < RANK; i++)
            {
                c.idx[i] = idx[i] - b.idx[i];
            }
            return c;
        }

        // Subtract a scalar from each element
        // 允许从每个坐标元素中减去一个标量值。
        ACOT_HOST_DEVICE
        Coord operator-(Index const val) const
        {
            Coord c;
            for (int i = 0; i < RANK; ++i)
            {
                c.idx[i] = idx[i] - val;
            }
            return c;
        }

        // Element-wise multiply
        // 执行 Coord 对象的元素级乘法
        ACOT_HOST_DEVICE
        Coord operator*(Coord const &b) const
        {
            Coord c;
            for (int i = 0; i < RANK; i++)
            {
                c.idx[i] = idx[i] * b.idx[i];
            }
            return c;
        }

        // Element-wise division
        // 执行 Coord 对象的元素级除法。
        ACOT_HOST_DEVICE
        Coord operator/(Coord const &b) const
        {
            Coord c;
            for (int i = 0; i < RANK; i++)
            {
                c.idx[i] = idx[i] / b.idx[i];
            }
            return c;
        }

        // Element-wise mod
        // 执行 Coord 对象的元素级取余操作。
        ACOT_HOST_DEVICE
        Coord operator%(Coord const &b) const
        {
            Coord c;
            for (int i = 0; i < RANK; i++)
            {
                c.idx[i] = idx[i] % b.idx[i];
            }
            return c;
        }

        // In-place addition
        //+=, 将另一个 Coord 对象的值加到当前对象上
        ACOT_HOST_DEVICE
        Coord &operator+=(Coord const &b)
        {
            for (int i = 0; i < RANK; ++i)
            {
                idx[i] += b.idx[i];
            }
            return *this;
        }

        // In-place equal
        // 判断两个 Coord 对象是否相等。
        ACOT_HOST_DEVICE
        bool operator==(Coord const &b) const
        {
            for (int i = 0; i < RANK; ++i)
            {
                if (idx[i] != b.idx[i])
                {
                    return false;
                }
            }
            return true;
        }

        // In-place equal
        // == 操作符的重载，判断一个 Coord 对象是否所有元素都等于某个标量值。
        ACOT_HOST_DEVICE
        bool operator==(Index const val) const
        {
            for (int i = 0; i < RANK; ++i)
            {
                if (idx[i] != val)
                {
                    return false;
                }
            }
            return true;
        }

        // Member acces operator
        // 下标操作符，返回指定维度的索引（非常量版本）
        ACOT_HOST_DEVICE
        Index &operator[](int dim)
        {
            return idx[dim];
        }

        // Member access operator
        // 下标操作符，返回指定维度的索引（常量版本）。
        ACOT_HOST_DEVICE
        Index const &operator[](int dim) const
        {
            return idx[dim];
        }

        // Gets the index of a given Coord element
        // At 函数，返回指定维度的元素，模板参数 DIM 指定维度
        template <int DIM>
        ACOT_HOST_DEVICE 
        Index &At()
        {
            return idx[DIM];
        }

        // Access via index; may limit unrolling potential
        // 另一个版本的 At，通过索引返回指定维度的元素。
        ACOT_HOST_DEVICE
        Index &At(int dim)
        {
            return idx[dim];
        }

        // Gets the index of a given Coord element 
        template <int DIM>
        ACOT_HOST_DEVICE
            Index const &
            At() const
        {
            return idx[DIM];
        }

        // Access via index; may limit unrolling potential
        ACOT_HOST_DEVICE
        Index const &At(int dim) const
        {
            return idx[dim];
        }

        // 通过传递一系列维度索引（如 0, 1, 2）来获取指定维度的子坐标
        template <int... Is>
        ACOT_HOST_DEVICE auto GetCoordByAxis() const
        {
            return Coord<sizeof...(Is), Index, LongIndex>{{idx[Is]...}};
        }

        // 返回两个 Coord 对象中每个维度的最小值
        ACOT_HOST_DEVICE
        static Coord Min(Coord const &a, Coord const &b)
        {
            Coord res;
            for (int i = 0; i < RANK; ++i)
            {
                res[i] = a[i] < b[i] ? a[i] : b[i];
            }
            return res;
        }

    private:
        // Indices
        Index idx[RANK];
    };

    // 下面是辅助函数

    // Helper to make a 1-element coordinate
    // 
    template <class T>
    ACOT_HOST_DEVICE
        Coord<1, T>
        MakeCoord(T dim0)
    {
        T values[1] = {dim0};
        return Coord<1, T>(values);
    }

    /// Helper to make a 2-element coordinate
    //主要用的是这个函数，2维tensor
    template <class T>
    ACOT_HOST_DEVICE
        Coord<2, T>
        MakeCoord(T dim0, T dim1)
    {
        T values[2] = {dim0, dim1};
        return Coord<2, T>(values);
    }

    /// Helper to make a 3-element coordinate
    template <class T>
    ACOT_HOST_DEVICE
        Coord<3, T>
        MakeCoord(T dim0, T dim1, T dim2)
    {
        T values[3] = {dim0, dim1, dim2};
        return Coord<3, T>(values);
    }

    /// Helper to make a 4-element coordinate
    template <class T>
    ACOT_HOST_DEVICE
        Coord<4, T>
        MakeCoord(T dim0, T dim1, T dim2, T dim3)
    {
        T values[4] = {dim0, dim1, dim2, dim3};
        return Coord<4, T>(values);
    }

} // namespace acot

#endif // ACOT_COORD_HPP