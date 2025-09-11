/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 * Copyright (c) 2016-     Facebook, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INT4_H
#define OP_API_INT4_H

#include <cmath>
#include <iostream>

namespace op {

// see framework/int4.h for description.
struct int4 {
    public:
    int4() : value_(0) {}  

    int4(int8_t val) {
        if (val < -8 || val > 7) {
            throw std::out_of_range("Value out of int4 range");
        }
        value_ = val;
    }

    // overload static_cast<int8_t>
    operator int8_t() const {
        return value_;
    }

    // overload static_cast<int32_t>
    operator int32_t() const {
        return static_cast<int>(value_);
    }

    static double size_of() {
        return 0.5;
    }

    int8_t get_value() const {
        return value_;
    }

    // overload op << for print
    friend std::ostream& operator<<(std::ostream& os, const int4& int4) {
        os << static_cast<int>(int4.value_);
        return os;
    }

private:
    int8_t value_;
    
};


} // namespace std


#endif // OP_API_INT4_H
