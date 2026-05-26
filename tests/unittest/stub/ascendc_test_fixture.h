/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_TEST_FIXTURE_H
#define ASCENDC_STUB_TEST_FIXTURE_H

#include <gtest/gtest.h>
#include "stub/ascendc_logger.h"
#include "stub/kernel_tensor.h"
#include "stub/kernel_struct_mm.h"
#include "stub/kernel_struct_unary.h"
#include "stub/kernel_operator_data_copy_ext.h"
#include "stub/kernel_operator_vec_vconv_intf.h"

class AscendCTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASCENDC_CLEAR_LOGS();
    }

    void TearDown() override
    {
        ASCENDC_CLEAR_LOGS();
    }
};

#endif // ASCENDC_STUB_TEST_BASE_H
