# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import ctypes


AIC_CORE_NUM_ATLAS_A2_910B1 = 24
AIC_CORE_NUM_ATLAS_A2_910B2 = 24
AIC_CORE_NUM_ATLAS_A2_910B2C = 24
AIC_CORE_NUM_ATLAS_A2_910B3= 20
AIC_CORE_NUM_ATLAS_A2_910B4 = 20
AIC_CORE_NUM_ATLAS_A2_910B4_1 = 20


def get_ascendc_sync_base_addr(device_id : int):
    # C函数原型：rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len);
    runtime_lib = ctypes.CDLL("libruntime.so")
    ffts_addr = ctypes.c_uint64()
    ffts_len = ctypes.c_uint32()

    import acl
    acl.rt.set_device(device_id) # set_device must be called before rtGetC2cCtrlAddr
    ret = runtime_lib.rtGetC2cCtrlAddr(ctypes.byref(ffts_addr), ctypes.byref(ffts_len))
    if ret != 0:
        raise Exception("rtGetC2cCtrlAddr failed ret = {}".format(ret))
    return ffts_addr.value, ffts_len.value


def check_autotune_avalible():

    ERROR_STR = (
        "mskpp.autotune is not found in current CANN toolkit. The autotune feature "
        "is only supported in CANN toolkit 8.1.RC1.beta1 or higher version. "
        "The lower version needs to be upgraded."
    )

    import mskpp
    if not hasattr(mskpp, "autotune"):
        raise Exception(ERROR_STR)


def assert_kernel_run_one_time():
    '''
    This is used to prevent the kernel from being run multiple times due to known issue 1.

    Known issue 1: When a mskpp-defined kernel is run multiple times, the kernel input parameters always retain
                   the values of the first run.
    '''
    assert_kernel_run_one_time.call_count = getattr(assert_kernel_run_one_time, 'call_count', 0) + 1
    if assert_kernel_run_one_time.call_count > 1:
        raise RuntimeError("Kernel function can only be called once in current CANN toolkit")