# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import ctypes


def get_ascendc_sync_base_addr():
    # C函数原型：rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len);
    runtime_lib = ctypes.CDLL("libruntime.so")
    ffts_addr = ctypes.c_uint64()
    ffts_len = ctypes.c_uint32()

    import acl
    acl.rt.set_device(0)
    ret = runtime_lib.rtGetC2cCtrlAddr(ctypes.byref(ffts_addr), ctypes.byref(ffts_len))
    if ret != 0:
        raise Exception("rtGetC2cCtrlAddr failed ret = {}".format(ret))
    return ffts_addr.value, ffts_len.value