# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass

from catlass.base_dsl.typing import Float32, Int32, Numeric, Pointer
from catlass.types import TlaCrossFlag, TlaFlag

# ---------------- 公共常量 ----------------
PRE_LAUNCH = 2
UB_S_OTMP_BUF_STAGES = 2
P_L1_BUF = 3
L0_STAGES = 2
VL_F32 = 64
VL_F16 = 128


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------- 参数打包 ----------------
@dataclass
class FaCtx:
    qHeads_: Int32
    kvHeads_: Int32
    maxQSeqlen_: Int32
    maxKvSeqlen_: Int32
    kvSeqlen_: Int32
    qFormat_: Int32
    kvFormat_: Int32
    scaleValue_: Float32
    mm1L0ATotalStages_: int
    mm1L0BTotalStages_: int
    mm2L0ATotalStages_: int
    mm2L0BTotalStages_: int
    firstBatchTaskNum_: Int32
    totalTaskNum_: Int32
    embed_: int
    embedV_: int
    qBaseTile_: int
    kvBaseTile_: int
    groupSize: Int32
    qNOffset: int
    kNOffset: int
    vNOffset: int
    oNOffset: int
    qSOffset: Int32
    kSOffset: Int32
    vSOffset: Int32
    oSOffset: Int32
    paged: bool = True
    max_blocks_per_batch: int = 0
    DTYPE_Q: type[Numeric] | None = None
    DTYPE_K: type[Numeric] | None = None
    DTYPE_P: type[Numeric] | None = None
    DTYPE_V: type[Numeric] | None = None


@dataclass
class FaTask:
    """task 级运行期标量；调用点按需构造部分字段，故保留 None 默认。"""

    curBatch: Int32 | None = None
    qHeadIdx: Int32 | None = None
    kvHeadIdx: Int32 | None = None
    qSTileIdx: Int32 | None = None
    qBOffset: Int32 | None = None
    kBOffset: Int32 | None = None
    vBOffset: Int32 | None = None
    oBOffset: Int32 | None = None
    curQSeqlen: Int32 | None = None
    curKvSeqlen: Int32 | None = None
    rowNum: Int32 | None = None
    gmOffsetV: Int32 | None = None


@dataclass
class FaBufs:
    l1Q_ptrs: list[Pointer]
    l1K_ptrs: list[Pointer]
    l1P_ptrs: list[Pointer]
    l1V_ptrs: list[Pointer]
    l0a_ptrs: list[Pointer]
    l0b_ptrs: list[Pointer]
    l0c_ptrs: list[Pointer]
    ubS_ptrs: list[Pointer]
    ubP_ptrs: list[Pointer]
    ubOTmp_ptrs: list[Pointer]
    ubO_ptr: Pointer
    ubO16_ptr: Pointer
    nowMax_ptr: Pointer
    expMax_ptrs: list[Pointer]
    nowSum_ptr: Pointer
    lastMax_ptr: Pointer
    lastSum_ptr: Pointer


@dataclass
class FaSync:
    q_l0a_data_ready: TlaFlag
    k_l0b_data_ready_0: TlaFlag
    k_l0b_data_ready_1: TlaFlag
    v_l0b_data_ready_0: TlaFlag
    v_l0b_data_ready_1: TlaFlag
    l0a_available_0: TlaFlag
    l0a_available_1: TlaFlag
    l0b_available_0: TlaFlag
    l0b_available_1: TlaFlag
    l0c_available_0: TlaFlag
    l0c_available_1: TlaFlag
    l0c_available_2: TlaFlag
    l0c_available_3: TlaFlag
    q_l1_data_ready: TlaFlag
    k_l1_data_ready_0: TlaFlag
    k_l1_data_ready_1: TlaFlag
    v_l1_data_ready_0: TlaFlag
    v_l1_data_ready_1: TlaFlag
    l0a_data_ready_0: TlaFlag
    l0a_data_ready_1: TlaFlag
    l0b_data_ready_0: TlaFlag
    l0b_data_ready_1: TlaFlag
    l0c_data_ready_0: TlaFlag
    l0c_data_ready_1: TlaFlag
    l0c_data_ready_2: TlaFlag
    l0c_data_ready_3: TlaFlag
    vec_ready_mte2_0: TlaFlag
    vec_ready_mte2_1: TlaFlag
    mte3_ready_mask_0: TlaFlag
    mte3_ready_mask_1: TlaFlag
    ub_p_available_0: TlaFlag
    ub_p_available_1: TlaFlag
    mte3_ready_rescale: TlaFlag
    ub_p_data_ready_0: TlaFlag
    ub_p_data_ready_1: TlaFlag
    ub_s_cross_ready_0: TlaCrossFlag
    ub_s_cross_ready_1: TlaCrossFlag
    ub_otmp_cross_ready_0: TlaCrossFlag
    ub_otmp_cross_ready_1: TlaCrossFlag
    ub_p_cross_ready_0: TlaCrossFlag
    ub_p_cross_ready_1: TlaCrossFlag
    ub_p_cross_ready_2: TlaCrossFlag
