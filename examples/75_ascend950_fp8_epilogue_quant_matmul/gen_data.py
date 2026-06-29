#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import argparse
import os
from typing import Optional

import numpy as np
import torch
from ml_dtypes import float8_e4m3fn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FP8_MAX = 448.0


def _resolve_workspace(data_root_cli: Optional[str]) -> str:
    if data_root_cli is not None:
        root = data_root_cli.strip()
        if root:
            return os.path.abspath(os.path.expanduser(root))
    return _SCRIPT_DIR


def gen_data(m: int, n: int, k: int, workspace: str) -> None:
    data_dir = os.path.join(workspace, "data")
    input_dir = os.path.join(data_dir, "input")
    golden_dir = os.path.join(data_dir, "golden")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    torch.manual_seed(0)
    a_fp32 = torch.randn((m, k), dtype=torch.float32) * 2
    b_fp32 = torch.randn((k, n), dtype=torch.float32) * 2

    a_max = a_fp32.abs().max(dim=1).values
    b_max = b_fp32.abs().max(dim=0).values
    a_max[a_max == 0] = 1.0
    b_max[b_max == 0] = 1.0

    a_scale_fp32 = a_max / _FP8_MAX
    b_scale_fp32 = b_max / _FP8_MAX

    a_fp8 = (a_fp32 / a_scale_fp32.view(-1, 1)).numpy().astype(float8_e4m3fn)
    b_fp8 = (b_fp32 / b_scale_fp32.view(1, -1)).numpy().astype(float8_e4m3fn)
    a_scale_fp8 = a_scale_fp32.numpy().astype(float8_e4m3fn)
    b_scale_fp8 = b_scale_fp32.numpy().astype(float8_e4m3fn)

    a_fp8.view(np.int8).tofile(os.path.join(input_dir, "a_8.bin"))
    b_fp8.view(np.int8).tofile(os.path.join(input_dir, "b_8.bin"))
    a_scale_fp8.view(np.int8).tofile(os.path.join(input_dir, "a_scale.bin"))
    b_scale_fp8.view(np.int8).tofile(os.path.join(input_dir, "b_scale.bin"))

    a_val = a_fp8.astype(np.float32)
    b_val = b_fp8.astype(np.float32)
    per_token = a_scale_fp8.astype(np.float32)
    per_channel = b_scale_fp8.astype(np.float32)

    c_fp32 = a_val @ b_val
    golden = c_fp32 * per_token[:, np.newaxis] * per_channel[np.newaxis, :]
    golden.astype(np.float32).tofile(os.path.join(golden_dir, "expected_data.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FP8 per-token/per-channel inputs and FP32 golden under "
                     "<data-root>/data/.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="DIR",
        help="Directory under which data/input and data/golden are created. "
             "Default: this script's directory.",
    )
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    args = parser.parse_args()
    workspace = _resolve_workspace(args.data_root)
    gen_data(args.m, args.n, args.k, workspace)
