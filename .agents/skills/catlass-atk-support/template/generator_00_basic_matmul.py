# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import random

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig

############### HELPER FUNCTION ###############

MEM_LIMIT = 10 * 1024 * 1024 * 1024

def check_memory_limit(m, k, n, dtype):
    """Estimate memory usage; shrink axes when exceeding limit."""
    items = m * k + k * n + m * n
    if dtype in ["fp32", "int32", "int"]:
        mem = items * 4
    elif dtype in ["bf16", "fp16", "int16"]:
        mem = items * 2
    elif dtype in ["int64", "fp64"]:
        mem = items * 8
    else:
        mem = items * 1

    if mem > MEM_LIMIT:
        axis = random.randint(0, 2)
        if axis == 0:
            m = max(int(m // 2), 1)
        elif axis == 1:
            k = max(int(k // 2), 1)
        else:
            n = max(int(n // 2), 1)
        m, k, n = check_memory_limit(m, k, n, dtype)

    return m, k, n

def assign_matmul_storage_shapes(
    case_config,
    m: int,
    k: int,
    n: int,
    *,
    inputs_name: str = "inputs",
    weights_name: str = "weights",
) -> None:
    """Set tensor storage shapes for logical ``(M,K) @ (K,N)`` matmul.

  ``torch_catlass`` derives logical ``M/K/N`` from storage dims plus
    ``transA``/``transB`` (see ``CatlassKernelWrapper::MatmulLike::GetKernelInfo``).
    """
    def _case_attr(case_config, name, default=False):
        for inp in case_config.inputs:
            if inp.name == name:
                return inp.range_values
        return default

    trans_a = _case_attr(case_config, "transA")
    trans_b = _case_attr(case_config, "transB")
    for inp in case_config.inputs:
        if inp.name == inputs_name:
            inp.shape = [k, m] if trans_a else [m, k]
        elif inp.name == weights_name:
            inp.shape = [n, k] if trans_b else [k, n]

################ TESTCASE GENERATOR #################


@GENERATOR_REGISTRY.register("generator_00_basic_matmul")
class MatmulGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        m = case_config.inputs[0].shape[0]
        k = case_config.inputs[0].shape[1]
        n = case_config.inputs[1].shape[1]
        attr2dtype = {"float16": "fp16", "bfloat16": "bf16"}
        items = case_config.inputs[2].range_values
        dd = random.choice(items)
        m, k, n = check_memory_limit(m, k, n, attr2dtype[dd])
        assign_matmul_storage_shapes(case_config, m, k, n)
        case_config.inputs[0].dtype = attr2dtype[dd]
        case_config.inputs[1].dtype = attr2dtype[dd]
        case_config.inputs[2].range_values = [dd]
        return case_config
