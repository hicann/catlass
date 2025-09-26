# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch_npu
from loguru import logger
from torch_npu.npu._format import Format

from catlass_test.common import (
    BishengDType,
    OpType,
    get_current_stream_ptr,
    is_transposed,
    torch_dtype_to_bisheng_dtype,
)
from catlass_test.compiler import TemplateCompiler


@lru_cache
def load_kernel_lib(kernel_path: str) -> Optional[ctypes.CDLL]:
    return ctypes.cdll.LoadLibrary(kernel_path)


class AdapterBase(ABC):
    def __init__(
        self,
        kernel_src_file: str,
        input_tensors: Dict[str, torch.Tensor],
        output_tensors: Dict[str, torch.Tensor] = {},
        attrs: Dict[str, Any] = {},
        op_type: OpType = OpType.MIX_AIC_1_2,
    ):
        self.kernel_src_file = kernel_src_file
        self.attrs = attrs
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        if self.output_tensors == {}:
            self.output_tensors = self.get_output_tensors()
        self.template_compiler = TemplateCompiler(kernel_src_file)
        self.op_type = op_type

    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        return self.input_tensors.get(
            tensor_name, self.output_tensors.get(tensor_name, torch.tensor(1))
        )

    def get_tensor_name_list(self) -> List[str]:
        return list(self.input_tensors.keys()) + list(self.output_tensors.keys())

    def get_tensor_ptr(self, tensor_name: str) -> ctypes.c_void_p:
        tensor = self.get_tensor(tensor_name)
        return ctypes.c_void_p(tensor.data_ptr() if tensor is not None else 0)

    def get_tensor_dtype(self, tensor_name: str) -> torch.dtype:
        return self.get_tensor(tensor_name).dtype

    def get_tensor_dtype_bisheng(self, tensor_name: str) -> BishengDType:
        return torch_dtype_to_bisheng_dtype(self.get_tensor_dtype(tensor_name))

    def get_tensor_layout(self, tensor_name: str) -> str:
        tensor = self.get_tensor(tensor_name)
        if len(tensor.shape) == 1:
            return "layout::VectorLayout"
        npu_format = torch_npu.get_npu_format(self.get_tensor(tensor_name))
        if npu_format == Format.ND:
            return (
                "layout::ColumnMajor"
                if self.attrs.get(f"Trans{tensor_name}", False) or is_transposed(tensor)
                else "layout::RowMajor"
            )
        elif npu_format == Format.FRACTAL_NZ:
            return ""
        return ""

    def __get_compile_params(self) -> Dict[str, str]:
        compile_params = {}
        template_compile_params = self.template_compiler.compile_params
        element_pattern: re.Pattern[str] = re.compile(r"Element([A-Za-z0-9_]+)")
        layout_pattern: re.Pattern[str] = re.compile(r"Layout([A-Za-z0-9_]+)")
        for template_compile_param in template_compile_params.keys():
            if var_name := element_pattern.search(template_compile_param):
                compile_params[template_compile_param] = self.get_tensor_dtype_bisheng(
                    var_name.group(1)
                )
            elif var_name := layout_pattern.search(template_compile_param):
                compile_params[template_compile_param] = self.get_tensor_layout(
                    var_name.group(1)
                )
            else:
                compile_params[template_compile_param] = self.get_compile_params(
                    template_compile_param
                )
        return compile_params

    def __get_runtime_params(self) -> List[Any]:
        runtime_params = []
        template_runtime_params = self.template_compiler.runtime_params
        ptr_pattern = re.compile(r"device([A-Za-z0-9_]+)")
        layout_pattern = re.compile(r"layout([A-Za-z0-9_]+)")
        for template_runtime_param in template_runtime_params.keys():
            if "stream" in template_runtime_param:
                runtime_params.append(get_current_stream_ptr())
            elif (var_name := ptr_pattern.search(template_runtime_param)) is not None:
                runtime_params.append(self.get_tensor_ptr(var_name.group(1)))
            elif (
                var_name := layout_pattern.search(template_runtime_param)
            ) is not None:
                logger.error("Do not support layout")
            else:
                runtime_params.append(self.get_runtime_params(template_runtime_param))
        return runtime_params

    def get_kernel(self) -> ctypes.CDLL:
        kernel_path = self.template_compiler.compile(
            self.__get_compile_params(), self.op_type
        )
        kernel_dll = load_kernel_lib(kernel_path)
        return kernel_dll

    def run(self):
        """执行用例"""
        logger.info(f"kernel template is {self.kernel_src_file}.")
        params = self.__get_runtime_params()
        logger.info(",".join([str(type(param).__name__) for param in params]))
        kernel = self.get_kernel()
        torch.npu.synchronize()
        kernel.run(*params)
        torch.npu.synchronize()

    def get_output_tensors(self) -> Dict[str, torch.Tensor]:
        output_shapes = self.get_output_shapes()
        output_dtypes = self.get_output_dtypes()
        output_tensors = {}
        output_names = set(output_shapes.keys()).intersection(set(output_dtypes.keys()))
        for output_name in output_names:
            output_tensors[output_name] = torch.zeros(
                output_shapes[output_name], dtype=output_dtypes[output_name]
            ).npu()

        return output_tensors

    def get_compile_params(self, param_name: str):
        pass

    def get_runtime_params(self, param_name: str):
        pass

    # abstract methods

    @abstractmethod
    def get_output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        pass

    @abstractmethod
    def get_output_dtypes(self) -> Dict[str, torch.dtype]:
        pass

    @abstractmethod
    def get_problem_shape(self) -> Any:
        pass

    @abstractmethod
    def get_problem_count(self) -> Any:
        pass
