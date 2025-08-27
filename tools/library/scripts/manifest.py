# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import shutil
import logging
import gemm_operation

LOGGER = logging.getLogger(__name__)


class OperationRegistry:
    register_functions = {}

    @classmethod
    def register(cls, name):
        def decorator(func):
            cls.register_functions[name] = func
            return func
        return decorator


class Manifest:

    def __init__(self, args):
        self.args = args
        self.operations = []
        self.operations_dict = {}
        self.enable_filter_out = True
        self.filtered_kernels = [args.kernels]

        self.target_generator = {
            'gemm': gemm_operation.GemmOperationGenerator
        }
        for _, func in OperationRegistry.register_functions.items():
            func(self)
        LOGGER.info(f'operations that will be generated in total: {len(self.operations)}')

        if len(self.operations) > 10000:
            raise Exception(
                'Due to limits of bisheng compiler, compiling more than 10,000 operations are not guaranteed'
            )

        self.register_all_operations_template = """
#include "catlass/library/library.h"
#include "catlass/library/manifest.h"

namespace Catlass {{
namespace Library {{

{api_decl_src}

void RegisterAllKernels(Manifest &manifest)
{{
    {api_call_src}
}}

}}
}}
"""

        self.function_decl_template = """void Register_{kernel_name}(Manifest &manifest);\n"""
        self.function_call_template = """    Register_{kernel_name}(manifest);\n"""

        self.register_template = """
#include "catlass/library/library.h"
#include "catlass/library/manifest.h"

namespace Catlass {{
namespace Library {{

{function_decls}

void RegisterCatlass{operation_type}Operations(Manifest &manifest)
{{
{function_calls}
}}

}}
}}
"""

    def append(self, operation):
        if self.filter_out(operation):
            return
        self.operations.append(operation)
        if operation.operation_type not in self.operations_dict.keys():
            self.operations_dict[operation.operation_type] = {}
        if operation.get_name() not in self.operations_dict[operation.operation_type].keys():
            self.operations_dict[operation.operation_type][operation.get_name()] = {}

        self.operations_dict[operation.operation_type][operation.get_name()] = operation

    def filter_out(self, operation):
        if not self.enable_filter_out:
            return False
        operation_name = operation.get_name()
        for kernel_name in self.filtered_kernels:
            if kernel_name in operation_name:
                return False
        return True

    def generate_code(self):
        workspace_dir = self.args.workspace_dir
        generated_dir = os.path.join(workspace_dir, 'generated')

        LOGGER.debug(f'generated_dir={generated_dir}')

        if os.path.exists(generated_dir) and not os.path.islink(generated_dir):
            shutil.rmtree(generated_dir)
        elif os.path.islink(generated_dir):
            raise PermissionError(
                f'generated directory {generated_dir} is a soft link, which is not allowed to be removed.'
            )
        else:
            pass

        os.mkdir(generated_dir)
        api_decl_src = []
        api_call_src = []
        for operation_type, names in self.operations_dict.items():
            api_decl_src.append('void RegisterCatlass{}Operations(Manifest &manifest);'.format(operation_type))
            api_call_src.append('  RegisterCatlass{}Operations(manifest);'.format(operation_type))

            # save kernel names of this operation type in here
            kernel_names = []

            # create subfolder for each type of operations
            operation_subdir = os.path.join(generated_dir, operation_type)
            if not os.path.exists(operation_subdir):
                os.mkdir(operation_subdir) # e.g. create generated/gemm

            with self.target_generator[operation_type](operation_type, generated_dir) as generator:
                for name, operation in names.items():
                    LOGGER.info(f'generating kernel: {name}')
                    kernel_names.append(name)
                    generator.gen(name, operation) # generate kernel instance

            function_calls = ''
            function_decls = ''
            for kernel_name in kernel_names:
                function_calls += self.function_call_template.format(kernel_name=kernel_name)
                function_decls += self.function_decl_template.format(kernel_name=kernel_name)
            operation_register_src = self.register_template.format(
                operation_type=operation_type,
                function_calls=function_calls,
                function_decls=function_decls
            )

            path = f'register_all_{operation_type}_operations.cpp'
            with open(os.path.join(operation_subdir, path), "w") as f:
                # e.g. create generated/gemm/register_all_gemm_operations.cpp
                f.write(operation_register_src)

        register_all_kernels_src = self.register_all_operations_template.format(
            api_decl_src='\n'.join(api_decl_src), api_call_src='\n'.join(api_call_src)
        )

        with open(os.path.join(generated_dir, 'register_all_kernels_generated.cpp'), "w") as register_all_file:
            register_all_file.write(register_all_kernels_src)
