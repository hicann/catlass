# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "code_generator.py"
REGISTER_PATTERN = re.compile(
    r"^void Register_(catlass_gemm_[^\(]+)\(Manifest &manifest\);$", re.MULTILINE
)


class CodeGeneratorKernelFilterTest(unittest.TestCase):
    def generate(self, kernels, expect_success=True):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as workspace:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--kernels",
                    kernels,
                    "--workspace-dir",
                    workspace,
                    "--arch",
                    "a2",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if not expect_success:
                return result
            self.assertEqual(result.returncode, 0, result.stderr)
            register_file = (
                Path(workspace) / "generated/gemm/register_all_gemm_operations.cpp"
            )
            return register_file.read_text() if register_file.exists() else ""

    def test_single_kernel_filter(self):
        names = REGISTER_PATTERN.findall(self.generate("00_basic_matmul"))
        self.assertTrue(names)
        self.assertTrue(all("00_basic_matmul" in name for name in names))

    def test_comma_delimited_kernel_filter(self):
        names = REGISTER_PATTERN.findall(
            self.generate("00_basic_matmul, 02_grouped_matmul_slice_m")
        )
        basic = [name for name in names if "00_basic_matmul" in name]
        grouped = [name for name in names if "02_grouped_matmul_slice_m" in name]
        self.assertTrue(basic)
        self.assertTrue(grouped)
        self.assertEqual(len(names), len(basic) + len(grouped))

    def test_unmatched_kernel_filter(self):
        self.assertEqual(self.generate("does_not_exist"), "")

    def test_empty_kernel_filter(self):
        result = self.generate("", expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(
            result.stderr,
            r"operations that will be generated in total: 1[0-9]{4,}",
        )


if __name__ == "__main__":
    unittest.main()
