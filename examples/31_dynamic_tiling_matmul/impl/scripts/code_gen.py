import os
from utils.config import Config

from templates.common_matmul_template import CommonMatmulTemplate
from templates.launch_map_template import LaunchMapTemplate

if __name__ == "__main__":

    kernel_info = {}

    os.makedirs(Config.WRAPPER_CODE_PATH, exist_ok=True)
    CommonMatmulTemplate.gen_code("CommonMatmulKernel", "common_matmul_kernel", 0, "half", kernel_info)
    CommonMatmulTemplate.gen_code("CommonMatmulKernel", "common_matmul_kernel", 0, "float", kernel_info)
    LaunchMapTemplate.gen_code(kernel_info)