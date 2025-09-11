import os

class Config:
    WRAPPER_CODE_PATH = "../wrapper"
    INCLUDE_PATH = "../../include"

    LAYOUT_TAG_SET = [0, 1]  # 0 is RowMajor, 1 is ColumnMajor
    LAYOUT_TAG_MAP = {0: "Catlass::layout::RowMajor", 1: "Catlass::layout::ColumnMajor"}

    DTYPE_MAP = {"half": 0, "float": 1}

    @staticmethod
    def get_tiling_key(
        kernel_serial, dtype, l_tag_a, l_tag_b, p_tag_a, p_tag_b, p_tag_c
    ):
        part1 = kernel_serial  # 56-63
        part2 = Config.DTYPE_MAP[dtype] << 4  # 48-55
        part3 = 0  # 40-47
        part4 = 0  # 32-39
        part5 = 0  # 24-31
        part6 = 0 | p_tag_c  # 16-23
        part7 = (p_tag_a << 4) | p_tag_b  # 8-15
        part8 = (l_tag_a << 4) | l_tag_b  # 0-7
        hex_str = f"0x{part1:02x}{part2:02x}{part3:02x}{part4:02x}{part5:02x}{part6:02x}{part7:02x}{part8:02x}"
        return hex_str
