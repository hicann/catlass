# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
from itertools import product
from dataclasses import dataclass

import library
from gemm_operation import GemmOperation
from manifest import OperationRegistry

LOGGER = logging.getLogger(__name__)


################## basic_matmul ##################
@OperationRegistry.register('basic_matmul')
def register_gemm_basic_matmul_operation(manifest):

    layouts = [
        [library.LayoutType.RowMajor, library.LayoutType.RowMajor, library.LayoutType.RowMajor],
    ]

    data_types = [
        [library.DataType.fp16, library.DataType.fp16, library.DataType.fp16]
    ]

    # 设定L1/L0TileShape的搜索范围、搜索步长、减枝函数，生成范围内全量搜索结点
    tile_shapes = list(generate_tile_shapes(
        tile_shape_constraint_for_pingpong, # 自定义减枝函数
        element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
        stages=(2),
        step=16,
        tile_shape_range=TileShapeRange(
            l1_tile_m_range=(32, 128),
            l1_tile_n_range=(128, 256),
            l1_tile_k_range=(128, 256),
            l0_tile_m_range=(32, 128),
            l0_tile_n_range=(128, 256),
            l0_tile_k_range=(32, 64)
        )
    ))
    LOGGER.info(f'basic_matmul tile_shapes size={len(tile_shapes)}')

    block_swizzle_descriptions = [
        'Gemm::Block::GemmIdentityBlockSwizzle<3, 1>',
        # 可自定义其他Swizzle参数
    ]

    # 正交tiling参数组合
    for layout, data_type, tile_shape, block_swizzle in product(
        layouts,
        data_types,
        tile_shapes,
        block_swizzle_descriptions
    ):
        l1_tile_shape, l0_tile_shape = tile_shape
        tensor_a = library.GemmTypeDescription(data_type[0], layout[0])
        tensor_b = library.GemmTypeDescription(data_type[1], layout[1])
        tensor_c = library.GemmTypeDescription(data_type[2], layout[2])
        op = GemmOperation(
            operation_type='gemm',
            kernel_type='basic_matmul',
            device_api='Gemm::Device::DeviceGemm',
            kernel_api='Gemm::Kernel::BasicMatmul',
            block_api='Gemm::Block::BlockMmad',
            dispatch='Gemm::MmadAtlasA2Pingpong<true>',
            l1_tile_shape=l1_tile_shape,
            l0_tile_shape=l0_tile_shape,
            a_type=tensor_a,
            b_type=tensor_b,
            c_type=tensor_c,
            block_swizzle=block_swizzle,
            epilogue='void',
            custom_headers='#include "catlass/gemm/kernel/basic_matmul.hpp"',
            cpp_instance='BasicMatmulGemmOperation'
        )
        manifest.append(op)
################## basic_matmul end ##################


################## grouped_matmul ##################
@OperationRegistry.register('grouped_matmul')
def register_gemm_grouped_matmul_operation(manifest):

    layouts = [
        [library.LayoutType.RowMajor, library.LayoutType.RowMajor, library.LayoutType.RowMajor],
    ]
    data_types = [
        [library.DataType.fp16, library.DataType.fp16, library.DataType.fp16],
    ]
    block_swizzle_descriptions = [
        'Gemm::Block::GemmIdentityBlockSwizzle<3, 1>',
    ]

    # generate L1/L0TileShape search space
    tile_shapes = list(generate_tile_shapes(
        tile_shape_constraint_for_preload_async, # 自定义减枝函数
        element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
        stages=(1, 2, 4, 2, 1), # Preload/L1/L0A/L0B/L0C stages
        step=16,
        tile_shape_range=TileShapeRange(
            l1_tile_m_range=(128, 256),
            l1_tile_n_range=(128, 256),
            l1_tile_k_range=(128, 256),
            l0_tile_m_range=(128, 256),
            l0_tile_n_range=(128, 256),
            l0_tile_k_range=(32, 32)
        )
    ))
    LOGGER.info(f'grouped_matmul tile_shapes size={len(tile_shapes)}')

    # 正交tiling参数组合
    for layout, data_type, tile_shape, block_swizzle in product(
        layouts,
        data_types,
        tile_shapes,
        block_swizzle_descriptions
    ):
        l1_tile_shape, l0_tile_shape = tile_shape
        tensor_a = library.GemmTypeDescription(data_type[0], layout[0])
        tensor_b = library.GemmTypeDescription(data_type[1], layout[1])
        tensor_c = library.GemmTypeDescription(data_type[2], layout[2])
        op = GemmOperation(
            operation_type='gemm',
            kernel_type='grouped_matmul',
            device_api='Gemm::Device::DeviceGemm',
            kernel_api='Gemm::Kernel::GroupedMatmul',
            block_api='Gemm::Block::BlockMmad',
            dispatch='Gemm::MmadAtlasA2PreloadAsync<1,2,4,2,1,true,true>',
            l1_tile_shape=l1_tile_shape,
            l0_tile_shape=l0_tile_shape,
            a_type=tensor_a,
            b_type=tensor_b,
            c_type=tensor_c,
            block_swizzle=block_swizzle,
            epilogue='void',
            custom_headers='#include "catlass/gemm/kernel/grouped_matmul.hpp"',
            cpp_instance='GroupedMatmulGemmOperation'
        )
        manifest.append(op)
################## grouped_matmul end ##################


################## grouped_matmul_slice_m ##################
@OperationRegistry.register('grouped_matmul_slice_m')
def register_gemm_grouped_matmul_slice_m_operation(manifest):

    layouts = [
        [library.LayoutType.RowMajor, library.LayoutType.RowMajor, library.LayoutType.RowMajor],
    ]
    data_types = [
        [library.DataType.fp16, library.DataType.fp16, library.DataType.fp16],
    ]
    block_swizzle_descriptions = [
        'Gemm::Block::GemmIdentityBlockSwizzle<3, 1>',
    ]

    # generate L1/L0TileShape search space
    tile_shapes = list(generate_tile_shapes(
        tile_shape_constraint_for_preload_async, # 自定义减枝函数
        element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
        stages=(1, 2, 4, 2, 1), # Preload/L1/L0A/L0B/L0C stages
        step=16,
        tile_shape_range=TileShapeRange(
            l1_tile_m_range=(128, 256),
            l1_tile_n_range=(128, 256),
            l1_tile_k_range=(128, 256),
            l0_tile_m_range=(128, 256),
            l0_tile_n_range=(128, 256),
            l0_tile_k_range=(64, 64)
        )
    ))
    LOGGER.info(f'grouped_matmul_slice_m tile_shapes size={len(tile_shapes)}')

    # 正交tiling参数组合
    for layout, data_type, tile_shape, block_swizzle in product(
        layouts,
        data_types,
        tile_shapes,
        block_swizzle_descriptions
    ):
        l1_tile_shape, l0_tile_shape = tile_shape
        tensor_a = library.GemmTypeDescription(data_type[0], layout[0])
        tensor_b = library.GemmTypeDescription(data_type[1], layout[1])
        tensor_c = library.GemmTypeDescription(data_type[2], layout[2])
        op = GemmOperation(
            operation_type='gemm',
            kernel_type='grouped_matmul_slice_m',
            device_api='Gemm::Device::DeviceGemm',
            kernel_api='Gemm::Kernel::GroupedMatmulSliceM',
            block_api='Gemm::Block::BlockMmad',
            dispatch='Gemm::MmadAtlasA2PreloadAsync<1,2,4,2,1,true,true>',
            l1_tile_shape=l1_tile_shape,
            l0_tile_shape=l0_tile_shape,
            a_type=tensor_a,
            b_type=tensor_b,
            c_type=tensor_c,
            block_swizzle=block_swizzle,
            epilogue='void',
            cpp_instance='GroupedMatmulSliceMGemmOperation',
            custom_headers='#include "catlass/gemm/kernel/grouped_matmul_slice_m.hpp"',
            custom_args={'element_group_list': 'int64_t'}
        )
        manifest.append(op)
################## grouped_matmul_slice_m end ##################


################## optimized_matmul_padding_ab ##################
OPTIMIZED_MATMUL_TILE_COPY_DECLARATION = """
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void
>
struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
    using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType>;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementAccumulator = typename Base::ElementAccumulator;

    // When matrix A is row-major, if the number of rows in matrix A is less than 16, 
    // using the CopyGmToL1IntervalDataCopy method can improve the transfer efficiency.
    // The situation is similar for matrix B. If the above conditions are met, 
    // please uncomment the following and comment out the original matrix A transfer method

    // using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;

    using CopyGmToL1A = typename Base::CopyGmToL1A;
    using CopyGmToL1B = typename Base::CopyGmToL1B;

    using CopyL1ToL0A = typename Base::CopyL1ToL0A;
    using CopyL1ToL0B = typename Base::CopyL1ToL0B;

    using CopyL0CToGm = typename Base::CopyL0CToGm; 
    using BiasTypeSelector = typename Base::BiasTypeSelector; 
    using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
    using CopyL1ToBT = typename Base::CopyL1ToBT;
};
"""


@OperationRegistry.register('optimized_matmul_padding_ab')
def register_gemm_optimized_matmul_padding_ab_operation(manifest):

    layouts = [
        [library.LayoutType.RowMajor, library.LayoutType.RowMajor, library.LayoutType.RowMajor],
    ]
    data_types = [
        [library.DataType.fp16, library.DataType.fp16, library.DataType.fp16],
    ]
    block_swizzle_descriptions = [
        'Gemm::Block::GemmIdentityBlockSwizzle<3, 0>',
    ]

    layout_padding = {
        library.LayoutType.RowMajor: 'layout::PaddingRowMajor',
        library.LayoutType.ColumnMajor: 'layout::PaddingColumnMajor',
    }

    # generate L1/L0TileShape search space
    tile_shapes = list(generate_tile_shapes(
        tile_shape_constraint_for_pingpong, # 自定义减枝函数
        element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
        stages=(2),
        step=16,
        tile_shape_range=TileShapeRange(
            l1_tile_m_range=(32, 128),
            l1_tile_n_range=(128, 256),
            l1_tile_k_range=(128, 256),
            l0_tile_m_range=(32, 128),
            l0_tile_n_range=(128, 256),
            l0_tile_k_range=(32, 64)
        )
    ))
    LOGGER.info(f'optimized_matmul_padding_ab tile_shapes size={len(tile_shapes)}')

    # 正交tiling参数组合
    for layout, data_type, tile_shape, block_swizzle in product(
        layouts,
        data_types,
        tile_shapes,
        block_swizzle_descriptions
    ):
        l1_tile_shape, l0_tile_shape = tile_shape
        tensor_a = library.GemmTypeDescription(data_type[0], layout[0])
        tensor_b = library.GemmTypeDescription(data_type[1], layout[1])
        tensor_c = library.GemmTypeDescription(data_type[2], layout[2])
        op = GemmOperation(
            operation_type='gemm',
            kernel_type='optimized_matmul_padding_ab',
            device_api='Gemm::Device::DeviceGemm',
            kernel_api='Gemm::Kernel::OptimizedMatmul',
            block_api='Gemm::Block::BlockMmad',
            dispatch='Gemm::MmadAtlasA2Preload<true,true>',
            l1_tile_shape=l1_tile_shape,
            l0_tile_shape=l0_tile_shape,
            a_type=tensor_a,
            b_type=tensor_b,
            c_type=tensor_c,
            block_swizzle=block_swizzle,
            epilogue='void',
            cpp_instance='OptimizedMatmulGemmOperation',
            custom_headers='#include "catlass/gemm/kernel/optimized_matmul.hpp"',
            custom_common_decls=OPTIMIZED_MATMUL_TILE_COPY_DECLARATION,
            custom_args={
                'layout_padding_a': layout_padding.get(layout[0], 'unknown_layout'),
                'layout_padding_b': layout_padding.get(layout[1], 'unknown_layout'),
                'compute_length_a': '49152', # 96 * 1024 / sizeof(half)
                'compute_length_b': '49152', # 96 * 1024 / sizeof(half)
            }
        )
        manifest.append(op)
################## optimized_matmul_padding_ab end ##################


################## optimized_matmul_padding_a_only ##################
@OperationRegistry.register('optimized_matmul_padding_a_only')
def register_gemm_optimized_matmul_padding_a_only_operation(manifest):

    layouts = [
        [library.LayoutType.RowMajor, library.LayoutType.RowMajor, library.LayoutType.RowMajor],
    ]
    data_types = [
        [library.DataType.fp16, library.DataType.fp16, library.DataType.fp16],
    ]
    block_swizzle_descriptions = [
        'Gemm::Block::GemmIdentityBlockSwizzle<3, 0>',
    ]

    layout_padding = {
        library.LayoutType.RowMajor: 'layout::PaddingRowMajor',
        library.LayoutType.ColumnMajor: 'layout::PaddingColumnMajor',
    }

    # generate L1/L0TileShape search space
    tile_shapes = list(generate_tile_shapes(
        tile_shape_constraint_for_pingpong, # 自定义减枝函数
        element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
        stages=(2),
        step=16,
        tile_shape_range=TileShapeRange(
            l1_tile_m_range=(32, 128),
            l1_tile_n_range=(128, 256),
            l1_tile_k_range=(128, 256),
            l0_tile_m_range=(32, 128),
            l0_tile_n_range=(128, 256),
            l0_tile_k_range=(32, 64)
        )
    ))
    LOGGER.info(f'optimized_matmul_padding_a_only tile_shapes size={len(tile_shapes)}')

    # 正交tiling参数组合
    for layout, data_type, tile_shape, block_swizzle in product(
        layouts,
        data_types,
        tile_shapes,
        block_swizzle_descriptions
    ):
        l1_tile_shape, l0_tile_shape = tile_shape
        tensor_a = library.GemmTypeDescription(data_type[0], layout[0])
        tensor_b = library.GemmTypeDescription(data_type[1], layout[1])
        tensor_c = library.GemmTypeDescription(data_type[2], layout[2])
        op = GemmOperation(
            operation_type='gemm',
            kernel_type='optimized_matmul_padding_a_only',
            device_api='Gemm::Device::DeviceGemm',
            kernel_api='Gemm::Kernel::OptimizedMatmul',
            block_api='Gemm::Block::BlockMmad',
            dispatch='Gemm::MmadAtlasA2Preload<true,true>',
            l1_tile_shape=l1_tile_shape,
            l0_tile_shape=l0_tile_shape,
            a_type=tensor_a,
            b_type=tensor_b,
            c_type=tensor_c,
            block_swizzle=block_swizzle,
            epilogue='void',
            cpp_instance='OptimizedMatmulGemmOperation',
            custom_headers='#include "catlass/gemm/kernel/optimized_matmul.hpp"',
            custom_common_decls=OPTIMIZED_MATMUL_TILE_COPY_DECLARATION,
            custom_args={
                'layout_padding_a': layout_padding.get(layout[0], 'unknown_layout'),
                'compute_length_a': '49152', # 96 * 1024 / sizeof(half)
            }
        )
        manifest.append(op)
################## optimized_matmul_padding_a_only end ##################


################## optimized_matmul_padding_b_only ##################
@OperationRegistry.register('optimized_matmul_padding_b_only')
def register_gemm_optimized_matmul_padding_b_only_operation(manifest):

    layouts = [
        [library.LayoutType.RowMajor, library.LayoutType.RowMajor, library.LayoutType.RowMajor],
    ]
    data_types = [
        [library.DataType.fp16, library.DataType.fp16, library.DataType.fp16],
    ]
    block_swizzle_descriptions = [
        'Gemm::Block::GemmIdentityBlockSwizzle<3, 0>',
    ]

    layout_padding = {
        library.LayoutType.RowMajor: 'layout::PaddingRowMajor',
        library.LayoutType.ColumnMajor: 'layout::PaddingColumnMajor',
    }

    # generate L1/L0TileShape search space
    tile_shapes = list(generate_tile_shapes(
        tile_shape_constraint_for_pingpong, # 自定义减枝函数
        element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
        stages=(2),
        step=16,
        tile_shape_range=TileShapeRange(
            l1_tile_m_range=(32, 128),
            l1_tile_n_range=(128, 256),
            l1_tile_k_range=(128, 256),
            l0_tile_m_range=(32, 128),
            l0_tile_n_range=(128, 256),
            l0_tile_k_range=(32, 64)
        )
    ))
    LOGGER.info(f'optimized_matmul_padding_b_only tile_shapes size={len(tile_shapes)}')

    # 正交tiling参数组合
    for layout, data_type, tile_shape, block_swizzle in product(
        layouts,
        data_types,
        tile_shapes,
        block_swizzle_descriptions
    ):
        l1_tile_shape, l0_tile_shape = tile_shape
        tensor_a = library.GemmTypeDescription(data_type[0], layout[0])
        tensor_b = library.GemmTypeDescription(data_type[1], layout[1])
        tensor_c = library.GemmTypeDescription(data_type[2], layout[2])
        op = GemmOperation(
            operation_type='gemm',
            kernel_type='optimized_matmul_padding_b_only',
            device_api='Gemm::Device::DeviceGemm',
            kernel_api='Gemm::Kernel::OptimizedMatmul',
            block_api='Gemm::Block::BlockMmad',
            dispatch='Gemm::MmadAtlasA2Preload<true,true>',
            l1_tile_shape=l1_tile_shape,
            l0_tile_shape=l0_tile_shape,
            a_type=tensor_a,
            b_type=tensor_b,
            c_type=tensor_c,
            block_swizzle=block_swizzle,
            epilogue='void',
            cpp_instance='OptimizedMatmulGemmOperation',
            custom_headers='#include "catlass/gemm/kernel/optimized_matmul.hpp"',
            custom_common_decls=OPTIMIZED_MATMUL_TILE_COPY_DECLARATION,
            custom_args={
                'layout_padding_b': layout_padding.get(layout[1], 'unknown_layout'),
                'compute_length_b': '49152', # 96 * 1024 / sizeof(half)
            }
        )
        manifest.append(op)
################## optimized_matmul_padding_b_only end ##################


################## optimized_matmul_without_padding ##################
@OperationRegistry.register('optimized_matmul_without_padding')
def register_gemm_optimized_matmul_without_padding_operation(manifest):

    layouts = [
        [library.LayoutType.RowMajor, library.LayoutType.RowMajor, library.LayoutType.RowMajor],
    ]
    data_types = [
        [library.DataType.fp16, library.DataType.fp16, library.DataType.fp16],
    ]
    block_swizzle_descriptions = [
        'Gemm::Block::GemmIdentityBlockSwizzle<3, 0>',
    ]

    # generate L1/L0TileShape search space
    tile_shapes = list(generate_tile_shapes(
        tile_shape_constraint_for_pingpong, # 自定义减枝函数
        element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
        stages=(2),
        step=16,
        tile_shape_range=TileShapeRange(
            l1_tile_m_range=(32, 128),
            l1_tile_n_range=(128, 256),
            l1_tile_k_range=(128, 256),
            l0_tile_m_range=(32, 128),
            l0_tile_n_range=(128, 256),
            l0_tile_k_range=(32, 64)
        )
    ))
    LOGGER.info(f'optimized_matmul_without_padding tile_shapes size={len(tile_shapes)}')

    # 正交tiling参数组合
    for layout, data_type, tile_shape, block_swizzle in product(
        layouts,
        data_types,
        tile_shapes,
        block_swizzle_descriptions
    ):
        l1_tile_shape, l0_tile_shape = tile_shape
        tensor_a = library.GemmTypeDescription(data_type[0], layout[0])
        tensor_b = library.GemmTypeDescription(data_type[1], layout[1])
        tensor_c = library.GemmTypeDescription(data_type[2], layout[2])
        op = GemmOperation(
            operation_type='gemm',
            kernel_type='optimized_matmul_without_padding',
            device_api='Gemm::Device::DeviceGemm',
            kernel_api='Gemm::Kernel::OptimizedMatmul',
            block_api='Gemm::Block::BlockMmad',
            dispatch='Gemm::MmadAtlasA2Preload<true,true>',
            l1_tile_shape=l1_tile_shape,
            l0_tile_shape=l0_tile_shape,
            a_type=tensor_a,
            b_type=tensor_b,
            c_type=tensor_c,
            block_swizzle=block_swizzle,
            epilogue='void',
            cpp_instance='OptimizedMatmulGemmOperation',
            custom_headers='#include "catlass/gemm/kernel/optimized_matmul.hpp"',
            custom_common_decls=OPTIMIZED_MATMUL_TILE_COPY_DECLARATION
        )
        manifest.append(op)
################## optimized_matmul_without_padding end ##################


############### search space generation methods ###############
L1_SIZE_MAX = 512 * 1024
L0A_SIZE_MAX = 64 * 1024
L0B_SIZE_MAX = 64 * 1024
L0C_SIZE_MAX = 128 * 1024


def tile_shape_constraint_for_pingpong(
    l1_tile_shape,
    l0_tile_shape,
    element_sizes_tuple,
    stages_tuple
):
    # constraint function for "Gemm::MmadAtlasA2Pingpong"
    l1_m, l1_n, l1_k = l1_tile_shape
    l0_m, l0_n, l0_k = l0_tile_shape
    element_a_size, element_b_size, element_accumulator_size = element_sizes_tuple
    stages = stages_tuple

    l1a_tile_size = l1_m * l1_k * element_a_size
    l1b_tile_size = l1_n * l1_k * element_b_size
    l0a_tile_size = l0_m * l0_k * element_a_size
    l0b_tile_size = l0_k * l0_n * element_b_size
    l0c_tile_size = l0_m * l0_n * element_accumulator_size

    # the basic blocks of L1 and L0 differ on the m and n axes is not supported yet
    if l1_m != l0_m or l1_n != l0_n:
        return False

    # check L1
    if (l1a_tile_size * stages + l1b_tile_size * stages) > L1_SIZE_MAX:
        return False

    # check L0A
    if l0a_tile_size * stages > L0A_SIZE_MAX:
        return False

    # check L0B
    if l0b_tile_size * stages > L0B_SIZE_MAX:
        return False

    # check L0C
    if l0c_tile_size > L0C_SIZE_MAX:
        return False

    if l1_m * l1_n < 64 * 64:
        return False

    return True


def tile_shape_constraint_for_preload_async(
    l1_tile_shape,
    l0_tile_shape,
    element_sizes_tuple,
    stages_tuple
):
    # constraint function for "Gemm::MmadAtlasA2PreloadAsync"
    l1_m, l1_n, l1_k = l1_tile_shape
    l0_m, l0_n, l0_k = l0_tile_shape
    element_a_size, element_b_size, element_accumulator_size = element_sizes_tuple
    _, l1_stages, l0a_stages, l0b_stages, l0c_stages, = stages_tuple

    l1a_tile_size = l1_m * l1_k * element_a_size
    l1b_tile_size = l1_n * l1_k * element_b_size
    l0a_tile_size = l0_m * l0_k * element_a_size
    l0b_tile_size = l0_k * l0_n * element_b_size
    l0c_tile_size = l0_m * l0_n * element_accumulator_size

    # the basic blocks of L1 and L0 differ on the m and n axes is not supported yet
    if l1_m != l0_m or l1_n != l0_n:
        return False

    # check L1
    if (l1a_tile_size * l1_stages + l1b_tile_size * l1_stages) > L1_SIZE_MAX:
        return False

    # check L0A
    if l0a_tile_size * l0a_stages > L0A_SIZE_MAX:
        return False

    # check L0B
    if l0b_tile_size * l0b_stages > L0B_SIZE_MAX:
        return False

    # check L0C
    if l0c_tile_size * l0c_stages > L0C_SIZE_MAX:
        return False

    if l1_m * l1_n < 64 * 64:
        return False

    return True


@dataclass
class TileShapeRange:
    l1_tile_m_range: tuple
    l1_tile_n_range: tuple
    l1_tile_k_range: tuple
    l0_tile_m_range: tuple
    l0_tile_n_range: tuple
    l0_tile_k_range: tuple


def generate_tile_shapes(
    constraint_func: callable = tile_shape_constraint_for_pingpong,
    element_sizes: tuple = (2, 2, 4),
    stages: tuple = (2),
    step: int = 16,
    tile_shape_range: TileShapeRange = TileShapeRange(
        l1_tile_m_range=(32, 128),
        l1_tile_n_range=(128, 256),
        l1_tile_k_range=(128, 256),
        l0_tile_m_range=(32, 128),
        l0_tile_n_range=(128, 256),
        l0_tile_k_range=(32, 64)
    )
):
    if step % 16 != 0:
        raise ValueError(f"step must be multiples of 16")

    def generator(
        element_sizes,
        stages
    ):
        params_ranges = [
            range(tile_shape_range.l1_tile_m_range[0], tile_shape_range.l1_tile_m_range[1] + step, step),
            range(tile_shape_range.l1_tile_n_range[0], tile_shape_range.l1_tile_n_range[1] + step, step),
            range(tile_shape_range.l1_tile_k_range[0], tile_shape_range.l1_tile_k_range[1] + step, step),
            range(tile_shape_range.l0_tile_m_range[0], tile_shape_range.l0_tile_m_range[1] + step, step),
            range(tile_shape_range.l0_tile_n_range[0], tile_shape_range.l0_tile_n_range[1] + step, step),
            range(tile_shape_range.l0_tile_k_range[0], tile_shape_range.l0_tile_k_range[1] + step, step)
        ]
        for l1_m, l1_n, l1_k, l0_m, l0_n, l0_k in product(*params_ranges):
            if constraint_func is None or constraint_func(
                (l1_m, l1_n, l1_k),
                (l0_m, l0_n, l0_k),
                element_sizes,
                stages
            ):
                yield ((l1_m, l1_n, l1_k), (l0_m, l0_n, l0_k))
    return generator(element_sizes, stages)

############### search space generation methods end ###############