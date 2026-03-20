# Epilogue/Tile 类模板概述

Epilogue的tile层API作为[BlockEpilogue](../../../../../../docs/04_api/include/catlass/gemm/block/block_mmad.md)的模板参数，一般需要在kernel模板组装时做声明。

## API 清单
| 组件名                         | 描述 |
| :----------------------------------------------------------- | :------: |
| [tile_copy](./tile_copy.md)     |   完成mmad所需要的所有tile层搬运模板的集合  |
| [tile_mmad](./tile_mmad.md)     |   tile层mmad计算  |
| [copy_gm_to_l1  ](./copy_gm_to_l1.mdd)     |   将tile块从GM搬运到L1  |