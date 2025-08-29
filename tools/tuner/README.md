## msTuner_CATLASS (Mindstudio Tuner for CATLASS) - Tiling自动寻优工具

mstuner_catlass是用于CATLASS模板库算子Tiling参数寻优的工具，支持自定义搜索空间，实例化搜索空间内全量算子，批量完成在板性能测试，为算子Tiling参数寻优提供参考。

### 使用示例

以m=256，n=512，k=1024 的**basic_matmul**的tiling参数寻优为例，使用预设的搜索空间配置，执行以下命令完成工具的编译。

```bash
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=basic_matmul mstuner_catlass
```

输入mstuner_catlass命令行命令，启动性能测试，

```bash
export LD_LIBRARY_PATH=$PWD/output/lib64/:$LD_LIBRARY_PATH
./output/bin/mstuner_catlass --m=256 --n=512 --k=1024 --device=0 --output=results.csv
```

运行成功如下所示(实际运行结果因硬件差异与硬件性能波动不一定完全相同)，

```bash
$ ./output/bin/mstuner_catlass --m=256 --n=512 --k=1024 --device=0 --output=results.csv
[INFO ] Set profile output file /path_to_my_repo/catlass/output/results.csv
[INFO ] Start to initialize device 0
[INFO ] Initializing device 0 success
[INFO ] Initializing 1701 operations
[WARN ] Current freq 800 is lower than rated freq 1800, run warm up
[INFO ] Warm up finished, rated freq 1800, current freq 1800
================================

             case_id : 1
   task_duration(us) : 19.380
           device_id : 0
           operation : Gemm
         description : catlass_gemm_basic_matmul_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_32x128x128_32x128x32_swizzle3x0
       l0_tile_shape : 32x128x32
       l1_tile_shape : 32x128x128
             swizzle : swizzle3x0
                   m : 256
                   n : 512
                   k : 1024
                   A : fp16:row
                   B : fp16:row
                   C : fp16:row

================================

...

================================
Top 10:
case_id,task_duration(us),device_id,operation,description,m,n,k,A,B,C
489,12.740,7,Gemm,catlass_gemm_basic_matmul_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_64x128x128_64x128x64_swizzle3x1,256,512,1024,fp16:row,fp16:row,fp16:row
...
[INFO ] Save profile data to /path_to_my_repo/catlass/output/results.csv success
```

## 编译

支持通过`-DCATLASS_LIBRARY_KERNELS=<kernel_name>`命令过滤算子，当算子的description信息包含`kernel_name`时，该算子用例代码会被生成并编译，比如通过如下命令指定编译`basic_matmul`类算子，

```bash
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=basic_matmul mstuner_catlass
```

可直接指定具体的单个算子实例的description信息，比如通过如下命令指定仅编译上一节的使用示例中所展示的case_id为1的算子，

```bash
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=catlass_gemm_basic_matmul_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_32x128x128_32x128x32_swizzle3x1 mstuner_catlass
```

当前已支持如下算子类型：

- basic_matmul
- grouped_matmul

编译也可通过如下的cmake命令完成，

```bash
mkdir build
cd build
cmake .. -DCATLASS_LIBRARY_KERNELS=basic_matmul
make mstuner_catlass -j
cmake --install . --component catlass_kernels
cmake --install . --component mstuner_catlass
```

## 工具运行命令

mstuner_catlass 支持以下命令：

| 命令          | 示例                          | 描述                                                         |
| ------------- | ----------------------------- | ------------------------------------------------------------ |
| --help, -h    | --help                        | 展示工具支持的命令                                           |
| --kernels     | --kernels=basic_matmul        | 过滤寻优的算子类型，其与算子的description列字符串进行子串匹配，未匹配时该算子会被跳过 |
| --output      | --output=./profile_result.csv | 指定算子性能数据落盘文件路径                                 |
| --device      | --device=0                    | 指定运行的单卡ID                                             |
| --m           | --m=256                       | 指定输入矩阵的维度m                                          |
| --n           | --n=256                       | 指定输入矩阵的维度m                                          |
| --k           | --k=256                       | 指定输入矩阵的维度k                                          |
| --A           | --A=fp16:row                  | 通过指定矩阵A的数据类型与内存排布过滤算子                      |
| --B           | --B=fp16:column               | 通过指定矩阵B的数据类型与内存排布过滤算子                    |
| --C           | --C=fp16:row                  | 通过指定矩阵C的数据类型与内存排布过滤算子                    |
| --group_count | --group_count=128             | 指定grouped_matmul类算子的group数量                          |

当搜索空间配置并生成了多种A、B、C的数据类型与内存排布时，支持通过`--A/--B/--C=数据类型:内存排布`命令对算子进行过滤。
 - 数据类型支持`u8, int8, int32, fp16, bf16, fp32`
 - 内存排布支持`row, column, nZ, zN, zZ, padding_row_major, padding_column_major, nN`
注意：不指定`--output`时，不会落盘算子性能数据。

## 搜索空间配置

mstuner_catlass支持在 `tools/library/scripts/search_space.py`文件中对算子tiling参数的搜索空间进行自定义配置，支持自定义配置layouts、data types、L1/L0 Tile Shapes、Swizzle策略等参数自动正交生成全量搜索空间，同时支持自定义剪枝函数加速搜索空间遍历，每种正交配置组合会实例化为一个独立算子，当搜索空间配置为较大时，会导致数万个算子被实例化，导致编译耗时较长，且可能无法保证编译成功，同时算子数量较多时，算子下发前注册耗时也较长，因此建议合理配置搜索空间，尽量控制在5000以内以获得最佳的工具体验。

算子数量可通过查看日志文件`build/tools/library/catlass_library_code_generation.log`，如下所示，basic_matmul的搜索空间会实例化1701个算子，

```txt
INFO:search_space:basic_matmul tile_shapes size=1701
INFO:search_space:grouped_matmul tile_shapes size=576
INFO:manifest:operations that will be generated in total: 1701
...
```

以`basic_matmul`算子的搜索空间为例，其配置位于函数`register_gemm_basic_matmul_operation`中，

- layouts配置

  ```python
      layouts = [
          [library.LayoutType.RowMajor, library.LayoutType.RowMajor, library.LayoutType.RowMajor],
      ]
  ```

- data types配置

  ```python
      data_types = [
          [library.DataType.fp16, library.DataType.fp16, library.DataType.fp16]
      ]
  ```

- L1/L0 Tile Shapes配置与自定义剪枝函数`tile_shape_constraint_for_pingpong`

  ```python
      tile_shapes = list(generate_tile_shapes(
          tile_shape_constraint_for_pingpong, # set constraint function based on dispatch policy
          # below are arguments for constraint function
          element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
          stages=[2], # stages of dispatch policy for estimating boundary conditions, e.g. 2 for UB stages
          step=16, # step size for iterating the next tile shape on each dimension of L1/L0 tile shape
          tile_shape_range=TileShapeRange(
            l1_tile_m_range=(32, 128),  # range of L1TileShape::M/N/K
            l1_tile_n_range=(128, 256),
            l1_tile_k_range=(128, 256),
            l0_tile_m_range=(32, 128),  # range of L0TileShape::M/N/K
            l0_tile_n_range=(128, 256),
            l0_tile_k_range=(32, 64)
          )
      ))
  ```

- Swizzle策略配置

  ```python
      block_swizzle_descriptions = [
          'Gemm::Block::GemmIdentityBlockSwizzle<3, 0>',
      ]
  ```
