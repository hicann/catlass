# CATLASS 项目文档

## 1 Practice
>
> 代码实践，指导开发者按步骤上手CATLASS各层级代码开发和使用，逐渐具备完整算子开发、测试、调优、模型使用的能力。

- [01_quick_start](./1_Practice/01_quick_start.md)：介绍模板库的环境准备，和提供的算子样例的编译执行。
- [02_host_example_assembly](./1_Practice/02_host_example_assembly.md)：host侧组装matmul讲解
- [03_kernel_development](./1_Practice/03_kernel_development.md)：kernel代码拆解，展示模板组装机制、arguments、params及关键函数等代码
- [04_block_mmad_development](./1_Practice/04_block_mmad_development.md)：block_mmad代码拆解，模板组装机制、主要接口说明
- [05_block_scheduler_development](./1_Practice/05_block_scheduler_development.md)：block_schedule代码拆解，模板组装机制、主要接口说明
- [06_tile_development](./1_Practice/06_tile_development.md)：tile_copy/tile_mmad代码拆解，模板组装机制、主要接口说明
- [07_epilogue_adaptation](./1_Practice/07_epilogue_adaptation.md)：gemm算子的host/kernel层适配epilogue，及epilogue的block/tile开发
- [08_evaluation](./1_Practice/08_evaluation.md)：调测工具使用
- [09_example_contribution_guide](./1_Practice/09_example_contribution_guide.md)：完整样例的设计、开发、测试、合入流程
- [10_matmul_optimization](./1_Practice/10_matmul_optimization.md)：介绍模板库下的基础调优方式，包括如何通过Tiling调参、应用不同的Dispatch策略的方式，快速获得性能提升。
- 11_example_integration：样例适配、接入到整网的方式
- evaluation_tools
    - [ascendc_dump](./1_Practice/evaluation_tools/ascendc_dump.md)
    - [msdebug](./1_Practice/evaluation_tools/msdebug.md)
    - [performance_tools](./1_Practice/evaluation_tools/performance_tools.md)
    - [print](./1_Practice/evaluation_tools/print.md)
- others（folder）：存放内部和外部贡献的难以归类的实践文档
    - tla_rebuild：TLA样例改造

## 2 Design

- [00_project_overview](./2_Design/00_project_overview.md)：项目介绍、分层模块化设计、代码仓结构设计
- 01_kernel_design：算法设计
    - [01_example_design](./2_Design/01_kernel_design/01_example_design.md)：库上样例设计文档一览（将各样例文档放到样例文件夹内，此处只做归纳、牵引）
    - [02_swizzle](./2_Design/01_kernel_design/02_swizzle.md)：对模板库中`Swizzle`策略的基本介绍，这影响了AI Core上计算基本块间的顺序。
    - [03_dispatch_policies](./2_Design/01_kernel_design/03_dispatch_policies.md)：对模板库在`Block`层面上`BlockMmad`中的一个重要模板参数`DispatchPolicy`的介绍。
    - [04_matmul_summary](./2_Design/01_kernel_design/04_matmul_summary.md)：对模板库的`examples`目录内已有的`matmul`模板设计进行介绍，包含样例模板清单、理论模板清单、工程优化清单、模板应用浅述，可用于matmul性能调优时参考。
    - [05_aswt](./2_Design/01_kernel_design/05_aswt.md)：自适应滑窗tiling策略说明
    - 06_quant_summary：低精度专题
- 02_tla：
    - [01_layout](./2_Design/02_tla/01_layout.md)：TLA设计的layout结构和相关接口说明
    - 02_layout_tag：RowMajor、ColumnMajor、zN、nZ等layoutTag介绍和接口说明
    - [03_tensor](./2_Design/02_tla/03_tensor.md)：tensor结构体
- 03_evg

## 3 API

- [README](./3_API/README.md)：API清单入口
- [gemm api](./3_API/gemm_api.md)：Gemm API
- [Ascend C API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0003.html)：昇腾社区Ascend C API列表

## Appendix
>
>外部开源文章、视频

- 常见问题 [Q&A](./Q&A.md)
- 技术文章
    - 基础入门
        - [C++ template详解](https://www.runoob.com/w3cnote/c-templates-detail.html)
        - [Ascend C算子开发文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)
    - 概念理解
    - 问题定位
    - 性能优化
    - 优秀实践
- 培训视频
    - [昇腾社区在线课程](https://www.hiascend.com/edu/courses)：通过体系化的课程视频学习昇腾。相关课程推荐：
        - [Ascend C算子开发（入门）](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
        - [Ascend C算子开发（进阶）](https://www.hiascend.com/developer/courses/detail/1696414606799486977)
        - [Ascend C算子开发（高级）](https://www.hiascend.com/developer/courses/detail/1696690858236694530)
    - 昇腾训练营 CATLASS相关特辑 
        - [一站式掌握CATLASS模板库基本概念](https://www.bilibili.com/video/BV1f1BDBMES2/?spm_id_from=333.1387.collection.video_card.click&vd_source=ae7f2ef56954c6e4a7397c8386a66b47)：`昇腾CANN`【码力全开特辑】CATLASS学习系列课程第一讲。完整介绍CATLASS整体情况、算子快速上手、发展全景、生态共建。
        - [CATLASS算子开发初体验](https://www.bilibili.com/video/BV1DmBhBNEu8/?spm_id_from=333.1387.collection.video_card.click&vd_source=ae7f2ef56954c6e4a7397c8386a66b47)：`昇腾CANN`【码力全开特辑】CATLASS学习系列课程第二讲。以基础Matmul算子为例，完整介绍基于NPU的矩阵乘理论建模、代码实现（host/kernel/block三个层面）。
