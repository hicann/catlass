# Ascend C Templates
## Ascend C模板库简介
Ascend C Templates，中文名为Ascend C模板库，是一个聚焦于提供高性能矩阵乘类算子基础模板的代码库。  

通过抽象分层的方式将矩阵类算子代码模板化。算子计算逻辑可以进行白盒化组装，让算子代码可复用，可替换，可局部修改。针对昇腾硬件特点进行设计，可以支持复杂场景流水排布，如FA等。在上层代码逻辑共享的同时，可以支持底层硬件差异特化。 

本代码仓为Ascend C模板库联创代码仓。结合昇腾生态力量，共同设计研发算子模板，并提供典型算子的高性能实现代码样例

## 模板分层设计

![image](docs/images/api_level.png) 

分层详细介绍和各层级api，见[api](docs/api.md)文档。
补充说明：当前CANN社区版/商用版版本暂不支持device层实现，device层特性需获取最新CANN POC版本

## 目录结构说明
详细介绍见[code_organization](docs/code_organization.md)
``` 
├── docs     // 文档
├── examples // kernel使用样例
├── include  // 模板头文件
└── scripts  // 相关脚本
```
## 软件硬件配套说明
- 硬件型号支持  
Atlas A2服务器

- 平台：aarch64
- 配套软件：CANN 8.0.0.beta1及之后版本（参考《[CANN软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)》安装CANN开发套件包以及配套固件和驱动）  
cmake >= 3.15
## 快速上手
详细请参考[quickstart](docs/quickstart.md)  
设置环境变量  
```
# root用户安装（默认路径）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
执行一个样例matmul算子。  
在代码仓目录下，运行编译脚本。
```
bash scripts/build.sh 00_basic_matmul
```
切换到可执行文件的编译目录`build/bin`下，执行算子样例程序。
```
cd build/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID（可选）
./00_basic_matmul 256 512 1024 0
```
## 算子kernel列表
<table>
    <tr>
        <th>算子名称</th>
        <th>支持输入数据类型</th>
        <th>支持输出数据类型</th>
        <th>支持输入数据排布</th>
        <th>支持输出数据排布</th>
    </tr>
    <tr>
        <td rowspan="4">basic_matmul</td>
        <td>half</td>
        <td>half</td>
        <td rowspan="4">rowMajor/columnMajor</td>
        <td rowspan="4">rowMajor</td>
    </tr>
    <tr>
        <td>half</td>
        <td>float</td>
    </tr>
    <tr>
        <td>bfloat16</td>
        <td>bfloat16</td>
    </tr>
    <tr>
        <td>int8</td>
        <td>int32</td>
    </tr>
    <tr>
        <td rowspan="4">batched_matmul</td>
        <td>half</td>
        <td>half</td>
        <td rowspan="4">rowMajor/columnMajor</td>
        <td rowspan="4">rowMajor</td>
    </tr>
    <tr>
        <td>half</td>
        <td>float</td>
    </tr>
    <tr>
        <td>bfloat16</td>
        <td>bfloat16</td>
    </tr>
    <tr>
        <td>int8</td>
        <td>int32</td>
    </tr>
    <tr>
        <td rowspan="4">grouped_matmul</td>
        <td>half</td>
        <td>half</td>
        <td rowspan="4">（A矩阵）rowMajor；（B矩阵）rowMajor/columnMajor</td>
        <td rowspan="4">rowMajor</td>
    </tr>
    <tr>
        <td>half</td>
        <td>float</td>
    </tr>
    <tr>
        <td>bfloat16</td>
        <td>bfloat16</td>
    </tr>
    <tr>
        <td>int8</td>
        <td>int32</td>
    </tr>
    <tr>
        <td>matmul_add</td>
        <td>half</td>
        <td>half</td>
        <td>rowMajor/columnMajor</td>
        <td>rowMajor</td>
    </tr>
    <tr>
        <td rowspan="4">padding_matmul</td>
        <td>half</td>
        <td>half</td>
        <td rowspan="4">rowMajor/columnMajor</td>
        <td rowspan="4">rowMajor</td>
    </tr>
    <tr>
        <td>half</td>
        <td>float</td>
    </tr>
    <tr>
        <td>bfloat16</td>
        <td>bfloat16</td>
    </tr>
    <tr>
        <td>int8</td>
        <td>int32</td>
    </tr>
    <tr>
        <td rowspan="4">optimized_matmul</td>
        <td>half</td>
        <td>half</td>
        <td rowspan="4">rowMajor/columnMajor</td>
        <td rowspan="4">rowMajor</td>
    </tr>
    <tr>
        <td>half</td>
        <td>float</td>
    </tr>
    <tr>
        <td>bfloat16</td>
        <td>bfloat16</td>
    </tr>
    <tr>
        <td>int8</td>
        <td>int32</td>
    </tr>
</table>

参考[推荐使用配置](docs/recommended_configuration.md)了解如何达到最佳性能。

## 合作贡献者
华南理工大学 陆璐教授团队

## 版权声明
Copyright (c) 2025 Huawei Technologies Co., Ltd. 

This file is a part of the CANN Open Software.  
Licensed under CANN Open Software License Agreement Version 1.0 (the "License").  
Please refer to the License for details. You may not use this file except in compliance with the License.  

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,   
EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,     
MERCHANTABILITY, OR FITNESS FOR A PARTICULAR   PURPOSE.  
See LICENSE in the root of the software repository for the full text of the License.

## 许可证
[CANN Open Software License Agreement Version 1.0](LICENSE)
