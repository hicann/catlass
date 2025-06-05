# CATLASS
## CATLASS简介
CATLASS，中文名为昇腾算子模板库，是一个聚焦于提供高性能矩阵乘类算子基础模板的代码库。  

通过抽象分层的方式将矩阵类算子代码模板化。算子计算逻辑可以进行白盒化组装，让算子代码可复用，可替换，可局部修改。针对昇腾硬件特点进行设计，可以支持复杂场景流水排布，如FA等。在上层代码逻辑共享的同时，可以支持底层硬件差异特化。 

本代码仓为CATLASS联创代码仓。结合昇腾生态力量，共同设计研发算子模板，并提供典型算子的高性能实现代码样例

## 模板分层设计

![image](docs/images/api_level.png) 

分层详细介绍和各层级api，见[api](docs/api.md)文档。

## 目录结构说明
``` 
├── docs     // 文档
├── examples // kernel使用样例
├── include  // 模板头文件
└── scripts  // 相关脚本
```
## 软件硬件配套说明
硬件型号支持：  
  - Atlas 800T A2 服务器
  - Atlas 200T A2 Box16服务器

平台：aarch64/x86

配套软件：
  - gcc >= 9.3
  - cmake >= 3.15
  - python >= 3.10

CANN版本要求：

| CANN包类别 | 版本要求                    | 获取方式                                                                                                             |
| ---------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 社区版     | 8.2.RC1.alpha002 及之后版本 | [社区CANN包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha002) |
| 商用版     | 8.1.RC1及之后版本           | 请咨询对应Support/SupportE获取                                                                                       |
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
