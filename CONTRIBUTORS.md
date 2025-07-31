# 贡献指南

## 版本策略规范

### 版本策略

版本号遵循[语义化规范（SemVer）](https://semver.org/)，格式为`v<主版本>.<次版本>.<修订版本>.<feature标签>`（feature标签可选）。

| 版本类型       | 版本号格式          | 更新频率/触发条件                      | 示例                  |
|----------------|---------------------|----------------------------------------|-----------------------|
| **主版本**     | `vX.0.0`            | 重要独立功能/架构重大变更              | `v1.0.0` → `v2.0.0`  |
| **次版本**     | `vX.Y.0` (Y+1)      | 常规迭代（1-2个月）                    | `v1.1.0` → `v1.2.0`  |
| **修订版本**   | `vX.Y.Z` (Z+1)      | 紧急修复且不涉及新功能                 | `v1.2.0` → `v1.2.1`  |
| **feature标签**| `vX.Y.Z.feature`    | 临时测试功能（非正式发布）             | `v1.3.0.demo`        |

#### 刷新策略

- **主版本刷新**：
  - 在最新提交打`v(X+1).0.0` Tag
  - 基于最新次版本Tag节点创建归档分支`vX`
  
- **次版本/修订版本刷新**：
  - 在最新提交打`vX.(Y+1).0`或`vX.Y.(Z+1)` Tag

### 分支管理策略

#### 1. 新特性开发

- 所有新特性直接在 `master` 分支开发
- 开发完成后直接合入 `master`

#### 2. Bug修复流程

##### 当前活跃主版本修复

- 基于受影响次版本Tag创建临时分支（格式：`vX.Y.Z-bugfix`）
- 完成修复并验证后打修订版本Tag（格式：`vX.Y.(Z+1)`）
- 创建新Tag后立即删除临时分支

##### 已归档主版本修复

- 直接在归档分支（如 `v1`）上进行修复
- 完成修复后打修订版本Tag
- **必须**将修复内容同步到 `master` 分支

#### 3. 特殊特性开发

- 基于特定版本Tag创建独立开发分支
- 需要测试发布时打Feature Tag（格式：`vX.Y.Z.feature`）
- 如需合入主分支，必须通过 cherry-pick 合并到 `master`
- Feature标签版本**禁止**用于生产环境

## CATLASS Developers

### 昇腾团队

### 华南理工大学 陆璐教授团队

## ©️ 版权声明

Copyright (c) 2025 Huawei Technologies Co., Ltd.

This file is a part of the CANN Open Software.  
Licensed under CANN Open Software License Agreement Version 1.0 (the "License").  
Please refer to the License for details. You may not use this file except in compliance with the License.  

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY, OR FITNESS FOR A PARTICULAR   PURPOSE.  
See LICENSE in the root of the software repository for the full text of the License.

## 📜 许可证

[CANN Open Software License Agreement Version 1.0](LICENSE)