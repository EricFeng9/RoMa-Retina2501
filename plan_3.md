我现在在做一个课题，目标是目标是基于我们已有的生成数据集（结构完全相同且对齐的cf-oct-fa图像对）训练出一个支持cf-fa，cf-oct（cf均为fix）的多模态配准模型。
为了探究 RoMa 架构在眼底多模态配准任务上的**原始性能基线 (Baseline)**，我们需要回归最原始的 RoMa 架构。
**当前的 Plan 2 目标已变更为：搭建并评估原始 RoMa (Vanilla RoMa) 在该任务上的表现，作为后续改进的参照组。**

此计划的核心是：**移除所有针对眼底血管的特定设计（如掩码引导、解剖偏置、特定课程学习等），完全依赖 RoMa 自身强大的特征匹配能力。**

---

## 核心架构：Vessel-Weighted RoMa (血管加权 RoMa)

我们将基于 RoMa 原架构，引入针对眼底血管的 **Hard Example Mining** 策略，通过 Loss 加权强迫模型关注血管特征。

### 1. 模型架构 (Architecture)

*   **Backbone**:
    *   **Coarse**: 冻结的 **DINOv2 (ViT-L/14)**。
    *   **Fine**: 原始 RoMa 的 CNN Decoder。
*   **输入**: 仅输入图像 $I^A, I^B$。**不**将血管掩码作为模型输入（掩码仅用于 Loss 计算）。
*   **Matcher**:
    *   标准 Transformer Decoder，不引入解剖偏置 (Anatomical Bias) 或 Masked Attention。

### 2. Loss 加权机制 (The Core Strategy)

利用 Dataset 提供的 `vessel_mask` (血管掩码) 对 Loss 进行空间加权。

*   **基本原则**:
    *   **In-Vessel (血管区域)**: Weight = **10.0**。惩罚这里的匹配错误。
    *   **Background (背景区域)**: Weight = **1.0**。维持基准惩罚。
    *   **公式**: $W(x) = 1.0 + (10.0 - 1.0) \times M_{vessel}(x)$
*   **适用范围**:
    *   **Coarse Loss**: 加权 Cross-Entropy / Focal Loss。强迫粗粒度特征在血管处对齐。
    *   **Fine Loss**: 加权 Regression Loss。强迫精细匹配在血管处达到亚像素精度。
*   **视盘 (Optic Disc)**:
    *   不做特殊处理（Weight=1.0 或 随血管掩码），假设主血管汇聚足以定位视盘。

### 3. 数据集 (Dataset)

修改 `FIVES_extract_v2.py`:
*   不再生成 Gaussian Soft Weight。
*   生成 **Hard Weight Map** (`vessel_weight`): 像素值为 1.0 或 10.0。
*   输出 `vessel_mask` 供可视化检查。

---

## 预期目标

1.  **特征显著性**: 模型应迅速学会“血管是唯一可靠的特征”，忽略背景噪声。
2.  **收敛速度**: 相比 Vanilla 版本，Loss 下降应更快，且主要由血管区域驱动。
3.  **鲁棒性**: 在 Coarse 阶段就能通过血管拓扑结构锁定大致位置。
