我现在在做一个课题，目标是目标是基于我们已有的生成数据集（结构完全相同且对齐的cf-oct-fa图像对）训练出一个支持cf-fa，cf-oct（cf均为fix）的多模态配准模型。
为了探究 RoMa 架构在眼底多模态配准任务上的**原始性能基线 (Baseline)**，我们需要回归最原始的 RoMa 架构。
**当前的 Plan 2 目标已变更为：搭建并评估原始 RoMa (Vanilla RoMa) 在该任务上的表现，作为后续改进的参照组。**

此计划的核心是：**移除所有针对眼底血管的特定设计（如掩码引导、解剖偏置、特定课程学习等），完全依赖 RoMa 自身强大的特征匹配能力。**

---

## 核心架构：原始 RoMa (Vanilla RoMa)

我们将完全保留 RoMa 原论文 (CVPR 2024) 的设计理念，不引入任何额外的解剖学先验。

### 1. 模型架构 (Architecture)

*   **Backbone (特征提取)**
    *   **Coarse (粗粒度)**: 冻结的 **DINOv2 (ViT-L/14)**。直接按原论文方式使用，不添加 Modality Adapter。依靠 DINOv2 在大规模自然图像上预训练的强大泛化能力来提取语义特征。
    *   **Fine (细粒度)**: 原始 RoMa 使用的卷积层结构 (通常是简单的 ResNet block 或 VGG style block，视官方实现而定)，用于提取局部精细特征。**不**强制替换为特定的 VGG19，除非那是官方默认。
    *   **输入**: 直接输入归一化后的灰度或RGB图像。**不**输入血管掩码。

*   **Transformer Match Decoder**
    *   使用标准的 Transformer Decoder 进行特征交互。
    *   **注意力机制**: 使用标准的 Dot-Product Attention + Softmax。
    *   **关键点**: **移除**所有 "Anatomical Biasing" (解剖偏置)。注意力矩阵 $S(i, j)$ 仅由特征相似度决定，不受血管掩码 $M^{vessel}$ 的加权影响。让模型自己去学习应该关注哪里。

### 2. 损失函数 (Loss Function)

回归最原始的监督方式，仅利用几何变换真值进行监督。

*   **Coarse Loss (粗粒度损失)**:
    *   使用标准的 **Focal Loss** 或 **Cross Entropy Loss** (Regression-by-classification)。
    *   监督信号：基于数据增强生成的几何变换矩阵 $T$ 计算出的 GT 对应关系。
    *   **移除**基于血管掩码的权重 ($W_{vessel}$) 和梯度保底策略。所有像素（无论是血管还是背景）在损失函数中拥有平等的初始权重，或者仅使用 RoMa 默认的动态权重机制。

*   **Fine Loss (细粒度损失)**:
    *   使用标准的 **Robust Regression Loss** (如 Charbonnier Loss 或 L2 Loss)。
    *   监督信号：GT 坐标 $x_{gt}^B$。
    *   同样**移除**任何基于掩码的区域过滤或加权。

---

## 训练数据生成 (Data Generation)

利用现有的 `FIVES_extract_v2` 数据集，模拟自监督训练流程，但去除掩码依赖。

### 1. 几何变换 (Geometric Augmentation)
*   **模拟**: 随机生成放射变换矩阵 $T$ (旋转范围 $\pm 90^\circ$，平移，缩放)。
*   **图像**: 对 $I^B$ 应用 $T$ 得到 $I^B_{warped}$。
*   **真值**: 根据 $T$ 计算 $I^A$ 到 $I^B_{warped}$ 的密集坐标映射关系。

### 2. 掩码的处理
*   **训练时**: **完全不读取/不使用**血管掩码。模型只能看到 $I^A$ 和 $I^B_{warped}$。
*   **评估时**: 仅在计算指标 (如 MSE, Dice) 时在 CPU/计算端使用掩码来过滤无效背景区域，确保指标公平，但模型推断过程本身不依赖掩码。

---

## 训练策略 (Training Strategy)

*   **单一阶段**: 移除"课程学习" (Curriculum Learning)。
*   **流程**: 从头到尾使用统一的损失函数和学习率策略进行训练。
*   **超参数**: 参考 RoMa 官方推荐配置 (如 LR=1e-4, AdamW, etc.)。

---

## 推理与配准流程 (Inference)

1.  **特征匹配**:
    *   输入原始图像对 $I^A, I^B$。
    *   RoMa 输出密集匹配点对或变形场。
2.  **后处理**:
    *   使用标准的 **RANSAC** 估算放射矩阵。
    *   **移除** "Spatial Binning" (空间均匀化采样) 中人为强加的网格约束，除非这是 RANSAC 之前的标准降采样步骤 (如仅保留 Top-K confidence points)，但不应包含任何 specific 的针对血管分布的设计。
3.  **Warp**: 利用计算出的矩阵变换图像。

---

## 预期目标 (Baseline Goal)

通过这个 Baseline 实验，我们需要回答以下问题：
1.  **DINOv2 的直接迁移能力**: 未经微调或适配的 DINOv2 特征在眼底多模态 (CF vs OCT/FA) 差异下，是否仍能保持足够的相似性？
2.  **纹理vs结构**: 在没有血管掩码强制引导的情况下，RoMa 是会关注血管结构，还是会因为模态差异（如反色、对比度差异）而产生大量误匹配？
3.  **旋转鲁棒性**: 原始 RoMa 在不加额外约束的情况下，处理大尺度旋转 (Retina 图像常见) 的能力极限在哪里？

这个 Baseline 的结果将作为后续添加 "Modality Adapter" 和 "Mask Guidance" 效果提升的**标尺**。

---

## 实施细节修改 (Implementation Changes)

### 1. DataLoader
*   不需要返回 `image1` (mask0/1)。或者虽然返回但不传递给模型。
*   保持单纯的 Image Pair + Transform Matrix 格式。

### 2. Model (RoMa.py)
*   确保 `forward` 函数中不接受 `mask` 参数。
*   确保 `CoarseEncoder` 直接是 DINOv2，没有额外的 Adapter 层。
*   确保 `Attention` 计算没有 `bias` 项。

### 3. Training Script
*   移除所有计算 Loss 时的 Mask 加权逻辑。
*   移除 Lambda 调度逻辑。

### 4. Metrics
*   评估代码需保持一致（依然只在有效区域计算准确率），以便与后续改进模型公平对比。