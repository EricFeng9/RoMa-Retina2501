我现在在做一个课题，目标是目标是基于我们已有的生成数据集（结构完全相同且对齐的cf-oct-fa图像对）训练出一个支持cf-fa，cf-oct（cf均为fix）的多模态配准模型。我们目前在用fractMorph模型进行2D化改造后的模型，但是fractMorph2D只能针对一些微小血管形变进行对齐和修复，但是针对真实情况不对齐图像中出现的血管大尺度旋转（如15 30度的旋转角度）、大的位置偏移的情况下配准效果较差，1是原先的逐像素点移动的模式很差2是我们修改成让模型直接预测放射参数，效果也很差，几乎不收敛。我现在的想法是改造 RoMa (Robust Dense Feature Matching)，学习cf图像和oct/fa图像之间的关键点匹配，然后利用这些匹配出来的关键点计算放射矩阵，对跨模态图像进行配准。RoMa 在处理极端变化（如大尺度旋转、光照突变和不同模态纹理差异）上比 LoFTR 更具鲁棒性。其核心优势在于结合了冻结的基础模型特征(DINOv2)的全局鲁棒性和专门的 ConvNet 的精细定位能力。

@data/CF_260116_expand 数据集每一个文件夹下面是一组多模态数据集，格式为[编号]_[模态]_[其他备注].png。配准模式共有cffa（cf为fix），cfoct（cf为fix），octfa（oct为fix），cfocta（cf为fix）四种模式。_cf_clip_512(或者_cf_gen_clip_512)和_octa_gen用于cfocta配准的训练，剩下的_cf_512(_cf_gen_512)、_fa(_fa_gen)、_oct(_oct_gen)分别用于对应的cffa和cfoct和faoct配准。

**重要更新：** 每个数据集文件夹现在包含一张血管分割掩码图（格式如 [编号]_vessel_mask.png），用于表示该组数据中所有模态（CF/OCT/FA/CFCLIP/OCTA）共同的血管结构。该掩码将用于引导模型关注血管区域，屏蔽背景、病灶等非血管结构的干扰。由@data/CF_260116_expand/cf_260116_expand.py 根据需要返回给训练和测试脚本

为了实现基于 RoMa 的眼底图像跨模态配准，你需要将原有的"基于变形场"或"直接参数回归"的思路转变为"**基于特征匹配的几何解算**"思路。以下是为您定制的 RoMa 改造方案：**解剖特征驱动的鲁棒稠密配准**。

以下是为您整理的详细实现与训练计划，包含核心公式与步骤：

---

## 核心架构改造：从"通用匹配"到"解剖引导"

RoMa 采用粗到精(Coarse-to-Fine)的架构。我们将引入您的血管分割掩码 $M^{vessel}$ 来增强其对眼底解剖结构的感知。

通过引入血管分割掩码 $M^{vessel}$，改进后的 RoMa 模型采用以下**解剖引导**策略：

1.  **模态对齐与特征提取：** RoMa 默认使用冻结的 DINOv2 获取粗特征。由于眼底图(CF、OCT、FA)与自然图像存在领域差异，需进行模态对齐。在 DINOv2 前添加轻量级的 Modality Adapter，将不同模态映射到 DINOv2 熟悉的特征空间。
2.  **血管增强融合：** 将下采样的 $M^{vessel}$ 与图像特征在通道维度拼接或作为特征偏移量引入，增强模型对血管结构的感知。
3.  **注意力解耦与偏置 (Anatomical Biasing)：** Transformer 注意力机制采用**加性偏置 (Additive Bias)**，引入血管权重矩阵 $W_{pair}$，强制 Transformer 优先关注血管交叉点和分叉点，这些区域在跨模态中具有最高的拓扑一致性。
4.  **损失监督与梯度保底 (Loss Weighting)：** 损失函数给背景区域预留 0.1 的"低保"权重（梯度保底），确保即使匹配点落在背景上，也能产生微弱梯度推动模型学习大尺度的旋转信息。
5.  **空间分布约束：** 在 RANSAC 前进行空间均匀化 (Spatial Binning)，将图像划分为 8×8 个分区，每个分区只选取置信度最高的 N 个点对，确保匹配点遍布全图，提高放射变换矩阵的稳定性。

---

## 第一阶段：模型架构改造 (Architecture)

RoMa 的核心是结合冻结的 DINOv2 特征和专门的 ConvNet，通过 Transformer 建立稠密特征匹配。

### 1. 模态适配器与双流特征提取 (Backbone)

RoMa 默认使用冻结的 DINOv2 获取粗特征。由于眼底图(CF、OCT、FA)与自然图像存在领域差异，需进行模态对齐。

#### 粗特征编码器 $F_{coarse}$：

* **基础：** 冻结的 DINOv2 (ViT-L/14)
* **改造：** 在 DINOv2 前添加一个轻量级的 Modality Adapter（如 3×3 卷积层），将不同模态映射到 DINOv2 熟悉的特征空间
* **血管增强策略：** **不要**在 Backbone 阶段拼接掩码，而是将掩码 $M^{vessel}$ 传递给 Transformer 的注意力机制中使用。Backbone 只负责提取原始图像特征，保持特征的纯净性。

#### 精特征编码器 $F_{fine}$：

* **选择：** 采用专门的 VGG19。RoMa 的实验证明，VGG19 在精细定位任务中显著优于 ResNet

* **输入：** 图像对 $I^A$ (固定图 CF) 和 $I^B$ (待配准图 OCT/FA)。血管分割掩码 $M^{vessel}$ 不在此阶段使用。
* **输出：** 提取两层特征。
* **粗级特征：** $\tilde{F} \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8} \times D}$，用于全局匹配。
* **精级特征：** $\hat{F} \in \mathbb{R}^{\frac{H}{2} \times \frac{W}{2} \times d}$，用于亚像素微调。

* **解剖掩码融合策略：**
* **关键原则：** 血管掩码信息**不在 Backbone 中拼接**，而是在 Transformer 的注意力机制中通过**加性偏置 (Additive Bias)** 引入。这样保留了全局感知能力，同时避免了掩码直接污染特征空间，有助于 Transformer 建立正确的全局位置编码。





### 2. 掩码引导的 Transformer 匹配解码器 (Match Decoder)

RoMa 的解码器通过预测**锚点概率 (Anchor Probabilities)** 来实现回归。

#### 锚点离散化：

我们将配准空间离散化为 $K = 64 \times 64$ 个锚点。定义匹配条件分布：

$$P_{coarse}(x^B | x^A) = \sum_{k=1}^K \pi_k(x^A) B_{m_k}$$

其中 $\pi_k$ 是概率，$B$ 是均匀分布，$\{m_k\}$ 是网格锚点。

#### 解剖偏置注入 (Anatomical Biasing)：

在 Transformer 的注意力计算中，引入血管权重矩阵 $W_{pair}$：

$$S_{biased}(i, j) = S(i, j) + \lambda \cdot (M^A(i) \cdot M^B(j_{trans}))$$

* **物理意义：** 强制 Transformer 优先关注血管交叉点和分叉点，这些区域在跨模态中具有最高的拓扑一致性。
* **加性偏置的优势：** 允许非血管区域保留微弱的响应，保证了梯度的连续性；引导注意力机制向血管区域倾斜，而不是直接关闭非血管通道。

这种方式让 $I^A$ 的每一个像素都能在"血管及其邻域"的范围内感知 $I^B$ 的全局血管拓扑结构：既能处理大尺度旋转（±90°），又避免了纯背景噪声主导几何估计。

---

## 第二阶段：训练数据生成 (GT Label Generation)

利用你现有的**已对齐生成数据集**，你可以自动生成完美的训练标签 。

### 1. 模拟放射变换

在训练时，随机生成一个放射变换矩阵 $T$（包含 $\pm 90^\circ$ 的旋转、平移和缩放），并作用于 $I^B$ 得到 $I_{warped}^B$。

**增强策略：**
* **旋转：** 均匀采样 $[-90^\circ, 90^\circ]$ 范围内的角度，覆盖极大角度旋转情况
* **翻转：** 随机加入水平/垂直翻转，模拟不同采集方向
  * 水平翻转概率：随机应用
  * 垂直翻转概率：随机应用
  * 翻转与旋转可组合使用
* **全局感知：** 由于 DINOv2 提取的是语义级别的特征，即便血管旋转 90°，其局部的拓扑结构特征在 DINO 空间中依然保持高度相似

### 2. 建立坐标映射 (Ground Truth with Vessel Constraint)

对于 $I^A$ 中的任意像素点 $i = (x, y)$，其对应的真值点 $j_{gt}$ 在 $I_{warped}^B$ 中的位置为：
$$j_{gt} = T \cdot [x, y, 1]^T$$

**引入血管掩码的GT筛选：**
* **粗级标签 $\mathcal{M}_c^{gt}$：** 若两个 1/8 网格中心点满足以下条件，则标记为正样本对：
  1. 重投影距离小于阈值（如 1 个网格单位）
  2. **$\tilde{M}^A(i) = 1$ 且 $\tilde{M}^B(j_{gt}) = 1$（两点均在血管区域内）**
* **精级标签 $\mathcal{M}_f^{gt}$：** 直接使用 $j_{gt}$ 作为 $L_2$ 损失的目标坐标，同样需满足掩码约束。

**作用：** 通过掩码筛选，训练数据只包含血管区域的匹配点对，模型从一开始就学习忽略非血管结构，避免在病灶、视盘等区域产生误匹配 。

---

## 第三阶段：损失函数设计 (Supervision)

训练损失函数：回归与分类的结合

RoMa 提出了一种改进的损失公式：全局匹配阶段建模多模态分布（回归转分类），细化阶段建模单模态分布（鲁棒回归）。

### 1. 粗级损失 $\mathcal{L}_{coarse}$ (Regression-by-classification)

利用您的对齐数据集，通过模拟放射变换 $T$ 生成真值坐标 $x_{gt}^B$。使用 KL 散度最小化预测分布与真值锚点分布的差异：

$$\mathcal{L}_{coarse} = -\frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} \max(W_{vessel}, \varepsilon) \cdot \log P_c(\tilde{i}, \tilde{j})$$

其中：
* $\mathcal{M}$ 是匹配点对集合
* $W_{vessel}$ 是血管权重
* $\varepsilon = 0.1$ 是梯度保底阈值
* $P_c(\tilde{i}, \tilde{j})$ 是预测的锚点概率分布

**梯度保底 (Gradient Guard)：**
* 即使是非血管区域，也保留 $\varepsilon=0.1$ 的权重，确保模型能学习全图的大尺度旋转信息。
* **物理意义：** 即使点对落在背景上，也会产生微弱的梯度推动模型学习大尺度的几何变换。血管区域的高权重则负责后期的精细对齐。

### 2. 精级细化损失 $\mathcal{L}_{fine}$ (Robust Regression)

在细化阶段，RoMa 使用广义 Charbonnier 损失，其梯度在局部类似 L2，但在全局趋于零，能有效抑制异常值：

$$\mathcal{L}_{fine} \propto (||W(x^A) - x_{gt}^B||^2 + s)^{1/4}$$

其中：
* $W(x^A)$ 是变换后的坐标
* $x_{gt}^B$ 是真值坐标
* $s$ 是平滑参数

**优势：** 相比 L2 损失，Charbonnier 损失对异常值更加鲁棒，适合处理跨模态匹配中的噪声和误匹配。

---

## 针对眼底配准的专项策略

### 1. 极大尺度旋转应对 (±90° Support)

* **训练增强：** 将旋转采样范围扩大至 $[-90°, 90°]$，并随机加入水平/垂直翻转。
* **全局感知：** 由于 DINOv2 提取的是语义级别的特征，即便血管旋转 90°，其局部的拓扑结构特征在 DINO 空间中依然保持高度相似。RoMa 的 Transformer 解码器在不使用位置编码的情况下非常健壮。

### 2. 空间均匀化采样 (Spatial Binning for RANSAC)

* **问题：** RoMa 生成的是稠密变形场，直接解算矩阵容易受主血管区域干扰。
* **操作：** 从 RoMa 的稠密输出中，基于置信度 $p(x^A)$ 进行采样。
* **网格约束：** 
  - 将 $518 \times 518$ 的图像划分为 $8 \times 8 = 64$ 个网格分区
  - 在每个分区中只选取置信度最高的前 5 个匹配点
  - 确保匹配点遍布全图，提高放射变换矩阵 $M$ 的稳定性

---

## 第四阶段：训练策略：解剖课程学习 (Curriculum Learning)

让模型先学"简单的全图对齐"，再学"难的血管对齐"。

* **阶段 1 (前 25 Epochs)：全图感知期**
  - 设置 $\lambda = 0$，不使用血管掩码偏置。
  - 让模型先利用全图信息学会处理 $\pm 90^\circ$ 的旋转和大尺度偏移。
* **阶段 2 (25-50 Epochs)：软掩码引导期**
  - 设置 $\lambda = 0.2$，引入弱的血管掩码加性偏置。
  - 模型开始向血管聚焦，但在非血管区域保留梯度。
* **阶段 3 (50 Epochs 以后)：精度冲刺期**
  - 设置 $\lambda = 0.05$，使用更弱的血管权重。
  - 进行最后的精度对齐冲刺。
  - **早停机制**：只在此阶段开启早停，patience=8。

**验证/测试阶段：**
- 设置 $\lambda = 0$，完全关闭血管分割图的注意力偏置。
- 确保模型在纯图像特征下进行推理，避免对掩码的依赖。

---

## 第五阶段：配准解算 (Inference & Registration)

模型训练完成后，配准不再是预测变形场，而是解算矩阵。

### 推理配置

**重要：验证/测试时关闭血管掩码偏置**
- 设置 $\lambda = 0$，完全关闭 Transformer 注意力机制中的血管掩码偏置
- 原因：
  1. 避免模型对掩码产生依赖，确保在没有精确掩码的真实场景下也能工作
  2. 测试模型在纯图像特征下的泛化能力
  3. 课程学习已经让模型学会了血管区域的重要性，推理时不需要显式引导

### 配准流程

1. **特征匹配（$\lambda = 0$）：** 
   * 输入未对齐的完整图像 $I^A, I^B$（包含背景，不做预过滤）。
   * 掩码 $M^{vessel}$ 在推理时不参与注意力计算（$\lambda = 0$）。
   * RoMa 输出点对集合 $\{(x_k, y_k) \leftrightarrow (x'_k, y'_k)\}$。
   * 模型依靠训练时学到的特征表示自动关注血管区域。

2. **空间均匀化采样 (Spatial Binning - 全图无掩码约束)：**
   * RoMa 生成的是稠密变形场，直接解算矩阵容易受主血管区域干扰。
   * **操作：** 从 RoMa 的稠密输出中，基于置信度 $p(x^A)$ 进行**全图均匀采样**。
   * **网格约束：** 
     - 将 $518 \times 518$ 的图像划分为 $8 \times 8 = 64$ 个网格分区（**全图范围**）
     - 在每个分区中只选取置信度最高的前 5 个匹配点
     - 最多保留 $8 \times 8 \times 5 = 320$ 个匹配点（实际通常更少）
     - **不使用掩码过滤**：让模型的置信度预测决定点的质量
   * **理由：** 
     - 强制匹配点在空间上均匀分布
     - 极大提高放射变换矩阵求解的稳定性和精度
     - 防止模型退化为单位矩阵 $I$
     - 端到端学习：模型自动学习给背景区域低置信度
   * 使用 **RANSAC** 算法剔除误匹配，并求解放射矩阵 $M$ ：
$$\min_M \sum_k ||M \cdot [x_k, y_k, 1]^T - [x'_k, y'_k, 1]^T||^2$$

3. **大尺度转换处理：** 由于课程学习的训练策略，模型即使在 $\lambda = 0$ 的情况下，也能在极大旋转角度（±90°）下依靠学到的血管拓扑特征建立准确匹配。RoMa 的 Transformer 解码器在不使用位置编码的情况下非常健壮，能够处理极大尺度的旋转。

4. **重采样与评估：** 
   * 利用 $M$ 对 $I^B$ 进行 Warp，完成配准
   * **评估时使用有效区域过滤**：计算 MSE 等指标时调用 `filter_valid_area()`，只在眼底圆形有效区域内计算，避免背景噪声影响评估结果

---

## 第五阶段：数据增强策略 (Data Augmentation)

在训练过程中，对图像进行放射变换的同时，**同步对血管掩码进行相同的几何变换**，以保证掩码与变换后的图像对齐（用于损失函数权重计算）。

### 增强策略

1. **同步放射变换：**
   * 生成随机放射矩阵 $T$（旋转 $\pm 45^\circ$，平移 $\pm 10\%$，缩放 $0.9 \sim 1.1$）
   * **随机翻转：** 10% 概率水平/垂直翻转（可与旋转组合）
   * 同时对 $I^B$ 和 $M^{vessel}_B$ 应用相同的变换得到 $I^B_{warped}$ 和 $M^{vessel}_{B,warped}$
   * **插值方法：** 图像使用双线性插值，掩码使用最近邻插值（保持二值性）
   * **不对图像进行掩码过滤**：保留完整图像（含背景），让模型端到端学习

2. **血管结构保持增强：**
   * 亮度、对比度调整（仅影响图像，不影响掩码）
   * 轻微高斯噪声（$\sigma < 0.02$，避免破坏血管边缘）
   * **禁止使用：** Elastic Deformation（会破坏掩码与图像的几何对应关系）

3. **掩码使用说明：**
   * 血管掩码 $M^{vessel}$ 仅用于：
     - **损失函数权重**：粗级损失中的 `torch.clamp(mask, min=0.1)` 梯度保底
     - **注意力偏置**（训练时）：$\lambda \cdot vessel\_bias$ 软引导
   * **不用于**：
     - ❌ 图像预过滤（不将背景置零）
     - ❌ 匹配点硬过滤（不根据掩码丢弃点）
     - ❌ 空间均匀化约束（不限制网格范围）

---

## 第六阶段：实现细节与模块设计 (Implementation Details)

### 数据加载器修改 (DataLoader)

在生成数据集（如 `FIVES_extract.py`）中添加掩码返回逻辑：

```python
def __getitem__(self, idx):
    # 原有逻辑：加载 I^A, I^B
    img_A = load_image(self.cf_paths[idx])
    img_B = load_image(self.oct_fa_paths[idx])
    
    # 新增：加载血管掩码
    mask_vessel = load_mask(self.vessel_mask_paths[idx])  # 二值图，0/1 或 0/255
    mask_vessel = (mask_vessel > 0.5).astype(np.float32)  # 归一化为 0/1
    
    # 生成随机放射变换（旋转 ±45°，平移 ±10%，缩放 0.9~1.1）
    T = generate_random_affine(rotation=(-45, 45), translation=(-0.1, 0.1), scale=(0.9, 1.1))
    
    # 随机翻转（概率 10%）
    flip_h = np.random.rand() < 0.1  # 水平翻转
    flip_v = np.random.rand() < 0.1  # 垂直翻转
    
    # 同步变换图像和掩码（图像不做过滤，保留完整）
    img_B_warped = cv2.warpAffine(img_B, T[:2], (W, H), flags=cv2.INTER_LINEAR)
    mask_B_warped = cv2.warpAffine(mask_vessel, T[:2], (W, H), flags=cv2.INTER_NEAREST)
    
    # 应用翻转（图像和掩码同步）
    if flip_h:
        img_B_warped = cv2.flip(img_B_warped, 1)
        mask_B_warped = cv2.flip(mask_B_warped, 1)
    if flip_v:
        img_B_warped = cv2.flip(img_B_warped, 0)
        mask_B_warped = cv2.flip(mask_B_warped, 0)
    
    # 构建 3x3 单应矩阵（包含翻转）
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = T
    if flip_h:
        H_flip_h = np.array([[-1, 0, W-1], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        H = H_flip_h @ H
    if flip_v:
        H_flip_v = np.array([[1, 0, 0], [0, -1, H-1], [0, 0, 1]], dtype=np.float32)
        H = H_flip_v @ H
    
    return {
        'image0': img_A / 255.0,              # 归一化到 [0, 1]，不做掩码过滤
        'image1': img_B_warped / 255.0,       # 保留完整图像（含背景）
        'image1_origin': img_B / 255.0,       # 原始图像（用于MSE计算）
        'vessel_mask0': mask_vessel,          # 血管掩码（用于损失权重）
        'vessel_mask1': mask_B_warped,        # 变换后的血管掩码
        'T_0to1': H,                          # 真值变换矩阵（3x3）
    }
```

### RoMa 模型修改要点

1. **Backbone 输出修改：**
   * **粗特征编码器：** 在冻结的 DINOv2 (ViT-L/14) 前添加 Modality Adapter（3×3 卷积层）
   * **不要**在 Backbone 中拼接血管掩码，Backbone 只提取纯净的图像特征
   * **精特征编码器：** 使用 VGG19 作为精细定位网络

2. **Transformer 匹配解码器修改：**
   * 实现锚点概率预测（64×64 个锚点）
   * **解剖偏置实现（在注意力机制中引入掩码）：**
   ```python
   # 相似度计算 (Anatomical Biasing)
   # 在 Transformer 的注意力计算中，不要在 Backbone 拼接 mask0
   # 而是在 Transformer 的 Attention Matrix 中引入偏置
   scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)  # [B, H, N_q, N_k]
   
   # 加性偏置引导，不截断梯度
   if vessel_bias is not None:
       # vessel_bias: [B, N_q, N_k] = M_vessel_A.unsqueeze(-1) * M_vessel_B.unsqueeze(-2)
       scores = scores + lambda_weight * vessel_bias.unsqueeze(1)  # broadcast across heads
   
   attn = F.softmax(scores, dim=-1)
   ```

3. **损失函数修改：**
   * **粗级损失（KL 散度 + 梯度保底）：**
   ```python
   # 锚点概率分布与真值分布的 KL 散度
   anchor_probs = model.predict_anchor_probs(feat_A, feat_B)
   gt_anchor_dist = create_gt_anchor_distribution(x_gt_B, K=64*64)
   log_probs = torch.log(anchor_probs + 1e-8)
   kl_loss = F.kl_div(log_probs, gt_anchor_dist, reduction='none').sum(dim=-1)  # [B, N0]
   
   # 梯度保底：给背景留 0.1 的权重
   # 使用 torch.clamp(mask, min=0.1) 作为损失权重系统
   loss_weight = torch.clamp(mask, min=0.1)  # [B, N0]
   loss_coarse = (kl_loss * loss_weight).sum() / (loss_weight.sum() + 1e-8)
   ```
   * **精级损失（Charbonnier）：**
   ```python
   # 广义 Charbonnier 损失
   diff = W(x_A) - x_gt_B
   loss_fine = ((diff ** 2 + s) ** 0.25).mean()
   ```

### 训练超参数建议

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 8e-4 | AdamW 优化器，带 Cosine 衰减 |
| Batch Size | 4-8 | 取决于 GPU 显存（512×512 图像） |
| 训练轮数 | 100-200 epochs | 监控验证集匹配精度 |
| 最小轮数 | 50 epochs | 前50 epoch不触发早停 |
| 早停 patience | 8 | 验证损失连续8个epoch不改善则停止 |
| 粗级阈值 | 1.0 (网格单位) | 1/8 分辨率下的距离阈值 |
| 精级窗口 | $w = 5$ | 5×5 像素窗口 |
| 课程学习 λ | 0 → 0.2 → 0.05 | 阶段1(0-25)→阶段2(25-50)→阶段3(50+) |
| 验证/测试 λ | 0 | 推理时关闭血管掩码偏置 |

---

## 修改总结：引入血管掩码后的完整流程

### 数据流 (Data Pipeline)

1. **输入：** 
   - 固定图 $I^A$ (CF)
   - 待配准图 $I^B$ (OCT/FA)
   - 血管分割掩码 $M^{vessel}$（二值图，表示所有模态共同的血管结构）

2. **数据增强：**
   - 对 $I^B$ 和 $M^{vessel}$ 同步应用随机放射变换 $T$
   - 掩码使用最近邻插值保持二值性
   - 筛选掉掩码面积损失过大的样本

3. **特征提取：**
   - 双流 Backbone 提取粗/精两级特征
   - **不在 Backbone 中拼接掩码**，保持特征的纯净性
   - 掩码单独传递给 Transformer

4. **解剖偏置注意力：**
   - 在 Transformer 的注意力分数矩阵上应用**加性偏置 (Additive Bias)**
   - 通过 $scores = scores + \lambda \cdot vessel\_bias$ 引导注意力向血管区域倾斜
   - **不截断梯度**，非血管区域保留微弱响应，确保梯度连续性

5. **损失计算：**
   - **粗级：** 仅在血管区域的GT点对上计算损失，血管边缘点权重更高
   - **精级：** 检查窗口内血管占比，过滤低质量窗口，亚像素计算仅考虑血管像素

6. **推理与配准：**
   - 输入待配准图像对及其掩码
   - RoMa 输出稠密变形场和匹配点对
   - 基于置信度进行空间均匀化采样（8×8 分区）
   - 掩码过滤非血管匹配点
   - RANSAC 求解放射矩阵
   - 应用变换完成配准

### 关键创新点

| 模块 | 原始 RoMa | 引入掩码后的改进 |
|------|-----------|----------------|
| **输入** | 仅图像对 | 图像对 + 血管掩码 |
| **粗特征编码器** | 冻结 DINOv2 | **DINOv2 + Modality Adapter (不拼接掩码)** |
| **精特征编码器** | VGG19 | VGG19（保持不变） |
| **匹配解码器** | 锚点概率预测 | **掩码引导的锚点概率预测** |
| **注意力机制** | 全局无约束匹配 | **解剖偏置 (+λ·vessel_bias)**，加性偏置，软性引导 |
| **损失函数** | KL 散度 + Charbonnier | **梯度保底 (torch.clamp(mask, min=0.1))**，保证早期收敛 |
| **训练策略** | 单一阶段 | **三阶段课程学习**：阶段1(λ=0, 0-25 epoch)→阶段2(λ=0.2, 25-50)→阶段3(λ=0.05, 50+) |
| **RANSAC** | 均匀采样 | **空间均匀化 (8×8 Binning)**，防退化 |
| **旋转支持** | 有限角度 | **±90° 支持**，极大尺度旋转 |

### 预期效果

通过引入"解剖引导"策略和三阶段课程学习，模型将能够：

1. **处理极大角度旋转（±90°）和翻转：** RoMa 的 Transformer 解码器在不使用位置编码的情况下非常健壮。由于 DINOv2 提取的是语义级别的特征，即便血管旋转 90°，其局部的拓扑结构特征在 DINO 空间中依然保持高度相似。
2. **处理大位置偏移：** 即使点对落在背景上，微弱的梯度保底（ε=0.1）也能推动模型学习大尺度的旋转信息。
3. **避免模型退化：** 空间均匀化 (8×8 Spatial Binning) 强制匹配点散开，防止解算的放射矩阵退化为单位矩阵 $I$。
4. **提升关键点质量：** 通过三阶段课程学习，模型从全图感知逐步过渡到血管区域关注，在训练后期自动聚焦于血管交叉点和分叉点。
5. **加速收敛与稳定训练：** 
   - 阶段1（λ=0）：快速学习大尺度几何变换
   - 阶段2（λ=0.2）：引入弱血管引导，平衡全局和局部
   - 阶段3（λ=0.05）：精细调优，避免过度依赖掩码
   - 前50 epoch保护期：确保模型充分学习基础特征后再启用早停
6. **推理时的鲁棒性：** 测试时设置 λ=0，确保模型不依赖掩码，具有更好的泛化能力。
7. **解决 LoFTR 无法处理的问题：** 能够处理 30° 以上大旋转问题，甚至支持 ±90° 的极大尺度旋转。

---

## 实施阶段建议 (Roadmap)

### 第一阶段：Backbone 适配
* **核心任务：** 冻结 DINOv2，添加 Modality Adapter，使用 VGG19 作为精细层。
* **预期目标：** 消除 CF 与 OCT/FA 的纹理鸿沟。

### 第二阶段：解剖引导训练
* **核心任务：** 引入 $M^{vessel}$ 加性偏置和分类损失 $\mathcal{L}_{coarse}$。
* **预期目标：** 实现对血管分叉点的强力锁定。

### 第三阶段：鲁棒性强化
* **核心任务：** 全量放射变换增强（旋转、翻转、缩放），旋转范围扩大到 ±90°。
* **预期目标：** 解决 LoFTR 无法处理的 30° 以上大旋转问题。

### 第四阶段：矩阵解算
* **核心任务：** 稠密 Warp → 空间均匀采样（8×8 分区）→ RANSAC。
* **预期目标：** 输出最终的放射变换矩阵 $M$。

---

## 训练与验证配置总结

### 训练配置（train_onGen.py / train_onReal.py）

**课程学习策略：**
```python
# 阶段1 (0-25 epoch): λ = 0.0 (全图感知期)
if current_epoch < 25:
    lambda_vessel = 0.0
    
# 阶段2 (25-50 epoch): λ = 0.2 (软掩码引导期)
elif current_epoch < 50:
    lambda_vessel = 0.2
    
# 阶段3 (50+ epoch): λ = 0.05 (精度冲刺期)
else:
    lambda_vessel = 0.05
```

**早停配置：**
- `min_epochs = 50`：前50 epoch不触发早停
- `patience = 8`：验证损失连续8个epoch不改善则停止
- `monitor = 'val_mse'`：监控验证集MSE

### 验证/测试配置（validation_step / test_onReal.py）

**推理时配置：**
```python
# 完全关闭血管掩码偏置
lambda_vessel = 0.0
```

**原因：**
1. 避免对掩码的过度依赖
2. 测试模型的纯特征匹配能力
3. 确保在真实场景（可能没有精确掩码）下也能工作

### 空间均匀化采样实现

**设计理念：端到端学习**
- ❌ 不在数据加载时对图像进行有效区域过滤（保留完整图像含背景）
- ❌ 不在匹配点选择时根据掩码丢弃点（让模型自己学习判断）
- ❌ 不在空间均匀化时限制网格范围（全图 8×8 均匀采样）
- ✅ 只在计算评估指标（MSE）时使用 `filter_valid_area` 过滤背景噪声

**在 RANSAC 之前应用：**
```python
def spatial_binning(pts0, pts1, mconf, img_size, grid_size=8, top_k=5):
    """
    空间均匀化采样：将全图划分为 grid_size x grid_size 个分区
    每个分区选取置信度最高的 top_k 个点
    
    注意：不使用掩码过滤，让模型端到端学习血管区域的重要性
    """
    H, W = img_size
    cell_h = H / grid_size
    cell_w = W / grid_size
    
    # 计算每个点所属的格子
    grid_y = (pts0[:, 1] / cell_h).astype(int)
    grid_x = (pts0[:, 0] / cell_w).astype(int)
    grid_y = np.clip(grid_y, 0, grid_size - 1)
    grid_x = np.clip(grid_x, 0, grid_size - 1)
    
    pts0_binned = []
    pts1_binned = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (grid_y == i) & (grid_x == j)
            if mask.sum() == 0:
                continue
            
            # 选取置信度最高的 top_k 个点
            indices = np.where(mask)[0]
            conf_in_cell = mconf[indices]
            
            if len(indices) > top_k:
                top_indices = np.argsort(-conf_in_cell)[:top_k]
                indices = indices[top_indices]
            
            pts0_binned.append(pts0[indices])
            pts1_binned.append(pts1[indices])
    
    return np.concatenate(pts0_binned), np.concatenate(pts1_binned)

# 使用示例
mkpts0_binned, mkpts1_binned = spatial_binning(
    mkpts0, mkpts1, mconf, 
    img_size=(518, 518), 
    grid_size=8, 
    top_k=5
)

# 然后使用 RANSAC
H, mask = cv2.findHomography(mkpts0_binned, mkpts1_binned, cv2.RANSAC, 3.0)
```

**效果：**
- 典型情况：1000+ 匹配点 → 最多 320 个均匀分布的点（64 格 × 5 点）
- 极大提高放射矩阵求解的稳定性和精度
- 通过课程学习中的注意力偏置，模型自动学习关注血管区域

### 一致性保证

- ✅ `train_onGen.py` 和 `train_onReal.py` 使用相同的课程学习策略
- ✅ `train_onGen.py`、`train_onReal.py`、`test_onReal.py` 的验证/测试过程统一设置 λ=0
- ✅ 所有训练脚本的早停配置一致（min_epochs=50, patience=8）
- ✅ 所有脚本在 RANSAC 前使用相同的空间均匀化采样（8×8网格，每格最多5个点，**全图无掩码约束**）
- ✅ 端到端学习：图像不做预过滤，匹配点不做硬过滤，只在评估指标计算时使用有效区域过滤

---

## 重要修改记录（2025-01-21）

### 设计理念变更：从"硬约束"到"端到端学习"

**修改前的问题：**
- 数据加载时对图像进行有效区域过滤（背景置零）
- 匹配点选择时根据掩码硬过滤（丢弃背景点）
- 空间均匀化时使用掩码限制网格范围
- 导致模型过度依赖人工规则，缺乏端到端学习能力

**修改后的方案：**
- ❌ **移除图像预过滤**：保留完整图像（含背景），让模型看到真实数据
- ❌ **移除匹配点硬过滤**：不根据掩码丢弃点，让模型的置信度预测决定点的质量
- ❌ **移除空间均匀化的掩码约束**：全图 8×8 均匀采样，不限制网格范围
- ✅ **保留评估时的有效区域过滤**：MSE 等指标只在眼底圆形区域内计算

**核心保留的软引导机制：**
1. **Transformer 注意力偏置**（$\lambda \cdot vessel\_bias$）：通过课程学习动态调整
2. **损失函数梯度保底**（`torch.clamp(mask, min=0.1)`）：即使背景区域也保留 0.1 的权重
3. **血管掩码传递**（`vessel_mask0/1`）：用于损失权重计算，但不做硬过滤

**预期效果：**
- 模型自动学习血管区域的重要性（通过注意力偏置和损失权重）
- 模型对背景区域的匹配给出低置信度（通过端到端学习）
- 提升泛化能力（避免对人工规则的过度依赖）
- 保持评估的公平性（只在有效区域计算指标）