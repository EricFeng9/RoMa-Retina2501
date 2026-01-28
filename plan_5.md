# RoMa Backbone 改造计划：集成 SuperPoint 特征提取

**目标**：在 RoMa 模型的粗特征提取（Coarse Feature Extraction）阶段，引入 **SuperPoint** 网络来替换现有的 DINOv2。

**背景与动机**：
*   **现状**：当前使用 DINOv2 提取语义特征。DINOv2 擅长全局语义理解（对象分类），但对细粒度的几何结构（如血管分叉、交叉点）不够敏感。
*   **改进**：SuperPoint 专为角点（Corner）和关键点检测设计。眼底血管网络中的分叉点和交叉点在几何上等同于“角点”。
*   **预期**：利用 SuperPoint 强悍的几何特征提取能力，提升模型对血管拓扑结构的感知力，从而改善匹配精度。

---

## 1. 新增 `SuperPointEncoder` 类

我们需要在 `src/loftr/backbone/roma_backbone_superPoints.py` 中实现一个轻量级的 SuperPoint 编码器。

### 功能需求
*   **结构**：复现 SuperPoint 的 Backbone 结构（VGG-style Encoder）。
*   **输入**：灰度图像 `[B, 1, H, W]`。
*   **输出**：稠密描述子 `[B, 256, H/8, W/8]`。
*   **权重加载**：支持加载官方预训练权重（`superpoint_v1.pth`）。
*   **冻结**：默认冻结权重，作为特征提取器使用。

### 代码草稿 (待实现)
```python
class SuperPointEncoder(nn.Module):
    def __init__(self, freeze=True, weight_path=None):
        # ... 定义 Conv1_x 到 Conv4_x ...
        # ... 定义 Descriptor Head (ConvDa, ConvDb) ...
        # ... 加载权重 ...
    
    def forward(self, x):
        # ... 提取 H/8 特征 ...
        # ... L2 Normalization (可选) ...
        return desc
```

## 2. 修改 `RoMaBackbone` 逻辑

在 `RoMaBackbone` 类中集成新模块，并在 `forward` 过程中实现特征融合。

### 修改点
1.  **初始化 (`__init__`)**:
    *   增加配置开关（如 `config.ROMA.COARSE_TYPE = 'superpoint'`）。
    *   当启用 SuperPoint 时，初始化 `SuperPointEncoder` 替代 `CoarseEncoder(DINOv2)`。
    *   更新 `self.out_channels` 或者 Transformer 的输入投影层维度（DINOv2 是 384，SuperPoint 是 256，融合 VGG 后总维度变化）。

2.  **前向传播 (`forward`)**:
    *   **提取特征**: 调用 `SuperPointEncoder` 得到 `feat_spp`。
    *   **分辨率对齐**: 
        *   SuperPoint 输出通常为 $1/8$ 分辨率。
        *   VGG 的粗特征 `feat_coarse_vgg` 如果是 $1/4$，则需要下采样到 $1/8$。
        *   **决策**: 统一对齐到 $1/8$ 分辨率。这有利于降低 Transformer 计算量（序列长度减少 4 倍），且 $1/8$ 对粗匹配已足够。
    *   **特征融合 (已移除)**:
        *   ~~使用 `torch.cat([feat_vgg_aligned, feat_spp], dim=1)`。~~
        *   **修正**: 为了保证粗特征的纯净性和完全冻结状态，**不融合 VGG 特征**。直接使用 `feat_spp` 作为粗特征输入 Transformer。
        *   这样可以避免 VGG 在微调过程中导致的特征漂移（Feature Drift）和边缘效应。

## 3. 配置与权重管理

需要调整配置文件和权重文件路径。

*   **权重文件**: 需要下载 `superpoint_v1.pth` 并放置在项目目录中（如 `weights/`）。
*   **Config 更新**:
    *   在 `configs/roma_multimodal.py` 中添加各类参数。
    ```python
    cfg.ROMA.COARSE_TYPE = 'superpoint' # 选项: 'dinov2', 'superpoint'
    cfg.ROMA.SUPERPOINT_PATH = 'weights/superpoint_v1.pth'
    ```

## 4. 维度匹配检查 (Sanity Check)

实施前需确认不同模块的维度兼容性：

| 模块 | 原始维度 (Dim) | 原始分辨率 (Scale) | 目标操作 |
| :--- | :--- | :--- | :--- |
| **VGG Coarse** | 256 | 1/4 (部分实现取1/8) | 下采样至 1/8 |
| **DINOv2** | 384 | 1/14 (Patch) -> 插值 | **移除** |
| **SuperPoint** | 256 | 1/8 | 保持 |
| **Transformer** | Input Proj | - | 修改 Input Proj 接受 256 dim (仅 SuperPoint) |

**注意**：RoMa 的 Transformer 输入层通常有一个 `input_proj` 层。由于我们移除了 VGG 融合，Backbone 输出维度变为 **256 (SuperPoint only)**。我们需要确保 `RoMaTransformer` 的初始化代码能正确计算 `in_channels`，不再加上 `d_vgg`。

## 5. 执行步骤

1.  **创建文件**: 实现 `SuperPointEncoder` 类。
2.  **修改 Backbone**: 集成上述类。
3.  **解决维度冲突**: 检查 `roma_model.py` 或 `loftr_module` 中 Transformer 的输入维度定义，确保它动态适配 Backbone 的输出。
4.  **运行测试**: 使用 `test_backbone.py` (需创建) 或简单脚本跑通 `forward`，打印 Shape。
