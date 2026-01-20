# RoMa 多模态眼底图像配准系统

基于 RoMa (Robust Dense Feature Matching) 架构的多模态眼底图像配准实现。

## 项目结构

```
RoMa/
├── src/
│   ├── loftr/
│   │   ├── backbone/
│   │   │   └── roma_backbone.py          # 双流特征提取器 (DINOv2 + VGG19)
│   │   ├── loftr_module/
│   │   │   └── roma_transformer.py       # Transformer 匹配解码器
│   │   └── roma_model.py                 # RoMa 主模型
│   ├── losses/
│   │   └── roma_loss.py                  # 损失函数 (KL散度 + Charbonnier)
│   └── lightning/
│       └── lightning_roma.py             # PyTorch Lightning 训练模块
├── experiments/
│   └── train_roma_multimodal.py          # 训练脚本
├── configs/
│   └── roma_multimodal.py                # 配置文件
├── scripts/
│   └── train_roma.sh                     # 启动脚本
└── data/
    └── FIVES_extract/
        └── FIVES_extract.py              # 数据集加载器
```

## 核心创新

### 1. 解剖引导的特征匹配

- **血管掩码增强**: 通过血管分割掩码引导模型关注血管结构
- **解剖偏置注意力**: Transformer 中加入血管权重矩阵，优先匹配血管交叉点和分叉点
- **梯度保底机制**: 给非血管区域保留 ε=0.1 的权重，确保能学习大尺度旋转

### 2. 双流特征提取

- **粗特征流**: 冻结的 DINOv2 (可选) + VGG19
- **精特征流**: VGG19 多尺度特征
- **模态适配器**: 轻量级卷积网络，将眼底图像映射到预训练模型特征空间

### 3. 锚点概率预测

- **回归转分类**: 将配准问题建模为锚点概率分布
- **KL 散度损失**: 最小化预测分布与真值分布的差异
- **Charbonnier 精细损失**: 鲁棒回归，抑制异常值

### 4. 空间均匀化采样

- **8×8 分区策略**: 强制匹配点在空间上均匀分布
- **RANSAC 求解**: 鲁棒估计放射变换矩阵
- **防退化机制**: 避免模型输出单位矩阵

### 5. 课程学习

- **阶段1 (前20% epochs)**: 全图感知期，λ_vessel = 0
- **阶段2 (中间50% epochs)**: 软掩码引导期，λ_vessel = 0.5
- **阶段3 (最后30% epochs)**: 精度冲刺期，λ_vessel = 1.0

## 使用方法

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision pytorch-lightning
pip install opencv-python numpy loguru yacs tensorboard
```

### 2. 数据集准备

数据集路径硬编码在 `experiments/train_roma_multimodal.py` 中：

```python
DATA_ROOT = "/data/student/Fengjunming/LoFTR/data/FIVES_extract"
```

确保数据集格式符合 `data/FIVES_extract/FIVES_extract.py` 的要求。

### 3. 启动训练

#### 方式1: 使用启动脚本（推荐）

```bash
# 修改 scripts/train_roma.sh 中的参数
bash scripts/train_roma.sh
```

#### 方式2: 直接运行 Python 脚本

```bash
python experiments/train_roma_multimodal.py --mode cffa --name roma_cffa_v1 --batch_size 4 --gpus 1
```

### 4. 训练参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--mode` | 配准模式 (cffa/cfoct/octfa/cfocta) | cffa |
| `--name` | 实验名称 | roma_multimodal_fives |
| `--batch_size` | 每个GPU的批量大小 | 4 |
| `--num_workers` | 数据加载线程数 | 8 |
| `--img_size` | 图像输入尺寸 | 512 |
| `--vessel_sigma` | 血管高斯软掩码的σ | 6.0 |
| `--max_epochs` | 最大训练轮数 | 100 |
| `--gpus` | 使用的GPU | 1 |

### 5. 输出结果

训练结果保存在 `results/roma_{mode}/{name}/` 目录下：

```
results/roma_cffa/roma_cffa_v1/
├── best_checkpoint/
│   ├── model.ckpt              # 最优模型权重
│   └── log.txt                 # 最优指标记录
├── latest_checkpoint/
│   ├── model.ckpt              # 最新模型权重
│   └── log.txt                 # 最新指标记录
├── epoch1/
│   ├── sample1/
│   │   ├── fix.png             # 固定图
│   │   ├── moving.png          # 形变后的移动图
│   │   ├── moving_origin.png   # 原始移动图
│   │   ├── moving_result.png   # 配准结果
│   │   ├── chessboard.png      # 棋盘图
│   │   ├── matches.png         # 匹配可视化
│   │   ├── mask0_vessel.png    # 血管掩码
│   │   └── mask1_vessel_warped.png
│   └── ...
└── log.txt                     # 完整训练日志
```

## 模型配置

核心配置在 `configs/roma_multimodal.py` 中：

```python
# Transformer 配置
D_MODEL = 256         # Transformer 维度
N_HEADS = 8           # 注意力头数
N_LAYERS = 4          # Transformer 层数
LAMBDA_VESSEL = 1.0   # 血管偏置权重

# 损失函数配置
WEIGHT_COARSE = 1.0   # 粗级损失权重
WEIGHT_FINE = 1.0     # 精级损失权重
EPSILON = 0.1         # 梯度保底参数
ANCHOR_SIGMA = 0.02   # 锚点高斯分布标准差
```

## 与 LoFTR 的区别

| 特性 | LoFTR | RoMa |
|-----|-------|------|
| **特征提取** | ResNet | VGG19 + DINOv2 (可选) |
| **匹配机制** | 直接预测匹配点 | 锚点概率分布 |
| **损失函数** | 分类损失 + 回归损失 | KL散度 + Charbonnier |
| **血管引导** | 无 | 解剖偏置注意力 |
| **课程学习** | 无 | 三阶段动态调整 |
| **旋转支持** | ±30° | ±90° |
| **配准方式** | 像素级变形场 | 放射矩阵 (RANSAC) |

## 评估指标

训练过程中会记录以下指标：

- **loss_c**: 粗级损失 (KL散度)
- **loss_f**: 精级损失 (Charbonnier)
- **auc@5/10/20**: 对极误差小于阈值的匹配点比例
- **mse_viz**: 配准结果的均方误差 (可视化用)

## 已知问题与限制

1. **DINOv2 加载**: 当前默认关闭 DINOv2，仅使用 VGG19。如需启用，需确保网络连接或手动下载预训练权重。

2. **精细匹配**: 当前精细匹配模块为简化版，直接返回粗匹配结果。完整实现需添加局部窗口匹配。

3. **内存占用**: Transformer 在高分辨率图像上内存占用较大，建议使用 512×512 分辨率。

4. **批量大小**: 受限于 GPU 内存，推荐 batch_size=4。如果 OOM，可减小到 2。

## 调试建议

1. **匹配点过少**: 降低 `CONF_THRESH`，增加 `TOP_K`
2. **H_est 为单位矩阵**: 增加训练轮数，检查损失是否收敛
3. **MSE 过高**: 调整 `LAMBDA_VESSEL`，增强血管引导
4. **训练不稳定**: 降低学习率，增加 warmup 步数

## 参考文献

1. RoMa: Robust Dense Feature Matching (CVPR 2024)
2. DINOv2: Learning Robust Visual Features (ICCV 2023)
3. LoFTR: Detector-Free Local Feature Matching (CVPR 2021)

## 联系方式

如有问题，请联系项目维护者或提交 Issue。
