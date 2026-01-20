"""
RoMa 多模态配准配置文件
"""
from yacs.config import CfgNode as CN

_CN = CN()

# ---------------------------------------------------------------------------- #
# RoMa 模型配置
# ---------------------------------------------------------------------------- #
_CN.ROMA = CN()
_CN.ROMA.USE_DINOV2 = True  # 是否使用 DINOv2 (推荐启用以支持大角度旋转)
_CN.ROMA.DINOV2_MODEL = 'dinov2_vits14'  # DINOv2 模型类型
_CN.ROMA.DINOV2_PATH = None  # 本地权重路径 (可选，如: '/path/to/dinov2_vits14.pth')
_CN.ROMA.D_MODEL = 256  # Transformer 维度
_CN.ROMA.N_HEADS = 8  # 注意力头数
_CN.ROMA.N_LAYERS = 4  # Transformer 层数
_CN.ROMA.LAMBDA_VESSEL = 1.0  # 血管偏置权重
_CN.ROMA.NUM_ANCHORS = 4096  # 锚点数量 (64*64)
_CN.ROMA.CONF_THRESH = 0.01  # 置信度阈值
_CN.ROMA.TOP_K = 1000  # 最多保留的匹配点数
_CN.ROMA.FINE_WINDOW_SIZE = 5  # 精细匹配窗口大小

# 损失函数配置
_CN.ROMA.LOSS = CN()
_CN.ROMA.LOSS.WEIGHT_COARSE = 1.0  # 粗级损失权重
_CN.ROMA.LOSS.WEIGHT_FINE = 1.0  # 精级损失权重
_CN.ROMA.LOSS.EPSILON = 0.1  # 梯度保底参数
_CN.ROMA.LOSS.SMOOTH_PARAM = 0.01  # Charbonnier 平滑参数
_CN.ROMA.LOSS.ANCHOR_SIGMA = 0.02  # 锚点高斯分布标准差

# ---------------------------------------------------------------------------- #
# 训练配置
# ---------------------------------------------------------------------------- #
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64  # 标准 Batch Size
_CN.TRAINER.CANONICAL_LR = 8e-4  # 标准学习率
_CN.TRAINER.TRUE_LR = 8e-4
_CN.TRAINER.TRUE_BATCH_SIZE = 4
_CN.TRAINER.SEED = 66
_CN.TRAINER.PLOT_MODE = 'evaluation'

# 优化器配置
_CN.TRAINER.OPTIMIZER = 'adamw'
_CN.TRAINER.ADAM_DECAY = 0.0
_CN.TRAINER.WARMUP_STEP = 1875
_CN.TRAINER.WARMUP_RATIO = 0.1
_CN.TRAINER.WARMUP_TYPE = 'linear'
_CN.TRAINER.SCHEDULER = 'MultiStepLR'
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSINE_DECAY_MAX_EPOCH = 100

# 梯度裁剪
_CN.TRAINER.GRAD_CLIP = 1.0
_CN.TRAINER.GRAD_CLIP_TYPE = 'norm'

# ---------------------------------------------------------------------------- #
# 数据集配置
# ---------------------------------------------------------------------------- #
_CN.DATASET = CN()
_CN.DATASET.MGDPT_IMG_RESIZE = 518  # 图像大小 (需为 14 的倍数)
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

def get_cfg_defaults():
    """获取默认配置"""
    return _CN.clone()
