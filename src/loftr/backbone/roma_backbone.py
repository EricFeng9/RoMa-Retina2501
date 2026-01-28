"""
RoMa Backbone: 双流特征提取器 (粗特征 + 精特征)
- 粗特征: 冻结的 DINOv2 + Modality Adapter
- 精特征: VGG19
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ModalityAdapter(nn.Module):
    """模态适配器：将眼底图像映射到 DINOv2 特征空间"""
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] 灰度图
        Returns:
            [B, 3, H, W] RGB格式
        """
        return self.conv(x)


class CoarseEncoder(nn.Module):
    """粗特征编码器: Modality Adapter + 冻结的 DINOv2"""
    def __init__(self, dinov2_model='dinov2_vits14', freeze_dinov2=True, dinov2_path=None):
        super().__init__()
        self.adapter = ModalityAdapter(in_channels=1, out_channels=3)
        
        # 加载 DINOv2 (支持本地路径和在线下载两种方式)
        try:
            if dinov2_path and os.path.exists(dinov2_path):
                # 从本地加载权重，但结构仍需通过 torch.hub 获取（不下载权重）
                print(f"从本地路径加载 DINOv2 权重: {dinov2_path}")
                self.dinov2 = torch.hub.load('facebookresearch/dinov2', dinov2_model, pretrained=False)
                state_dict = torch.load(dinov2_path, map_location='cpu')
                # 兼容某些保存方式中包含 'model' 或 'state_dict' 的情况
                if 'model' in state_dict: state_dict = state_dict['model']
                if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
                self.dinov2.load_state_dict(state_dict)
            else:
                # 在线下载 (需要网络连接)
                print(f"在线下载 DINOv2: {dinov2_model}")
                self.dinov2 = torch.hub.load('facebookresearch/dinov2', dinov2_model)
        except Exception as e:
            # 如果加载失败，使用 ResNet18 替代（性能会下降）
            print(f"⚠️ DINOv2 加载失败 ({e})，使用 ResNet18 替代（性能会下降）")
            resnet = models.resnet18(pretrained=True)
            self.dinov2 = nn.Sequential(*list(resnet.children())[:-2])
        
        if freeze_dinov2:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()
    
    def forward(self, x, vessel_mask=None):
        """
        Args:
            x: [B, 1, H, W] 灰度图
            vessel_mask: [B, 1, H, W] 血管掩码（不在此处使用，仅为接口兼容）
        Returns:
            feat: [B, C, H_patch, W_patch] 粗特征
        
        注意：按照图片要求，不在 Backbone 中拼接掩码，保持特征纯净性。
        掩码将在 Transformer 的注意力机制中通过加性偏置引入。
        """
        # 模态适配: 使用可学习的 Adapter 将灰度图映射到 DINOv2 输入空间
        x_adapted = self.adapter(x)
        
        # DINOv2 特征提取
        with torch.no_grad() if not self.training else torch.enable_grad():
            # 使用 forward_features 获取 patch tokens
            feat_out = self.dinov2.forward_features(x_adapted)
            if isinstance(feat_out, dict):
                feat = feat_out['x_norm_patchtokens'] # [B, N, D]
            else:
                # 剥离 CLS
                feat = feat_out[:, 1:] 
                
            B, N, D = feat.shape
            H_patch = W_patch = int(N**0.5)
            feat = feat.permute(0, 2, 1).reshape(B, D, H_patch, W_patch)
        
        return feat, x_adapted


class FineEncoder(nn.Module):
    """精特征编码器: VGG19"""
    def __init__(self, pretrained=True):
        super().__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        
        # 提取 VGG19 的卷积层
        self.features = vgg19.features
        
        # 定义两个输出层：粗级 (1/8) 和精级 (1/4)
        # VGG19 结构: pool 层索引: 4(1/2), 9(1/4), 18(1/8), 27(1/16)
        self.coarse_layers = nn.Sequential(*list(self.features.children())[:18])  # 到第3个pool前，1/4 (256ch)
        self.fine_layers = nn.Sequential(*list(self.features.children())[:10])    # 到第2个pool，1/4 (128ch)
    
    def forward(self, x, vessel_mask=None):
        """
        Args:
            x: [B, 1, H, W] 灰度图
            vessel_mask: [B, 1, H, W] 血管掩码（不在此处使用，仅为接口兼容）
        Returns:
            feat_coarse: [B, 256, H/4, W/4] 粗级特征
            feat_fine: [B, 128, H/4, W/4] 精级特征
        
        注意：不在 Backbone 中拼接掩码，保持特征纯净性。
        """
        # VGG 期望 3 通道输入
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        feat_fine = self.fine_layers(x)
        feat_coarse = self.coarse_layers(x)
        
        return feat_coarse, feat_fine


class RoMaBackbone(nn.Module):
    """
    RoMa 双流 Backbone:
    - 粗特征流: Modality Adapter + DINOv2 (冻结)
    - 精特征流: VGG19
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 粗特征编码器 (DINOv2)
        use_dinov2 = config.get('ROMA', {}).get('USE_DINOV2', False)
        if use_dinov2:
            self.coarse_encoder = CoarseEncoder(
                dinov2_model=config.get('ROMA', {}).get('DINOV2_MODEL', 'dinov2_vits14'),
                freeze_dinov2=True,
                dinov2_path=config.get('ROMA', {}).get('DINOV2_PATH', None)  # 支持本地权重路径
            )
        else:
            # 如果不使用 DINOv2，使用 VGG 作为粗特征
            self.coarse_encoder = None
        
        # 精特征编码器 (VGG19)
        self.fine_encoder = FineEncoder(pretrained=True)
        
    def forward(self, x, vessel_mask=None):
        """
        Args:
            x: [B, 1, H, W] 灰度图
            vessel_mask: [B, 1, H, W] 血管掩码（不在此处使用，仅为接口兼容）
        Returns:
            feat_c: [B, D, H/8, W/8] 粗特征
            feat_f: [B, d, H/4, W/4] 精特征
        
        注意：按照图片要求，不在 Backbone 中拼接掩码。
        Backbone 只负责提取纯净的图像特征，掩码将在 Transformer 的注意力机制中使用。
        """
        # 精特征 (VGG19) - 不传入 vessel_mask
        feat_coarse_vgg, feat_fine = self.fine_encoder(x, None)
        
        # 粗特征 (DINOv2 或 VGG) - 不传入 vessel_mask
        x_adapted = None
        if self.coarse_encoder is not None:
            feat_coarse_dino, x_adapted = self.coarse_encoder(x, None)
            # 统一分辨率：将 VGG 粗特征对齐到 DINOv2 的 Patch 分辨率 (如 37x37)
            # 这样做可以减小 Transformer 的序列长度，防止 vessel_bias 矩阵过大导致 OOM
            if feat_coarse_vgg.shape[2:] != feat_coarse_dino.shape[2:]:
                feat_coarse_vgg = F.interpolate(feat_coarse_vgg, size=feat_coarse_dino.shape[2:], 
                                                mode='bilinear', align_corners=False)
            feat_c = torch.cat([feat_coarse_vgg, feat_coarse_dino], dim=1)
        else:
            feat_c = feat_coarse_vgg
        
        return feat_c, feat_fine, x_adapted
