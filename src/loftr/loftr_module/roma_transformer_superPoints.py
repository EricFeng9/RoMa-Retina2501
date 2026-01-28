"""
RoMa Transformer 匹配解码器
- 实现锚点概率预测
- 实现解剖偏置注意力机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionEncodingSine(nn.Module):
    """
    标准的正弦位置编码 (Sinusoidal Positional Encoding)
    这对 Transformer 感知空间结构至关重要，否则它就是 "Permutation Invariant" 的，
    导致所有匹配点都因为边界效应塌缩到图像边缘。
    """
    def __init__(self, d_model, max_shape=(256, 256), temp=10000):
        super().__init__()
        self.d_model = d_model
        self.temp = temp
        self.scale = 2 * math.pi

        dim_t = torch.arange(self.d_model // 2, dtype=torch.float32)
        dim_t = self.temp ** (2 * (dim_t // 2) / (self.d_model // 2))
        self.register_buffer("dim_t", dim_t)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            pos: [1, C, H, W] (broadcastable)
        """
        B, C, H, W = x.shape
        # 生成网格坐标
        y_embed, x_embed = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij")
        y_embed = y_embed.float().unsqueeze(0) # [1, H, W]
        x_embed = x_embed.float().unsqueeze(0) # [1, H, W]

        # 归一化并缩放
        eps = 1e-6
        y_embed = y_embed / (H + eps) * self.scale
        x_embed = x_embed / (W + eps) * self.scale

        # 计算正弦编码
        dim_t = self.dim_t
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # [1, H, W, C/2] -> [1, C/2, H, W]
        pos_x = pos_x.permute(0, 3, 1, 2)
        pos_y = pos_y.permute(0, 3, 1, 2)
        
        # 拼接
        pos = torch.cat((pos_y, pos_x), dim=1) # [1, C, H, W]
        return pos


class AnatomicalBiasedAttention(nn.Module):
    """解剖偏置的多头注意力机制"""
    def __init__(self, d_model, n_heads, lambda_vessel=1.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.lambda_vessel = lambda_vessel
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, vessel_bias=None):
        """
        Args:
            query: [B, N_q, D]
            key: [B, N_k, D]
            value: [B, N_k, D]
            vessel_bias: [B, N_q, N_k] 血管权重矩阵 (可选)
        Returns:
            out: [B, N_q, D]
            attn: [B, N_heads, N_q, N_k]
        
        注意：按照图片要求，在 Attention Matrix 中使用加性偏置 (Additive Bias)。
        不在 Backbone 拼接 image0 和 mask0，而是在这里通过加性偏置引导注意力。
        """
        B, N_q, D = query.shape
        N_k = key.shape[1]
        
        # 线性投影
        Q = self.q_proj(query).reshape(B, N_q, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, N_q, d]
        K = self.k_proj(key).reshape(B, N_k, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).reshape(B, N_k, self.n_heads, self.d_head).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, H, N_q, N_k]
        
        # 加入血管偏置 (Additive Bias) - 按照图片要求
        # 公式: scores = scores + λ * vessel_bias
        if vessel_bias is not None:
            # vessel_bias: [B, N_q, N_k] -> [B, 1, N_q, N_k] (broadcast across heads)
            scores = scores + self.lambda_vessel * vessel_bias.unsqueeze(1)
        
        # Softmax 归一化
        attn = F.softmax(scores, dim=-1)
        
        # 加权求和
        out = torch.matmul(attn, V)  # [B, H, N_q, d]
        out = out.transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)
        
        return out, attn


class TransformerLayer(nn.Module):
    """Transformer 层：Self-Attention + Cross-Attention + FFN"""
    def __init__(self, d_model, n_heads, lambda_vessel=1.0):
        super().__init__()
        self.self_attn = AnatomicalBiasedAttention(d_model, n_heads, lambda_vessel)
        self.cross_attn = AnatomicalBiasedAttention(d_model, n_heads, lambda_vessel)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x0, x1, vessel_bias=None):
        """
        Args:
            x0: [B, N0, D] 图像0的特征
            x1: [B, N1, D] 图像1的特征
            vessel_bias: [B, N0, N1] 血管偏置矩阵
        Returns:
            x0_out: [B, N0, D]
            x1_out: [B, N1, D]
        """
        # Self-Attention
        x0_norm = self.norm1(x0)
        x0 = x0 + self.self_attn(x0_norm, x0_norm, x0_norm)[0]
        
        x1_norm = self.norm1(x1)
        x1 = x1 + self.self_attn(x1_norm, x1_norm, x1_norm)[0]
        
        # Cross-Attention
        x0_norm = self.norm2(x0)
        x1_norm = self.norm2(x1)
        x0 = x0 + self.cross_attn(x0_norm, x1_norm, x1_norm, vessel_bias)[0]
        x1 = x1 + self.cross_attn(x1_norm, x0_norm, x0_norm, 
                                   vessel_bias.transpose(-2, -1) if vessel_bias is not None else None)[0]
        
        # FFN
        x0 = x0 + self.ffn(self.norm3(x0))
        x1 = x1 + self.ffn(self.norm3(x1))
        
        return x0, x1


class AnchorPredictor(nn.Module):
    """锚点概率预测器"""
    def __init__(self, d_model, num_anchors=64*64):
        super().__init__()
        self.num_anchors = num_anchors
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_anchors)
        )
        
    def forward(self, feat0, feat1):
        """
        Args:
            feat0: [B, N0, D] 图像0的特征 (已通过 Cross-Attention 融合图像1信息)
            feat1: [B, N1, D] 图像1的特征
        Returns:
            anchor_probs: [B, N0, K] 每个点的锚点概率分布
        """
        # 直接基于 contextualized feat0 预测锚点分布
        # 避免构造 [B, N0, N1, K] 的巨大张量
        anchor_logits = self.predictor(feat0)  # [B, N0, K]
        anchor_probs = F.softmax(anchor_logits, dim=-1)
        
        return anchor_probs


class RoMaTransformer(nn.Module):
    """
    RoMa Transformer 匹配解码器
    - 多层 Transformer (带解剖偏置注意力)
    - 锚点概率预测
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        d_model = config.get('ROMA', {}).get('D_MODEL', 256)
        n_heads = config.get('ROMA', {}).get('N_HEADS', 8)
        n_layers = config.get('ROMA', {}).get('N_LAYERS', 4)
        lambda_vessel = config.get('ROMA', {}).get('LAMBDA_VESSEL', 1.0)
        num_anchors = config.get('ROMA', {}).get('NUM_ANCHORS', 64*64)
        
        self.d_model = d_model
        self.num_anchors = num_anchors
        
        # 特征投影层 (将 Backbone 特征投影到 d_model 维度)
        # 动态计算输入通道数
        use_dinov2 = config.get('ROMA', {}).get('USE_DINOV2', False)
        dinov2_model = config.get('ROMA', {}).get('DINOV2_MODEL', 'dinov2_vits14')
        superpoint_path = config.get('ROMA', {}).get('SUPERPOINT_PATH', None)
        
        d_vgg = 256
        d_coarse = 0
        
        if use_dinov2:
            if 'vits' in dinov2_model:
                d_coarse = 384
            elif 'vitb' in dinov2_model:
                d_coarse = 768
            elif 'vitl' in dinov2_model:
                d_coarse = 1024
            elif 'vitg' in dinov2_model:
                d_coarse = 1536
            else:
                d_coarse = 384 # Default fallback
        elif superpoint_path is not None:
             d_coarse = 256 # SuperPoint 特征维度
        
        # 如果既没有dinov2也没有superpoint，in_channels就只是vgg的维度
        # 但目前Backbone的逻辑是VGG特征总是存在的，而coarse encoder是可选的
        # 并且在superpoint_backbone中，feat_c是拼接后的结果
        in_channels = d_vgg + d_coarse

        # 兜底：如果没开启任何 Coarse Encodere, in_channels = d_vgg
        
        self.feat_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, lambda_vessel)
            for _ in range(n_layers)
        ])
        
        # 锚点预测器
        self.anchor_predictor = AnchorPredictor(d_model, num_anchors)
        
        # [Fix] 位置编码
        self.pos_encoding = PositionEncodingSine(d_model)
        
    def compute_vessel_bias(self, mask0, mask1):
        """
        计算血管偏置矩阵
        Args:
            mask0: [B, 1, H, W] 图像0的血管掩码
            mask1: [B, 1, H, W] 图像1的血管掩码
        Returns:
            vessel_bias: [B, N0, N1] 血管权重矩阵
        """
        B, _, H, W = mask0.shape
        
        # 展平为序列
        mask0_flat = mask0.reshape(B, 1, H*W).transpose(1, 2)  # [B, H*W, 1]
        mask1_flat = mask1.reshape(B, 1, H*W).transpose(1, 2)  # [B, H*W, 1]
        
        # 外积：vessel_bias[i,j] = mask0[i] * mask1[j]
        vessel_bias = torch.matmul(mask0_flat, mask1_flat.transpose(1, 2))  # [B, H*W, H*W]
        
        return vessel_bias
        
    def forward(self, feat_c0, feat_c1, mask0=None, mask1=None):
        """
        Args:
            feat_c0: [B, D, H/8, W/8] 图像0的粗特征
            feat_c1: [B, D, H/8, W/8] 图像1的粗特征
            mask0: [B, 1, H, W] 图像0的血管掩码
            mask1: [B, 1, H, W] 图像1的血管掩码
        Returns:
            anchor_probs: [B, N0, K] 锚点概率分布
            feat0_out: [B, N0, D] 最终特征
            feat1_out: [B, N1, D] 最终特征
        """
        # 自动适配输入通道数
        if feat_c0.shape[1] != self.feat_proj.in_channels:
             # 如果遇到维度不匹配，动态重建投影层 (Quick Fix)
             # 注意：这会导致权重重置，但在训练开始时是可以接受的
             # 更严谨的做法是提前配置好
             device = feat_c0.device
             print(f"Warning: Re-initializing feat_proj due to dimension mismatch. Expected {self.feat_proj.in_channels}, got {feat_c0.shape[1]}")
             self.feat_proj = nn.Conv2d(feat_c0.shape[1], self.d_model, kernel_size=1).to(device)

        B, D_in, H, W = feat_c0.shape
        
        # 特征投影
        feat0 = self.feat_proj(feat_c0)  # [B, d_model, H, W]
        feat1 = self.feat_proj(feat_c1)
        
        # 展平为序列
        feat0 = feat0.reshape(B, self.d_model, H*W).transpose(1, 2)  # [B, H*W, d_model]
        feat1 = feat1.reshape(B, self.d_model, H*W).transpose(1, 2)
        
        # 计算血管偏置
        vessel_bias = None
        if mask0 is not None and mask1 is not None:
            # 下采样掩码到特征图分辨率
            mask0_down = F.interpolate(mask0, size=(H, W), mode='bilinear', align_corners=False)
            mask1_down = F.interpolate(mask1, size=(H, W), mode='bilinear', align_corners=False)
            vessel_bias = self.compute_vessel_bias(mask0_down, mask1_down)
        
        # [Fix] 添加位置编码
        # 在展平之前添加位置编码，赋予 Transformer 空间感知能力
        # feat0, feat1 目前是 [B, H*W, d_model]，需要先 reshape 回去或者在展平前加
        
        # 下面修正逻辑：在展平前加 positional encoding
        feat0 = feat0.transpose(1, 2).reshape(B, self.d_model, H, W) # 还原回 [B, C, H, W]
        feat1 = feat1.transpose(1, 2).reshape(B, self.d_model, H, W)
        
        feat0 = feat0 + self.pos_encoding(feat0)
        feat1 = feat1 + self.pos_encoding(feat1)
        
        # 再次展平
        feat0 = feat0.reshape(B, self.d_model, H*W).transpose(1, 2)
        feat1 = feat1.reshape(B, self.d_model, H*W).transpose(1, 2)

        # 通过 Transformer 层
        for layer in self.layers:
            feat0, feat1 = layer(feat0, feat1, vessel_bias)
        
        # 预测锚点概率
        anchor_probs = self.anchor_predictor(feat0, feat1)
        
        return anchor_probs, feat0, feat1
