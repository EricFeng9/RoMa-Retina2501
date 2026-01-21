"""
RoMa 完整模型：Backbone + Transformer + Fine Matcher
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .backbone.roma_backbone import RoMaBackbone
from .loftr_module.roma_transformer import RoMaTransformer


class FineMatching(nn.Module):
    """精细匹配模块：基于粗匹配结果进行亚像素细化"""
    def __init__(self, d_model=256, window_size=5):
        super().__init__()
        self.window_size = window_size
        self.refine_net = nn.Sequential(
            nn.Conv2d(d_model * 2, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)  # 输出 (dx, dy) 偏移
        )
        
    def forward(self, feat_f0, feat_f1, mkpts0_c, mkpts1_c):
        """
        Args:
            feat_f0: [B, d, H/4, W/4] 精细特征
            feat_f1: [B, d, H/4, W/4] 精细特征
            mkpts0_c: [B, N, 2] 粗匹配点 (在原图尺度)
            mkpts1_c: [B, N, 2] 粗匹配点 (在原图尺度)
        Returns:
            mkpts0_f: [B, N, 2] 精细匹配点
            mkpts1_f: [B, N, 2] 精细匹配点
        """
        B, d, H, W = feat_f0.shape
        N = mkpts0_c.shape[1]
        
        # 将坐标缩放到精细特征图尺度
        scale = torch.tensor([W-1, H-1], device=feat_f0.device)
        mkpts0_scaled = mkpts0_c / scale * torch.tensor([W-1, H-1], device=feat_f0.device)
        mkpts1_scaled = mkpts1_c / scale * torch.tensor([W-1, H-1], device=feat_f0.device)
        
        # 提取局部窗口特征 (简化版，使用grid_sample)
        # 这里简化处理：直接返回粗匹配结果
        # TODO: 实现完整的局部窗口匹配
        
        return mkpts0_c, mkpts1_c


class RoMa(nn.Module):
    """
    RoMa 主模型
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone
        self.backbone = RoMaBackbone(config)
        
        # Transformer 匹配解码器
        self.transformer = RoMaTransformer(config)
        
        # 精细匹配
        self.fine_matching = FineMatching(
            d_model=config.get('ROMA', {}).get('D_MODEL', 256),
            window_size=config.get('ROMA', {}).get('FINE_WINDOW_SIZE', 5)
        )
        
        # 锚点网格
        self.register_buffer('anchor_grid', self._create_anchor_grid(64, 64))
        
    def _create_anchor_grid(self, H, W):
        """创建锚点网格"""
        y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
        grid = torch.stack([x, y], dim=-1).reshape(-1, 2)  # [H*W, 2]
        return grid
        
    def extract_coarse_matches(self, anchor_probs, conf_thresh=0.01, top_k=1000):
        """
        从锚点概率分布中提取粗匹配点
        Args:
            anchor_probs: [B, N0, K] 锚点概率分布
            conf_thresh: 置信度阈值
            top_k: 最多保留的匹配点数
        Returns:
            mkpts0: [M, 2] 匹配点 (在图像0)
            mkpts1: [M, 2] 匹配点 (在图像1)
            mconf: [M] 匹配置信度
            b_ids: [M] batch索引
        """
        B, N0, K = anchor_probs.shape
        H = W = int(np.sqrt(N0))
        
        # 获取每个点的最高概率锚点
        max_probs, max_indices = anchor_probs.max(dim=-1)  # [B, N0]
        
        # 过滤低置信度点
        conf_mask = max_probs > conf_thresh  # [B, N0]
        
        mkpts0_list = []
        mkpts1_list = []
        mconf_list = []
        b_ids_list = []
        
        for b in range(B):
            valid_mask = conf_mask[b]
            if valid_mask.sum() == 0:
                continue
            
            valid_indices = torch.where(valid_mask)[0]  # [M]
            valid_probs = max_probs[b][valid_mask]
            valid_anchors = max_indices[b][valid_mask]
            
            # 限制数量
            if len(valid_indices) > top_k:
                topk_values, topk_indices = torch.topk(valid_probs, top_k)
                valid_indices = valid_indices[topk_indices]
                valid_probs = topk_values
                valid_anchors = valid_anchors[topk_indices]
            
            # 将索引转换为坐标
            y0 = (valid_indices // W).float() / (H - 1)
            x0 = (valid_indices % W).float() / (W - 1)
            pts0 = torch.stack([x0, y0], dim=-1)  # [M, 2]
            
            # 锚点坐标
            pts1 = self.anchor_grid[valid_anchors]  # [M, 2]
            
            mkpts0_list.append(pts0)
            mkpts1_list.append(pts1)
            mconf_list.append(valid_probs)
            b_ids_list.append(torch.full((len(valid_indices),), b, device=anchor_probs.device))
        
        if len(mkpts0_list) == 0:
            # 没有匹配点
            return (torch.empty((0, 2), device=anchor_probs.device),
                    torch.empty((0, 2), device=anchor_probs.device),
                    torch.empty((0,), device=anchor_probs.device),
                    torch.empty((0,), dtype=torch.long, device=anchor_probs.device))
        
        mkpts0 = torch.cat(mkpts0_list, dim=0)
        mkpts1 = torch.cat(mkpts1_list, dim=0)
        mconf = torch.cat(mconf_list, dim=0)
        b_ids = torch.cat(b_ids_list, dim=0)
        
        return mkpts0, mkpts1, mconf, b_ids
        
    def forward(self, data):
        """
        Args:
            data: dict with keys:
                - 'image0': [B, 1, H, W]
                - 'image1': [B, 1, H, W]
                - 'mask0': [B, 1, H, W] (optional) 血管掩码
                - 'mask1': [B, 1, H, W] (optional) 血管掩码
        Returns:
            data: updated dict with matching results
        
        注意：按照图片要求，不在 Backbone 中拼接 image0 和 mask0。
        Backbone 只提取纯净的图像特征，掩码在 Transformer 的注意力机制中使用。
        """
        # 1. 提取特征（不传入掩码，保持特征纯净性）
        feat_c0, feat_f0 = self.backbone(data['image0'], None)
        feat_c1, feat_f1 = self.backbone(data['image1'], None)
        
        # 2. Transformer 匹配 (粗级) - 传入掩码用于解剖偏置注意力
        anchor_probs, feat0_trans, feat1_trans = self.transformer(
            feat_c0, feat_c1,
            data.get('mask0', None),
            data.get('mask1', None)
        )
        
        # 3. 提取粗匹配点
        mkpts0_c, mkpts1_c, mconf, b_ids = self.extract_coarse_matches(
            anchor_probs, 
            conf_thresh=self.config.get('ROMA', {}).get('CONF_THRESH', 0.01),
            top_k=self.config.get('ROMA', {}).get('TOP_K', 1000)
        )
        
        # 将归一化坐标转换为像素坐标
        H, W = data['image0'].shape[2:]
        mkpts0_c_pix = mkpts0_c * torch.tensor([W-1, H-1], device=mkpts0_c.device)
        mkpts1_c_pix = mkpts1_c * torch.tensor([W-1, H-1], device=mkpts1_c.device)
        
        # 4. 精细匹配 (简化版：暂时跳过)
        mkpts0_f = mkpts0_c_pix
        mkpts1_f = mkpts1_c_pix
        
        # 5. 更新 data
        data.update({
            'mkpts0_c': mkpts0_c_pix,
            'mkpts1_c': mkpts1_c_pix,
            'mkpts0_f': mkpts0_f,
            'mkpts1_f': mkpts1_f,
            'mconf': mconf,
            'm_bids': b_ids,
            'anchor_probs': anchor_probs  # 用于损失计算
        })
        
        return data
