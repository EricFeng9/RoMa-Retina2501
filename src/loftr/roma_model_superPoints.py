"""
RoMa 完整模型：Backbone + Transformer + Fine Matcher (SuperPoint Edition)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Changed Import
from .backbone.roma_backbone_superPoints import RoMaBackbone
from .loftr_module.roma_transformer_superPoints import RoMaTransformer


class FineMatching(nn.Module):
    """精细匹配模块：基于粗匹配结果进行亚像素细化"""
    def __init__(self, d_model=256, window_size=5):
        super().__init__()
        self.window_size = window_size
        # 这里采用一个轻量级 CNN，在局部窗口特征上预测 (dx, dy) 偏移。
        # 输入: [N, 2*d_model, w, w] (fix/mov 的精特征 patch 拼接)
        # 输出: [N, 2] — 针对每一个粗匹配的亚像素偏移 (以原图像素为单位)。
        self.refine_net = nn.Sequential(
            nn.Conv2d(d_model * 2, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )
        
    def forward(self, feat_f0, feat_f1, mkpts0_c_pix, mkpts1_c_pix, m_bids, img_hw):
        """
        Args:
            feat_f0: [B, d, Hf, Wf] 图像0的精细特征 (VGG)
            feat_f1: [B, d, Hf, Wf] 图像1的精细特征 (VGG)
            mkpts0_c_pix: [M, 2] 粗匹配点 (image0, 原图像素坐标)
            mkpts1_c_pix: [M, 2] 粗匹配点 (image1, 原图像素坐标)
            m_bids: [M] 每个匹配所属的 batch 索引
            img_hw: (H_img, W_img) 原图尺寸
        Returns:
            mkpts0_f: [M, 2] 精细匹配点 (image0)
            mkpts1_f: [M, 2] 精细匹配点 (image1，已细化)
        """
        # 如果没有任何粗匹配，直接返回
        if mkpts0_c_pix.numel() == 0:
            return mkpts0_c_pix, mkpts1_c_pix
        
        B, d, Hf, Wf = feat_f0.shape
        H_img, W_img = img_hw
        device = feat_f0.device
        window_size = self.window_size
        r = window_size // 2
        
        # 1. 预先构建局部窗口的 offset 网格（在特征图归一化坐标系中）
        # grid_sample 的坐标范围为 [-1, 1]，这里以特征图分辨率为基准，把
        # 像素偏移 [-r, r] 映射到 [-1, 1] 空间。
        if Wf > 1:
            xs = torch.linspace(-r, r, window_size, device=device) * 2.0 / (Wf - 1)
        else:
            xs = torch.zeros(window_size, device=device)
        if Hf > 1:
            ys = torch.linspace(-r, r, window_size, device=device) * 2.0 / (Hf - 1)
        else:
            ys = torch.zeros(window_size, device=device)
        offset_y, offset_x = torch.meshgrid(ys, xs, indexing='ij')  # [w, w]
        offset_grid = torch.stack([offset_x, offset_y], dim=-1)     # [w, w, 2]
        
        # 2. 对每个 batch 内的匹配，提取局部 patch 并通过 refine_net 预测 (dx, dy)
        mkpts1_refined = mkpts1_c_pix.clone()
        
        for b in range(B):
            mask_b = (m_bids == b)
            if mask_b.sum() == 0:
                continue
            
            pts0_b = mkpts0_c_pix[mask_b]  # [Mb, 2]
            pts1_b = mkpts1_c_pix[mask_b]  # [Mb, 2]
            Mb = pts0_b.shape[0]
            
            # 将原图像素坐标映射到特征图的归一化坐标 [-1, 1]
            # 这里假设精特征与原图对齐，仅分辨率不同，用 (x/(W_img-1))*2-1 近似
            x0_norm = (pts0_b[:, 0] / max(W_img - 1, 1e-6)) * 2.0 - 1.0
            y0_norm = (pts0_b[:, 1] / max(H_img - 1, 1e-6)) * 2.0 - 1.0
            x1_norm = (pts1_b[:, 0] / max(W_img - 1, 1e-6)) * 2.0 - 1.0
            y1_norm = (pts1_b[:, 1] / max(H_img - 1, 1e-6)) * 2.0 - 1.0
            
            centers0 = torch.stack([x0_norm, y0_norm], dim=-1).view(Mb, 1, 1, 2)  # [Mb,1,1,2]
            centers1 = torch.stack([x1_norm, y1_norm], dim=-1).view(Mb, 1, 1, 2)
            
            grid0 = offset_grid.unsqueeze(0) + centers0  # [Mb, w, w, 2]
            grid1 = offset_grid.unsqueeze(0) + centers1  # [Mb, w, w, 2]
            
            # 为每个匹配复制一份对应 batch 的特征图
            feat0_b = feat_f0[b].unsqueeze(0).expand(Mb, -1, -1, -1)  # [Mb, d, Hf, Wf]
            feat1_b = feat_f1[b].unsqueeze(0).expand(Mb, -1, -1, -1)  # [Mb, d, Hf, Wf]
            
            # 使用 grid_sample 提取局部窗口特征
            patches0 = F.grid_sample(
                feat0_b, grid0, mode='bilinear', align_corners=True
            )  # [Mb, d, w, w]
            patches1 = F.grid_sample(
                feat1_b, grid1, mode='bilinear', align_corners=True
            )  # [Mb, d, w, w]
            
            # 拼接 fix/mov patch，送入 refine_net 得到 (dx, dy)
            patch_pair = torch.cat([patches0, patches1], dim=1)  # [Mb, 2d, w, w]
            delta = self.refine_net(patch_pair)                  # [Mb, 2, w, w]
            delta = delta.mean(dim=[2, 3])                       # [Mb, 2]
            
            # 将偏移量看作“原图像素坐标”上的修正（网络会通过 loss_f 自行学习尺度）
            mkpts1_refined_b = pts1_b + delta
            mkpts1_refined[mask_b] = mkpts1_refined_b
        
        # 目前只对 moving 图像做细化，fix 端保持不变
        mkpts0_f = mkpts0_c_pix
        mkpts1_f = mkpts1_refined
        return mkpts0_f, mkpts1_f


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
        feat_c0, feat_f0, x_adapted0 = self.backbone(data['image0'], None)
        feat_c1, feat_f1, x_adapted1 = self.backbone(data['image1'], None)
        
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
        
        # 4. 精细匹配：在 VGG 精特征上做局部窗口细化
        mkpts0_f, mkpts1_f = self.fine_matching(
            feat_f0, feat_f1, mkpts0_c_pix, mkpts1_c_pix, b_ids, img_hw=(H, W)
        )
        
        # 5. 更新 data
        data.update({
            'mkpts0_c': mkpts0_c_pix,
            'mkpts1_c': mkpts1_c_pix,
            'mkpts0_f': mkpts0_f,
            'mkpts1_f': mkpts1_f,
            'mconf': mconf,
            'm_bids': b_ids,
            'anchor_probs': anchor_probs,  # 用于损失计算
            # [新增] 用于可视化的中间变量
            'feat_c0': feat_c0,
            'feat_c1': feat_c1,
            'feat_f0': feat_f0,
            'feat_f1': feat_f1,
            'x_adapted0': x_adapted0,
            'x_adapted1': x_adapted1,
        })
        
        return data
