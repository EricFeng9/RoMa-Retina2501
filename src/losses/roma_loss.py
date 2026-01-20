"""
RoMa 损失函数
- 粗级损失：KL 散度 (Regression-by-classification)
- 精级损失：Charbonnier 损失 (Robust Regression)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RoMaLoss(nn.Module):
    """
    RoMa 损失函数
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 损失权重
        self.loss_config = config.get('ROMA', {}).get('LOSS', {})
        self.weight_coarse = self.loss_config.get('WEIGHT_COARSE', 1.0)
        self.weight_fine = self.loss_config.get('WEIGHT_FINE', 1.0)
        
        # 梯度保底参数
        self.epsilon = self.loss_config.get('EPSILON', 0.1)
        
        # Charbonnier 损失参数
        self.smooth_param = self.loss_config.get('SMOOTH_PARAM', 0.01)
        
        # 锚点数量
        self.num_anchors = config.get('ROMA', {}).get('NUM_ANCHORS', 64*64)
        self.anchor_grid_size = int(np.sqrt(self.num_anchors))
        
        # 创建锚点网格
        self.register_buffer('anchor_grid', self._create_anchor_grid())
        
    def _create_anchor_grid(self):
        """创建归一化的锚点网格 [0, 1]"""
        H = W = self.anchor_grid_size
        y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
        grid = torch.stack([x, y], dim=-1).reshape(-1, 2)  # [K, 2]
        return grid
    
    def create_gt_anchor_distribution(self, pts_gt, img_size, sigma=0.02):
        """
        根据真值点创建锚点概率分布 (高斯分布)
        Args:
            pts_gt: [B, N, 2] 真值点坐标 (归一化到 [0, 1])
            img_size: (H, W)
            sigma: 高斯分布的标准差
        Returns:
            gt_dist: [B, N, K] 真值锚点分布
        """
        B, N, _ = pts_gt.shape
        K = self.num_anchors
        
        # 扩展维度
        pts_gt_exp = pts_gt.unsqueeze(2)  # [B, N, 1, 2]
        anchors_exp = self.anchor_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, K, 2]
        
        # 计算距离
        dist = torch.norm(pts_gt_exp - anchors_exp, dim=-1)  # [B, N, K]
        
        # 高斯分布
        gt_dist = torch.exp(-dist ** 2 / (2 * sigma ** 2))
        
        # 归一化
        gt_dist = gt_dist / (gt_dist.sum(dim=-1, keepdim=True) + 1e-8)
        
        return gt_dist
    
    def compute_coarse_loss(self, anchor_probs, data):
        """
        计算粗级损失 (KL 散度 + 梯度保底)
        Args:
            anchor_probs: [B, N0, K] 预测的锚点概率分布
            data: dict with keys:
                - 'T_0to1': [B, 3, 3] 真值变换矩阵
                - 'vessel_weight0': [B, 1, H, W] 血管权重 (可选)
        Returns:
            loss_c: scalar
            metrics: dict
        """
        B, N0, K = anchor_probs.shape
        H = W = int(np.sqrt(N0))
        img_H, img_W = data['image0'].shape[2:]
        
        # 生成粗级网格点 (在原图尺度)
        y, x = torch.meshgrid(torch.linspace(0, img_H-1, H), 
                             torch.linspace(0, img_W-1, W), indexing='ij')
        pts0 = torch.stack([x, y], dim=-1).reshape(-1, 2).to(anchor_probs.device)  # [N0, 2]
        pts0_homo = torch.cat([pts0, torch.ones((N0, 1), device=pts0.device)], dim=-1)  # [N0, 3]
        
        # 应用变换矩阵得到真值点
        T = data['T_0to1']  # [B, 3, 3]
        pts1_gt_homo = torch.matmul(T, pts0_homo.T)  # [B, 3, N0]
        pts1_gt = pts1_gt_homo[:, :2, :] / (pts1_gt_homo[:, 2:3, :] + 1e-8)  # [B, 2, N0]
        pts1_gt = pts1_gt.transpose(1, 2)  # [B, N0, 2]
        
        # 归一化到 [0, 1]
        pts1_gt_norm = pts1_gt / torch.tensor([img_W-1, img_H-1], device=pts1_gt.device)
        
        # 创建真值锚点分布
        gt_anchor_dist = self.create_gt_anchor_distribution(
            pts1_gt_norm, (img_H, img_W), 
            sigma=self.loss_config.get('ANCHOR_SIGMA', 0.02)
        )  # [B, N0, K]
        
        # KL 散度
        log_probs = torch.log(anchor_probs + 1e-8)
        kl_loss = F.kl_div(log_probs, gt_anchor_dist, reduction='none').sum(dim=-1)  # [B, N0]
        
        # 梯度保底：引入血管权重
        if 'vessel_weight0' in data:
            vessel_weight = data['vessel_weight0'].squeeze(1)  # [B, H, W]
            # 下采样到粗级分辨率
            vessel_weight_down = F.interpolate(
                vessel_weight.unsqueeze(1), size=(H, W), 
                mode='bilinear', align_corners=False
            ).squeeze(1)  # [B, H, W]
            vessel_weight_flat = vessel_weight_down.reshape(B, -1)  # [B, N0]
            
            # 梯度保底: max(W_vessel, epsilon)
            loss_weight = torch.clamp(vessel_weight_flat, min=self.epsilon)
        else:
            loss_weight = torch.ones_like(kl_loss)
        
        # 加权平均
        loss_c = (kl_loss * loss_weight).sum() / (loss_weight.sum() + 1e-8)
        
        metrics = {
            'loss_c': loss_c.item(),
            'kl_mean': kl_loss.mean().item()
        }
        
        return loss_c, metrics
    
    def compute_fine_loss(self, mkpts0_f, mkpts1_f, data):
        """
        计算精级损失 (Charbonnier 损失)
        Args:
            mkpts0_f: [M, 2] 精细匹配点 (image0)
            mkpts1_f: [M, 2] 精细匹配点 (image1)
            data: dict with keys:
                - 'T_0to1': [B, 3, 3] 真值变换矩阵
                - 'm_bids': [M] batch索引
        Returns:
            loss_f: scalar
            metrics: dict
        """
        if len(mkpts0_f) == 0:
            return torch.tensor(0.0, device=mkpts0_f.device), {'loss_f': 0.0}
        
        # 获取batch索引
        m_bids = data['m_bids']
        B = data['T_0to1'].shape[0]
        
        # 计算真值匹配点
        mkpts0_homo = torch.cat([mkpts0_f, torch.ones((len(mkpts0_f), 1), device=mkpts0_f.device)], dim=-1)  # [M, 3]
        
        losses = []
        for b in range(B):
            mask_b = m_bids == b
            if mask_b.sum() == 0:
                continue
            
            pts0_b = mkpts0_homo[mask_b]  # [M_b, 3]
            pts1_b = mkpts1_f[mask_b]  # [M_b, 2]
            
            # 应用变换
            T_b = data['T_0to1'][b]  # [3, 3]
            pts1_gt_homo = torch.matmul(T_b, pts0_b.T)  # [3, M_b]
            pts1_gt = pts1_gt_homo[:2, :] / (pts1_gt_homo[2:3, :] + 1e-8)  # [2, M_b]
            pts1_gt = pts1_gt.T  # [M_b, 2]
            
            # Charbonnier 损失
            diff = pts1_b - pts1_gt
            loss_b = ((diff ** 2).sum(dim=-1) + self.smooth_param) ** 0.25
            losses.append(loss_b.mean())
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=mkpts0_f.device), {'loss_f': 0.0}
        
        loss_f = torch.stack(losses).mean()
        
        metrics = {
            'loss_f': loss_f.item()
        }
        
        return loss_f, metrics
    
    def forward(self, data):
        """
        计算总损失
        Args:
            data: dict with keys:
                - 'anchor_probs': [B, N0, K]
                - 'mkpts0_f': [M, 2]
                - 'mkpts1_f': [M, 2]
                - 'T_0to1': [B, 3, 3]
                - 'vessel_weight0': [B, 1, H, W] (optional)
                - 'm_bids': [M]
        Returns:
            loss: scalar
            metrics: dict
        """
        # 粗级损失
        loss_c, metrics_c = self.compute_coarse_loss(data['anchor_probs'], data)
        
        # 精级损失
        loss_f, metrics_f = self.compute_fine_loss(data['mkpts0_f'], data['mkpts1_f'], data)
        
        # 总损失
        loss = self.weight_coarse * loss_c + self.weight_fine * loss_f
        
        metrics = {
            'loss': loss.item(),
            **metrics_c,
            **metrics_f
        }
        
        return loss, metrics
