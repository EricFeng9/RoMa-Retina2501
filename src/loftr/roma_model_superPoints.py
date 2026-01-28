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
    def __init__(self, d_model=128, window_size=5):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        
        # d_model should be 128 for VGG fine features. 
        # Input to refine_net is feat_f0 (128) + feat_f1 (128) = 256.
        # So d_model * 2 = 256.
        self.refine_net = nn.Sequential(
            nn.Conv2d(d_model * 2, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )
        
    def crop_patches(self, feat, mkpts, b_ids, window_size):
        """
        从特征图中裁剪局部窗口
        Args:
            feat: [B, C, H, W]
            mkpts: [M, 2] 像素坐标 (x, y)
            b_ids: [M]
            window_size: int
        Returns:
            patches: [M, C, W, W]
        """
        B, C, H, W = feat.shape
        M = mkpts.shape[0]
        
        # 归一化坐标到 [-1, 1]
        # grid_sample expects (x, y) in [-1, 1]
        # mkpts are in range [0, W_orig-1], [0, H_orig-1]
        # We need to map to feat resolution coords first?
        # Actually simplest to use normalized coords directly.
        # But mkpts input to this function are already scaled to pixels of ORIGINAL image in RoMa.forward?
        # Let's check RoMa.forward. It passes mkpts0_c_pix.
        # And feat is H/4, W/4.
        # So we should map pixels to [-1, 1].
        
        # NOTE: mkpts passed here are in original image resolution (H_orig, W_orig).
        # We assume feat covers the same FOV.
        # So we normalize using H_orig, W_orig (which we don't strictly have here, but we can infer if we assume feat is 1/4).
        # Actually safer to pass scalers. But assuming 1/4 is consistent with Backbone.
        
        # Wait, grid_sample uses relative coordinates [-1, 1]. 
        # So (0,0) is top-left, (W-1, H-1) is bottom-right.
        # If mkpts are in (0..W_orig), we normalize by W_orig.
        # W_orig = W * 4.
        
        # Generate grid for each keypoint
        # Grid range: [-win/2, win/2] pixels.
        
        device = feat.device
        
        # Create a base grid [1, 1, win, win, 2]
        r = window_size // 2
        grid_base_range = torch.arange(-r, r + 1, device=device)
        grid_y, grid_x = torch.meshgrid(grid_base_range, grid_base_range, indexing='ij')
        grid_base = torch.stack([grid_x, grid_y], dim=-1).float() # [win, win, 2]
        
        # Expand for all matches: [M, win, win, 2]
        grid = grid_base.unsqueeze(0).expand(M, -1, -1, -1)
        
        # Add keypoint centers
        # mkpts: [M, 2]. Scale to feature map resolution? 
        # grid_base is in pixels of feature map? Or original image?
        # Usually easier to work in normalized coords of the feature map.
        
        # Let's normalize everything to [-1, 1].
        # H, W are feature map dims.
        # mkpts are in original pixels.
        # Original dims approx H*4, W*4.
        
        # Normalize mkpts to [-1, 1]
        # x_norm = 2 * (x / (W_orig - 1)) - 1
        # If we don't know W_orig, we can try to use feature map W.
        # x_feat = x_orig / 4.0
        # x_norm = 2 * (x_feat / (W - 1)) - 1
        
        # Better:
        mkpts_feat = mkpts / 4.0 # Scale to feature map coords
        
        # Normalize to [-1, 1]
        # grid_sample coordinates: -1=left, 1=right.
        # size (W, H)
        inv_h = 1.0 / (H - 1)
        inv_w = 1.0 / (W - 1)
        
        # Add center to grid
        # grid is offset in pixels. 
        # sample_grid = (center + grid_offset) normalized
        center = mkpts_feat.unsqueeze(1).unsqueeze(1) # [M, 1, 1, 2]
        sampling_grid = center + grid # [M, win, win, 2] in feature pixels
        
        # Normalize sampling grid
        sampling_grid_norm = torch.zeros_like(sampling_grid)
        sampling_grid_norm[..., 0] = 2.0 * sampling_grid[..., 0] * inv_w - 1.0
        sampling_grid_norm[..., 1] = 2.0 * sampling_grid[..., 1] * inv_h - 1.0
        
        # Ensure we sample from proper batch indices
        # grid_sample doesn't support list of batch indices directly.
        # We have to reshape everything to [M, C, H, W] is not possible because feats are [B...].
        # We have to iterate or construct a hack.
        # A common trick: Reshape B into channels or use F.grid_sample on valid items.
        # But M matches can come from different batch items.
        
        # Solution:
        # Loop over unique batch ids? Or just loop B.
        patches_list = []
        for b in range(B):
            mask = (b_ids == b)
            if not mask.any():
                continue
            
            # Select proper points
            sub_grid = sampling_grid_norm[mask] # [m, win, win, 2]
            
            # Feature map for this batch [1, C, H, W]
            sub_feat = feat[b].unsqueeze(0) 
            
            # Grid sample
            # input: [1, C, H, W]
            # grid: [1, m, win*win, 2]? No.
            # Grid sample expects [N, H_out, W_out, 2].
            # We want output [m, C, win, win].
            # We can trick it: set N=m? No, feat has Batch=1.
            # We can treat 'm' as the height of the output grid?
            # grid: [1, m*win, win, 2] -> Output [1, C, m*win, win] -> reshape
            
            m = sub_grid.shape[0]
            flat_grid = sub_grid.reshape(1, m * window_size, window_size, 2)
            
            out = F.grid_sample(sub_feat, flat_grid, mode='bilinear', align_corners=True, padding_mode='zeros')
            # out: [1, C, m*win, win]
            
            out = out.view(C, m, window_size, window_size).permute(1, 0, 2, 3) # [m, C, win, win]
            
            # We need to put them back in order? 
            # The mask approach splits them. We need to collect valid indices to restore order.
            # Or just append and assume we sort later? No.
            
            # Easier way:
            # Reconstruct full tensor.
            patches_list.append((torch.where(mask)[0], out))
            
        # Reassemble
        patches = torch.zeros(M, C, window_size, window_size, dtype=feat.dtype, device=device)
        for indices, sub_patches in patches_list:
            patches[indices] = sub_patches
            
        return patches
        
    def forward(self, feat_f0, feat_f1, mkpts0_c, mkpts1_c, b_ids):
        """
        Args:
            feat_f0: [B, d, H/4, W/4] 精细特征
            feat_f1: [B, d, H/4, W/4] 精细特征
            mkpts0_c: [M, 2] 粗匹配点 (在原图尺度, pixels)
            mkpts1_c: [M, 2] 粗匹配点
            b_ids: [M]
        Returns:
            mkpts0_f: [M, 2] 精细匹配点
            mkpts1_f: [M, 2] 精细匹配点
        """
        if mkpts0_c.shape[0] == 0:
            return mkpts0_c, mkpts1_c
            
        # 1. Extract patches
        # feat_f0 (128 ch)
        patch0 = self.crop_patches(feat_f0, mkpts0_c, b_ids, self.window_size) # [M, 128, 5, 5]
        patch1 = self.crop_patches(feat_f1, mkpts1_c, b_ids, self.window_size) # [M, 128, 5, 5]
        
        # 2. Concat
        patch_pair = torch.cat([patch0, patch1], dim=1) # [M, 256, 5, 5]
        
        # 3. Predict Delta
        # RefineNet: [M, 256, 5, 5] -> [M, 2, 5, 5] (assuming stride 1, pad 1 for 3x3 conv)
        delta_map = self.refine_net(patch_pair)
        
        # Take center or global average?
        # Assuming we simply take the center value as the offset for the keypoint
        # Or if the net predicts a heatmap, we'd do expectations.
        # But Output is 2 channels using simple Conv.
        # Let's take the center feature as the offset.
        c = self.window_size // 2
        delta = delta_map[:, :, c, c] # [M, 2]
        
        # Limit delta range to avoid flying away?
        # Often helpful to tanh * scale
        # delta = torch.tanh(delta) * (self.window_size / 2)
        
        # 4. Apply refinement to mkpts1 (usually we refine the second point to match the first?)
        # Or refining both?
        # RoMa usually refines matches relative to each other.
        # Let's refine mkpts1 to better match mkpts0?
        # Or this module outputs offsets for BOTH? 2 channels usually means dx, dy for one point.
        # Assuming optimizing mkpts1 to match mkpts0.
        mkpts1_f = mkpts1_c + delta
        mkpts0_f = mkpts0_c # Keep 0 fixed? Or maybe fine_net output 4 channels for both?
        # Config 2 channels usually implies 1 point update or relative update.
        
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
        # VGG Fine Features have 128 channels.
        # FineMatching receives concatenated features (128+128=256).
        # refine_net expects 2*d_model. So we MUST set d_model=128.
        self.fine_matching = FineMatching(
            d_model=128, # Force 128 to match VGG fine features
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
        
        # 4. 精细匹配 (恢复完整功能)
        # 传入 b_ids 以便正确采样
        mkpts0_f, mkpts1_f = self.fine_matching(feat_f0, feat_f1, mkpts0_c_pix, mkpts1_c_pix, b_ids)
        
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
