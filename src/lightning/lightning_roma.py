"""
PyTorch Lightning 模块：RoMa 训练与验证
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from loguru import logger

from src.loftr.roma_model import RoMa
from src.losses.roma_loss import RoMaLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors, aggregate_metrics
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_RoMa(pl.LightningModule):
    """
    PyTorch Lightning 模块：RoMa 多模态配准
    """
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        # 配置
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.dump_dir = dump_dir
        
        # 模型
        self.model = RoMa(config)
        
        # 损失函数
        self.loss_fn = RoMaLoss(config)
        
        # 加载预训练权重
        if pretrained_ckpt:
            state = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.load_state_dict(state, strict=False)
            logger.info(f"从 {pretrained_ckpt} 加载预训练权重")
        
        # 用于验证阶段的 metrics 累积
        self.validation_step_outputs = []
        
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        """自定义优化器步骤：实现课程学习"""
        # 课程学习：动态调整 lambda_vessel
        # 阶段1 (前25 epoch): lambda = 0 (全图感知)
        # 阶段2 (25-50 epoch): lambda = 0.2 (软掩码引导)
        # 阶段3 (50 epoch以后): lambda = 0.05 (精度冲刺)
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            current_epoch = self.current_epoch
            
            if current_epoch < 25:
                lambda_vessel = 0.0
            elif current_epoch < 50:
                lambda_vessel = 0.2
            else:
                lambda_vessel = 0.05
            
            # 动态更新所有 Transformer 层的 lambda_vessel
            for layer in self.model.transformer.layers:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'lambda_vessel'):
                    layer.self_attn.lambda_vessel = lambda_vessel
                if hasattr(layer, 'cross_attn') and hasattr(layer.cross_attn, 'lambda_vessel'):
                    layer.cross_attn.lambda_vessel = lambda_vessel
        
        # 执行优化步骤
        optimizer.step(closure=optimizer_closure)
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        # 前向传播
        data = self.model(batch)
        
        # 计算损失
        loss, metrics = self.loss_fn(data)
        
        # 记录
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/loss_c', metrics['loss_c'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train/loss_f', metrics['loss_f'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        # 验证/测试时关闭血管分割图的注意力偏置（lambda=0）
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            for layer in self.model.transformer.layers:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'lambda_vessel'):
                    layer.self_attn.lambda_vessel = 0.0
                if hasattr(layer, 'cross_attn') and hasattr(layer.cross_attn, 'lambda_vessel'):
                    layer.cross_attn.lambda_vessel = 0.0
        
        # 前向传播
        data = self.model(batch)
        
        # 计算损失
        loss, metrics = self.loss_fn(data)
        
        batch_size = batch['image0'].shape[0]
        
        # 总是尝试估计单应矩阵，即使没有足够的匹配点
        H_est_list = self._estimate_homography_batch(data, batch, batch_size)
        data['H_est'] = H_est_list
        
        # 计算几何误差 (如果有真值单应矩阵)
        if 'T_0to1' in batch and 'mkpts0_f' in data and len(data['mkpts0_f']) > 0:
            # 调用 metrics.py 中的函数计算误差 (针对 MultiModal 会计算单应重投影误差)
            compute_symmetrical_epipolar_errors(data)
            
            # 计算 AUC 指标
            auc_metrics = self._compute_auc_metrics(data['epi_errs'])
            for k, v in auc_metrics.items():
                self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # 记录损失
        self.log('val/avg_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/avg_loss_c', metrics['loss_c'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val/avg_loss_f', metrics['loss_f'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        # 生成可视化图
        figures = {}
        if self.config.TRAINER.PLOT_MODE == 'evaluation':
            figures = make_matching_figures(data, self.config, mode='evaluation')
        
        # 保存输出供 Callback 使用
        output = {
            'loss': loss.item(),
            'H_est': data.get('H_est', []),
            'figures': figures
        }
        
        self.validation_step_outputs.append(output)
        
        return output
    
    def on_validation_epoch_end(self):
        """验证周期结束时清空缓存"""
        self.validation_step_outputs.clear()
    
    def _estimate_homography_batch(self, data, batch, batch_size):
        """
        使用 RANSAC 估计单应矩阵
        Args:
            data: dict with keys:
                - 'mkpts0_f': [M, 2]
                - 'mkpts1_f': [M, 2]
                - 'm_bids': [M]
            batch: dict with images for mask calculation
            batch_size: int, 实际的 batch size
        Returns:
            H_list: list of [3, 3] numpy arrays, 长度为 batch_size
        """
        import cv2
        import numpy as np
        
        # 如果没有匹配点，返回单位矩阵
        if 'mkpts0_f' not in data or 'mkpts1_f' not in data or 'm_bids' not in data:
            return [np.eye(3, dtype=np.float32) for _ in range(batch_size)]
        
        if len(data['mkpts0_f']) == 0:
            return [np.eye(3, dtype=np.float32) for _ in range(batch_size)]
        
        mkpts0 = data['mkpts0_f'].cpu().numpy()
        mkpts1 = data['mkpts1_f'].cpu().numpy()
        m_bids = data['m_bids'].cpu().numpy()
        
        H_list = []
        
        for b in range(batch_size):
            mask_b = m_bids == b
            if mask_b.sum() < 4:
                H_list.append(np.eye(3, dtype=np.float32))
                continue
            
            pts0 = mkpts0[mask_b]
            pts1 = mkpts1[mask_b]
            
            # 空间均匀化采样 (8x8 分区)
            H, W = data['image0'].shape[2:]
            pts0_binned, pts1_binned = self._spatial_binning(pts0, pts1, 
                                                             img_size=(H, W), 
                                                             grid_size=8, 
                                                             top_k=5)
            
            if len(pts0_binned) < 4:
                H_list.append(np.eye(3, dtype=np.float32))
                continue
            
            # RANSAC
            try:
                H, mask_inliers = cv2.findHomography(pts0_binned, pts1_binned, 
                                                     cv2.RANSAC, 
                                                     ransacReprojThreshold=3.0)
                if H is None:
                    H = np.eye(3, dtype=np.float32)
            except:
                H = np.eye(3, dtype=np.float32)
            
            H_list.append(H)
        
        return H_list
    
    def _spatial_binning(self, pts0, pts1, img_size, grid_size=8, top_k=5):
        """
        空间均匀化采样：将图像划分为 grid_size x grid_size 个分区
        每个分区选取 top_k 个最近的点
        Args:
            pts0: [N, 2] 点集
            pts1: [N, 2] 点集
            img_size: (H, W)
            grid_size: 分区数量
            top_k: 每个分区保留的点数
        Returns:
            pts0_binned: [M, 2]
            pts1_binned: [M, 2]
        """
        import numpy as np
        
        if len(pts0) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

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
                in_grid = (grid_y == i) & (grid_x == j)
                if in_grid.sum() == 0:
                    continue
                
                # 随机选取 top_k 个点
                indices = np.where(in_grid)[0]
                if len(indices) > top_k:
                    # 如果有置信度信息，按置信度取 top_k 会更好，但这里没有置信度输入，先用随机
                    indices = np.random.choice(indices, top_k, replace=False)
                
                pts0_binned.append(pts0[indices])
                pts1_binned.append(pts1[indices])
        
        if len(pts0_binned) == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        
        pts0_binned = np.concatenate(pts0_binned, axis=0)
        pts1_binned = np.concatenate(pts1_binned, axis=0)
        
        return pts0_binned, pts1_binned
    
    def _compute_auc_metrics(self, epi_errs, thresholds=[5, 10, 20]):
        """
        计算 AUC 指标
        Args:
            epi_errs: [M] 对极误差
            thresholds: list of thresholds
        Returns:
            metrics: dict
        """
        metrics = {}
        valid_errs = epi_errs[epi_errs != float('inf')]
        
        if len(valid_errs) == 0:
            for th in thresholds:
                metrics[f'auc@{th}'] = 0.0
            return metrics
        
        for th in thresholds:
            auc = (valid_errs < th).float().mean().item()
            metrics[f'auc@{th}'] = auc
        
        return metrics
