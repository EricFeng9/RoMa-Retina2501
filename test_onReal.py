# -*- coding: utf-8 -*-
import sys
import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import torch.utils.data as data
import matplotlib.pyplot as plt

from src.config.default import get_cfg_defaults
from src.lightning.lightning_roma import PL_RoMa
from measurement import calculate_metrics
from src.utils.plotting import make_matching_figure

# 导入数据集类
from data.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from data.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from data.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset

# 数据集根目录
DATA_ROOTS = {
    'cffa': "/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_cffa",
    'cfoct': "/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_cfoct",
    'octfa': "/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_octfa",
    'cfocta': "/data/student/Fengjunming/LoFtr/data/CF_OCTA_v2_repaired"
}

class RoMaTestDatasetWrapper(data.Dataset):
    def __init__(self, dataset, img_size=518):
        self.dataset = dataset
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 统一解包 6 个返回值：(fix, moving_orig, moving_gt, fix_path, moving_path, T_0to1)
        fix_t, moving_original_t, moving_gt_t, fix_path, moving_path, T_0to1 = self.dataset[idx]
        
        # 统一归一化：将 [-1, 1] 转回 [0, 1]
        moving_original_t = (moving_original_t + 1.0) / 2.0
        moving_gt_t = (moving_gt_t + 1.0) / 2.0
        
        # Resize 到目标尺寸
        import torch.nn.functional as F
        if fix_t.shape[-1] != self.img_size:
            fix_t = F.interpolate(fix_t.unsqueeze(0), size=(self.img_size, self.img_size), 
                                   mode='bilinear', align_corners=False).squeeze(0)
            moving_original_t = F.interpolate(moving_original_t.unsqueeze(0), size=(self.img_size, self.img_size), 
                                   mode='bilinear', align_corners=False).squeeze(0)
            moving_gt_t = F.interpolate(moving_gt_t.unsqueeze(0), size=(self.img_size, self.img_size), 
                                   mode='bilinear', align_corners=False).squeeze(0)

        return {
            'image0': fix_t,  # [C, H, W]
            'image1': moving_original_t,  # [C, H, W] 原始未配准的moving
            'image1_gt': moving_gt_t,  # [C, H, W] 配准后的GT
            'pair_names': (os.path.basename(fix_path), os.path.basename(moving_path)),
            'dataset_name': 'MultiModal'
        }

def filter_valid_area(img1, img2):
    """筛选有效区域"""
    assert img1.shape[:2] == img2.shape[:2]
    
    if len(img1.shape) == 3:
        mask1 = np.any(img1 > 10, axis=2)
    else:
        mask1 = img1 > 0
    
    if len(img2.shape) == 3:
        mask2 = np.any(img2 > 10, axis=2)
    else:
        mask2 = img2 > 0
    
    valid_mask = mask1 & mask2
    
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return img1, img2
    
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    
    filtered_img1 = img1[row_min:row_max+1, col_min:col_max+1].copy()
    filtered_img2 = img2[row_min:row_max+1, col_min:col_max+1].copy()
    
    valid_mask_cropped = valid_mask[row_min:row_max+1, col_min:col_max+1]
    
    if len(filtered_img1.shape) == 3:
        filtered_img1[~valid_mask_cropped] = 0
    else:
        filtered_img1[~valid_mask_cropped] = 0
    
    if len(filtered_img2.shape) == 3:
        filtered_img2[~valid_mask_cropped] = 0
    else:
        filtered_img2[~valid_mask_cropped] = 0
    
    return filtered_img1, filtered_img2

def spatial_binning(pts0, pts1, mconf, img_size, grid_size=8, top_k=5):
    """
    空间均匀化采样：将图像划分为 grid_size x grid_size 个分区
    每个分区选取置信度最高的 top_k 个点
    
    Args:
        pts0: [N, 2] 匹配点（图像0）
        pts1: [N, 2] 匹配点（图像1）
        mconf: [N] 匹配置信度
        img_size: (H, W) 图像尺寸
        grid_size: 网格划分数量（默认8x8）
        top_k: 每个格子保留的最大点数（默认5）
    Returns:
        pts0_binned: [M, 2] 筛选后的匹配点
        pts1_binned: [M, 2] 筛选后的匹配点
    """
    if len(pts0) == 0:
        return pts0, pts1
    
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
            
            # 选取置信度最高的 top_k 个点
            indices = np.where(in_grid)[0]
            conf_in_cell = mconf[indices]
            
            if len(indices) > top_k:
                # 按置信度排序，取前 top_k
                top_indices = np.argsort(-conf_in_cell)[:top_k]
                indices = indices[top_indices]
            
            pts0_binned.append(pts0[indices])
            pts1_binned.append(pts1[indices])
    
    if len(pts0_binned) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    
    pts0_binned = np.concatenate(pts0_binned, axis=0)
    pts1_binned = np.concatenate(pts1_binned, axis=0)
    
    return pts0_binned, pts1_binned

def create_chessboard(img1, img2, grid_size=4):
    H, W = img1.shape
    assert img2.shape == (H, W)
    cell_h = H // grid_size
    cell_w = W // grid_size
    chessboard = np.zeros((H, W), dtype=img1.dtype)
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * cell_h
            y_end = (i + 1) * cell_h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w
            if (i + j) % 2 == 0:
                chessboard[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                chessboard[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
    return chessboard

def main():
    parser = argparse.ArgumentParser(description="RoMa Test Script")
    parser.add_argument('--mode', type=str, required=True, choices=['cffa', 'cfoct', 'octfa', 'cfocta'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--savedir', type=str, default='test_run')
    parser.add_argument('--img_size', type=int, default=518)
    args = parser.parse_args()

    ckpt_path = Path(f"results/{args.mode}/{args.name}/best_checkpoint/model.ckpt")
    if not ckpt_path.exists():
        logger.error(f"Ckpt not found: {ckpt_path}")
        return

    save_root = Path(f"test_results/{args.mode}/{args.savedir}")
    save_root.mkdir(parents=True, exist_ok=True)
    
    log_file = save_root / "log.txt"
    logger.add(log_file, rotation="10 MB")
    logger.info(f"Start Testing RoMa Mode: {args.mode}, Name: {args.name}")

    config = get_cfg_defaults()
    config.defrost()
    root_dir = Path(__file__).parent.resolve()
    config.ROMA.DINOV2_PATH = str(root_dir / "pretrained_models" / "dinov2_vits14_pretrain.pth")
    config.freeze()

    model = PL_RoMa.load_from_checkpoint(str(ckpt_path), config=config)
    model.cuda().eval()
    
    # 测试时关闭血管分割图的注意力偏置（lambda=0）
    if hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'layers'):
        for layer in model.model.transformer.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'lambda_vessel'):
                layer.self_attn.lambda_vessel = 0.0
            if hasattr(layer, 'cross_attn') and hasattr(layer.cross_attn, 'lambda_vessel'):
                layer.cross_attn.lambda_vessel = 0.0

    mode = args.mode
    root = DATA_ROOTS[mode]
    if mode == 'cffa':
        base_ds = CFFADataset(root_dir=root, split='test', mode='cf2fa')
    elif mode == 'cfoct':
        base_ds = CFOCTDataset(root_dir=root, split='test', mode='cf2oct')
    elif mode == 'octfa':
        base_ds = OCTFADataset(root_dir=root, split='test', mode='oct2fa')
    elif mode == 'cfocta':
        base_ds = CFOCTADataset(root_dir=root, split='test', mode='cf2octa')
    
    test_ds = RoMaTestDatasetWrapper(base_ds, img_size=args.img_size)
    test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    all_metrics = []
    
    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        pair_name = f"{Path(batch['pair_names'][0][0]).stem}_vs_{Path(batch['pair_names'][1][0]).stem}"
        sample_dir = save_root / pair_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            outputs = model.model(batch_gpu)
        
        # 获取匹配点
        mkpts0 = outputs['mkpts0_f'].cpu().numpy() if 'mkpts0_f' in outputs else np.array([])
        mkpts1 = outputs['mkpts1_f'].cpu().numpy() if 'mkpts1_f' in outputs else np.array([])
        mconf = outputs['mconf'].cpu().numpy() if 'mconf' in outputs else np.array([])
        
        img0 = (batch['image0'][0, 0].cpu().numpy() * 255).astype(np.uint8)
        img1 = (batch['image1'][0, 0].cpu().numpy() * 255).astype(np.uint8)
        img1_gt = (batch['image1_gt'][0, 0].cpu().numpy() * 255).astype(np.uint8)
        h, w = img0.shape
        
        # 使用空间均匀化采样计算单应矩阵
        H_est = np.eye(3, dtype=np.float32)
        if len(mkpts0) >= 4:
            # 应用空间均匀化采样（8x8网格，每格最多5个点）
            mkpts0_binned, mkpts1_binned = spatial_binning(
                mkpts0, mkpts1, mconf, 
                img_size=(h, w), 
                grid_size=8, 
                top_k=5
            )
            
            logger.info(f"Spatial binning: {len(mkpts0)} -> {len(mkpts0_binned)} points")
            
            if len(mkpts0_binned) >= 4:
                try:
                    H_est_found, _ = cv2.findHomography(
                        mkpts0_binned, mkpts1_binned, 
                        cv2.RANSAC, 
                        ransacReprojThreshold=3.0
                    )
                    if H_est_found is not None:
                        H_est = H_est_found.astype(np.float32)
                        logger.info(f"Homography estimated successfully")
                    else:
                        logger.warning(f"Homography estimation failed (returned None)")
                except Exception as e:
                    logger.warning(f"RANSAC failed: {e}")
            else:
                logger.warning(f"Not enough points after spatial binning: {len(mkpts0_binned)}")
        else:
            logger.warning(f"Not enough matches: {len(mkpts0)}")

        try:
            H_inv = np.linalg.inv(H_est)
            img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
        except:
            img1_result = np.zeros_like(img0)

        # 计算MSE：在moving_result和moving_gt之间计算
        try:
            res_f, gt_f = filter_valid_area(img1_result, img1_gt)
            mask = (res_f > 0) & (gt_f > 0)
            if np.any(mask):
                mse = np.mean(((res_f[mask] / 255.0) - (gt_f[mask] / 255.0)) ** 2)
            else:
                mse = np.mean(((img1_result / 255.0) - (img1_gt / 255.0)) ** 2)
        except:
            mse = np.mean(((img1_result / 255.0) - (img1_gt / 255.0)) ** 2)
        
        metrics = {'mse': mse, 'num_matches': len(mkpts0)}
        all_metrics.append(metrics)

        cv2.imwrite(str(sample_dir / "fix.png"), img0)
        cv2.imwrite(str(sample_dir / "moving.png"), img1)
        cv2.imwrite(str(sample_dir / "moving_result.png"), img1_result)
        cv2.imwrite(str(sample_dir / "moving_gt.png"), img1_gt)  # 改为moving_gt
        
        chessboard = create_chessboard(img1_result, img1_gt)  # 比较result和gt
        cv2.imwrite(str(sample_dir / "chessboard.png"), chessboard)
        
        if len(mkpts0) > 0:
            try:
                color = np.zeros((len(mkpts0), 4))
                color[:, 1] = 1.0 
                color[:, 3] = np.clip(mconf * 5, 0.1, 1.0)
                text = [f"Matches: {len(mkpts0)}", f"MSE: {mse:.6f}"]
                fig = make_matching_figure(img0, img1, mkpts0, mkpts1, color, text=text)
                fig.savefig(str(sample_dir / "matches.png"), bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                logger.error(f"Plot failed: {e}")

        logger.info(f"Sample: {pair_name} | Matches: {len(mkpts0)} | MSE: {mse:.6f}")

    if all_metrics:
        avg_mse = np.mean([m['mse'] for m in all_metrics])
        avg_matches = np.mean([m['num_matches'] for m in all_metrics])
        logger.info("="*30)
        logger.info(f"Results for {args.mode}:")
        logger.info(f"  Overall MSE: {avg_mse:.6f}")
        logger.info(f"  Average Matches: {avg_matches:.2f}")
        logger.info("="*30)

if __name__ == "__main__":
    main()
