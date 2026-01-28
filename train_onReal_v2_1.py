import sys
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pprint
from pathlib import Path
from loguru import logger
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
import logging

# 将当前脚本所在目录（根目录）添加到路径中
root_dir = Path(__file__).parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
    
from configs.roma_multimodal import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.lightning.lightning_roma import PL_RoMa
from src.utils.plotting import make_matching_figures

# 导入真实数据集
from dataset.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from dataset.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from dataset.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from dataset.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset

# 日志配置 (对齐 v2.4_mix)
loguru_logger = get_rank_zero_only_logger(logger)
loguru_logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
loguru_logger.add(sys.stderr, format=log_format, level="INFO")

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

def filter_valid_area(img1, img2):
    """筛选有效区域：只保留两张图片都不为纯黑像素的部分，并裁剪使有效区域填满画布"""
    assert img1.shape[:2] == img2.shape[:2], "两张图片的尺寸必须一致"
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

def compute_corner_error(H_est, H_gt, height, width):
    """计算角点误差 (Corner Error)"""
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    corners_homo = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    corners_gt_homo = (H_gt @ corners_homo.T).T
    corners_gt = corners_gt_homo[:, :2] / (corners_gt_homo[:, 2:] + 1e-6)
    corners_est_homo = (H_est @ corners_homo.T).T
    corners_est = corners_est_homo[:, :2] / (corners_est_homo[:, 2:] + 1e-6)
    try:
        errors = np.sqrt(np.sum((corners_est - corners_gt)**2, axis=1))
        mace = np.mean(errors)
    except:
        mace = float('inf')
    return mace

def create_chessboard(img1, img2, grid_size=4):
    """创建棋盘图"""
    H, W = img1.shape
    cell_h = H // grid_size
    cell_w = W // grid_size
    chessboard = np.zeros((H, W), dtype=img1.dtype)
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w
            if (i + j) % 2 == 0:
                chessboard[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                chessboard[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
    return chessboard

class RealDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        
        # 转换范围 [0, 1]
        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2
        
        def to_gray(tensor):
            if tensor.shape[0] == 3:
                gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
                return gray.unsqueeze(0)
            return tensor
            
        fix_gray = to_gray(fix_tensor)
        moving_orig_gray = to_gray(moving_original_tensor)
        moving_gray = to_gray(moving_gt_tensor)
        
        return {
            'image0': fix_gray,
            'image1': moving_orig_gray,
            'image1_gt': moving_gray,
            'T_0to1': T_0to1,
            'pair_names': (os.path.basename(fix_path), os.path.basename(moving_path)),
            'dataset_name': 'RealDataset'
        }

class PL_RoMa_Baseline_Real(PL_RoMa):
    def __init__(self, config, pretrained_ckpt=None):
        super().__init__(config, pretrained_ckpt)
        
    def optimizer_step(self, *args, **kwargs):
        # Baseline 不使用课程学习
        pl.LightningModule.optimizer_step(self, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        # 真实数据训练：Baseline 不传入任何 mask
        model_input = {
            'image0': batch['image0'],
            'image1': batch['image1']
        }
        data = self.model(model_input)
        
        # 将损失函数和后续处理需要的真值信息添加回去
        data['T_0to1'] = batch['T_0to1']
        data['image0'] = batch['image0']
        
        loss, metrics = self.loss_fn(data)
        
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/loss_c', metrics['loss_c'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train/loss_f', metrics['loss_f'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        model_input = {
            'image0': batch['image0'],
            'image1': batch['image1']
        }
        data = self.model(model_input)
        
        # 将损失函数和后续处理需要的真值信息添加回去
        data['T_0to1'] = batch['T_0to1']
        data['image0'] = batch['image0']
        data['dataset_name'] = batch.get('dataset_name', ['RealDataset'])  # 添加 dataset_name
        
        loss, metrics = self.loss_fn(data)
        batch_size = batch['image0'].shape[0]
        H_ests = self._estimate_homography_batch(data, batch, batch_size)
        
        figures = {}
        if getattr(self, 'force_viz', False):
            data['H_est'] = H_ests
            figures = make_matching_figures(data, self.config, mode='evaluation')
            
        # 计算对齐 v2.4_mix 的 auc 指标
        from src.utils.metrics import compute_symmetrical_epipolar_errors
        compute_symmetrical_epipolar_errors(data)
        valid_errs = data['epi_errs'][data['epi_errs'] != float('inf')]
        auc_metrics = {}
        for th in [5, 10, 20]:
            auc = (valid_errs < th).float().mean().item() if len(valid_errs) > 0 else 0.0
            auc_metrics[f'auc@{th}'] = auc

        output = {
            'loss': loss.item(),
            'loss_c': metrics['loss_c'],
            'loss_f': metrics['loss_f'],
            'H_est': H_ests,
            'figures': figures,
            'feat_c0': data.get('feat_c0'),
            'feat_c1': data.get('feat_c1'),
            'feat_f0': data.get('feat_f0'),
            'feat_f1': data.get('feat_f1'),
            'x_adapted0': data.get('x_adapted0'),
            'x_adapted1': data.get('x_adapted1'),
            **auc_metrics
        }
        self.validation_step_outputs.append(output)
        return output

class MultimodalValidationCallback(Callback):
    def __init__(self, args, result_dir):
        super().__init__()
        self.args = args
        self.result_dir = Path(result_dir)
        self.best_val = -1.0
        self.epoch_mses = []
        self.epoch_maces = []
        
    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []
        self.epoch_maces = []
        pl_module.validation_step_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        batch_mses, batch_maces = self._process_batch(trainer, pl_module, batch, outputs, None, save_images=False)
        self.epoch_mses.extend(batch_mses)
        self.epoch_maces.extend(batch_maces)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        display_metrics = {}
        for k in ['loss', 'train/loss_c', 'train/loss_f']:
            epoch_key = f"{k}_epoch"
            if epoch_key in metrics:
                name = k.replace('train/', '')
                display_metrics[name] = metrics[epoch_key].item()
            elif k in metrics:
                name = k.replace('train/', '')
                display_metrics[name] = metrics[k].item()
        if display_metrics:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
            loguru_logger.info(f"Epoch {epoch} 训练总结 >> {metric_str}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.epoch_mses: return
        
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses)
        avg_mace = sum(self.epoch_maces) / len(self.epoch_maces) if self.epoch_maces else float('inf')
        
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        
        display_metrics = {'mse_real': avg_mse, 'mace_real': avg_mace}
        for k in ['auc@5', 'auc@10', 'auc@20']:
            outputs = pl_module.validation_step_outputs
            if len(outputs) > 0:
                display_metrics[k] = sum(x[k] for x in outputs) / len(outputs)
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
        loguru_logger.info(f"Epoch {epoch} 验证总结 >> {metric_str}")
        
        pl_module.log("val_mse", avg_mse, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("val_mace", avg_mace, on_epoch=True, prog_bar=True, logger=True)
        
        latest_path = self.result_dir / "latest_checkpoint"
        latest_path.mkdir(exist_ok=True)
        trainer.save_checkpoint(latest_path / "model.ckpt")
        with open(latest_path / "log.txt", "w") as f:
            f.write(f"Epoch: {epoch}\nLatest MSE: {avg_mse:.6f}\nLatest MACE: {avg_mace:.4f}")
            
        is_best = False
        auc5 = display_metrics.get('auc@5', 0.0)
        auc10 = display_metrics.get('auc@10', 0.0)
        auc20 = display_metrics.get('auc@20', 0.0)
        combined_auc = (auc5 + auc10 + auc20) / 3.0
        
        # 将 combined_auc log 到模型中，以便 EarlyStopping 监控
        pl_module.log('combined_auc', combined_auc, on_epoch=True, prog_bar=True, logger=True)

        if combined_auc > self.best_val:
            self.best_val = combined_auc
            is_best = True
            best_path = self.result_dir / "best_checkpoint"
            best_path.mkdir(exist_ok=True)
            trainer.save_checkpoint(best_path / "model.ckpt")
            with open(best_path / "log.txt", "w") as f:
                f.write(f"Epoch: {epoch}\nBest Combined AUC: {combined_auc:.4f}\nAUC@10: {auc10:.4f}\nMSE: {avg_mse:.6f}\nMACE: {avg_mace:.4f}")
            loguru_logger.info(f"发现新的最优模型! Epoch {epoch}, 综合 AUC: {combined_auc:.4f}")

        if is_best or (epoch % 5 == 0):
            self._trigger_visualization(trainer, pl_module, epoch, is_best)

    def _trigger_visualization(self, trainer, pl_module, epoch, is_best=False):
        loguru_logger.info(f"正在为 Epoch {epoch} 生成可视化结果...")
        pl_module.force_viz = True
        
        suffix = "_best" if is_best else ""
        save_path = self.result_dir / f"epoch_{epoch}{suffix}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        dl = trainer.val_dataloaders
        if dl is None: return
        
        pl_module.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dl):
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = pl_module.validation_step(batch, batch_idx)
                self._process_batch(trainer, pl_module, batch, outputs, save_path, save_images=True)
                        
        pl_module.force_viz = False

    def _process_batch(self, trainer, pl_module, batch, outputs, epoch_dir, save_images=False):
        batch_size = batch['image0'].shape[0]
        mses = []
        maces = []
        H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
        Ts_gt = batch['T_0to1'].cpu().numpy()
        pair_names0 = batch['pair_names'][0]
        pair_names1 = batch['pair_names'][1]
        
        for i in range(batch_size):
            H_est = H_ests[i]
            img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            ref_key = 'image1_gt' if 'image1_gt' in batch else 'image1'
            img1_gt = (batch[ref_key][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            h, w = img0.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except:
                img1_result = img1.copy()
                
            try:
                res_f, orig_f = filter_valid_area(img1_result, img1_gt)
                mask = (res_f > 0)
                mse = np.mean(((res_f[mask]/255.)-(orig_f[mask]/255.))**2) if np.any(mask) else 0.0
            except:
                mse = 0.0
            mses.append(mse)
            
            mace = compute_corner_error(H_est, Ts_gt[i], h, w)
            maces.append(mace)
            
            if not save_images: continue
                
            sample_name = f"{Path(pair_names0[i]).stem}_vs_{Path(pair_names1[i]).stem}"
            save_path = epoch_dir / sample_name
            save_path.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(save_path / "fix.png"), img0)
            cv2.imwrite(str(save_path / "moving.png"), img1)
            cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
            
            try:
                cb = create_chessboard(img1_result, img0)
                cv2.imwrite(str(save_path / "chessboard.png"), cb)
            except: pass
            
            try:
                if 'figures' in outputs and 'evaluation' in outputs['figures'] and len(outputs['figures']['evaluation']) > i:
                    fig = outputs['figures']['evaluation'][i]
                    fig.savefig(str(save_path / "matches.png"), bbox_inches='tight')
                    plt.close(fig)
            except: pass

            # [新增] 特征图可视化
            from src.utils.plotting import visualize_feature_maps
            try:
                if 'feat_c0' in outputs and outputs['feat_c0'] is not None:
                    visualize_feature_maps(outputs['feat_c0'][i:i+1], save_path / "feat_coarse_fix.png", title="Coarse Fix")
                if 'feat_c1' in outputs and outputs['feat_c1'] is not None:
                    visualize_feature_maps(outputs['feat_c1'][i:i+1], save_path / "feat_coarse_mov.png", title="Coarse Mov")
                if 'feat_f0' in outputs and outputs['feat_f0'] is not None:
                    visualize_feature_maps(outputs['feat_f0'][i:i+1], save_path / "feat_fine_fix.png", title="Fine Fix")
                if 'feat_f1' in outputs and outputs['feat_f1'] is not None:
                    visualize_feature_maps(outputs['feat_f1'][i:i+1], save_path / "feat_fine_mov.png", title="Fine Mov")
                
                # [新增] 适配图可视化 (DINOv2 Input)
                if 'x_adapted0' in outputs and outputs['x_adapted0'] is not None:
                    img_ada = (outputs['x_adapted0'][i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(str(save_path / "adapted_fix.png"), cv2.cvtColor(img_ada, cv2.COLOR_RGB2BGR))
                if 'x_adapted1' in outputs and outputs['x_adapted1'] is not None:
                    img_ada = (outputs['x_adapted1'][i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(str(save_path / "adapted_mov.png"), cv2.cvtColor(img_ada, cv2.COLOR_RGB2BGR))
            except: pass
            
        return mses, maces

class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)

def parse_args():
    parser = argparse.ArgumentParser(description="RoMa Baseline Training on Real Data")
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa', 'cfocta'])
    parser.add_argument('--name', '-n', type=str, default='roma_baseline_real')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gpus', type=str, default='1')
    parser.add_argument('--pretrained_ckpt', type=str, default='weights/romav2.pt.1', help='Path to pretrained checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()
    config = get_cfg_defaults()
    
    config.defrost()
    config.ROMA.LAMBDA_VESSEL = 0.0
    config.DATASET.MGDPT_IMG_RESIZE = args.img_size
    config.TRAINER.SEED = 66
    config.freeze()
    
    pl.seed_everything(config.TRAINER.SEED)
    _n_gpus = setup_gpus(args.gpus)
    
    result_dir = Path(f"results/{args.mode}/{args.name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    loguru_logger.add(result_dir / "log.txt", enqueue=True, mode="a")
    
    model = PL_RoMa_Baseline_Real(config, pretrained_ckpt=args.pretrained_ckpt)
    
    # 根据模式选择真实数据集
    if args.mode == 'cffa':
        train_base = CFFADataset(root_dir='dataset/operation_pre_filtered_cffa', split='train', mode='fa2cf')
        val_base = CFFADataset(root_dir='dataset/operation_pre_filtered_cffa', split='val', mode='fa2cf')
    elif args.mode == 'cfoct':
        train_base = CFOCTDataset(root_dir='dataset/operation_pre_filtered_cfoct', split='train', mode='cf2oct')
        val_base = CFOCTDataset(root_dir='dataset/operation_pre_filtered_cfoct', split='val', mode='cf2oct')
    elif args.mode == 'octfa':
        train_base = OCTFADataset(root_dir='dataset/operation_pre_filtered_octfa', split='train', mode='fa2oct')
        val_base = OCTFADataset(root_dir='dataset/operation_pre_filtered_octfa', split='val', mode='fa2oct')
    elif args.mode == 'cfocta':
        train_base = CFOCTADataset(root_dir='dataset/CF_OCTA_v2_repaired', split='train', mode='cf2octa')
        val_base = CFOCTADataset(root_dir='dataset/CF_OCTA_v2_repaired', split='val', mode='cf2octa')

    train_loader = torch.utils.data.DataLoader(RealDatasetWrapper(train_base), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(RealDatasetWrapper(val_base), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.name)
    val_callback = MultimodalValidationCallback(args, str(result_dir))
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # 对齐 v2.4_mix 监控 combined_auc (auc@5, 10, 20 的平均值)
    early_stop = DelayedEarlyStopping(start_epoch=50, monitor='combined_auc', patience=10, mode='max')
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if _n_gpus > 0 else "cpu",
        devices=_n_gpus if _n_gpus > 0 else "auto",
        callbacks=[val_callback, lr_monitor, early_stop],
        logger=tb_logger,
        check_val_every_n_epoch=1
    )
    
    loguru_logger.info(f"Starting Real Baseline Training: {args.name} in mode {args.mode}")
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
