"""
RoMa 多模态眼底图像配准训练脚本
已移动至根目录，自动处理路径
"""
import sys
import os
from pathlib import Path

# 将当前脚本所在目录（根目录）添加到路径中
root_dir = Path(__file__).parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# 设置 Torch Hub 缓存到数据盘，防止撑爆 /home
os.environ['TORCH_HOME'] = str(root_dir / ".cache" / "torch")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import argparse
import pprint
from loguru import logger
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, EarlyStopping, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
import logging

from configs.roma_multimodal import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.lightning.lightning_roma import PL_RoMa
from data.FIVES_extract_v2.FIVES_extract_v2 import MultiModalDataset
from src.utils.plotting import make_matching_figures

# 数据集根目录：指向本地 data 目录
DATA_ROOT = "/data/student/Fengjunming/LoFTR/data/FIVES_extract_v2"  # 保持原路径或按需修改

# 配置日志格式
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
logging.getLogger("fsspec").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description="RoMa 多模态眼底图像配准训练脚本")
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa', 'cfocta'], help='配准模式')
    parser.add_argument('--name', '-n', type=str, default='roma_multimodal_fives', help='本次训练的名称')
    parser.add_argument('--batch_size', type=int, default=4, help='每个 GPU 的 Batch Size')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--img_size', type=int, default=518, help='图像输入尺寸 (需为 14 的倍数，如 518, 448)')
    parser.add_argument('--vessel_sigma', type=float, default=6.0, help='血管高斯软掩码的 σ（像素单位）')
    parser.add_argument('--main_cfg_path', type=str, default=None, help='主配置文件路径')
    
    parser.add_argument('--max_epochs', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--gpus', type=str, default='1', help='使用的 GPU')
    
    return parser.parse_args()

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*main thread is not in main loop.*")

class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MultiModalDataset(
                DATA_ROOT, mode=self.args.mode, split='train', 
                img_size=self.args.img_size, vessel_sigma=self.args.vessel_sigma)
        
        if stage in ('fit', 'validate') or stage is None:
            self.val_dataset = MultiModalDataset(
                DATA_ROOT, mode=self.args.mode, split='val', 
                img_size=self.args.img_size, vessel_sigma=self.args.vessel_sigma)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, **self.loader_params)

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

def create_chessboard(img1, img2, grid_size=4):
    """创建棋盘图"""
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

class LitProgressBar(TQDMProgressBar):
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch + 1}")

class DelayedEarlyStopping(EarlyStopping):
    """延迟早停：只在指定epoch后才开始计数"""
    def __init__(self, start_epoch=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
        
    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            return  # 不执行早停检查
        super().on_validation_end(trainer, pl_module)

class RoMaValidationCallback(Callback):
    """RoMa 验证回调：保存图像、计算MSE、管理权重"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.best_val_mse = float('inf')
        self.result_dir = Path(f"results/{args.mode}/{args.name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_mses = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            sub_dir = "step0"
        elif trainer.current_epoch == 0 and not trainer.training:
            sub_dir = "initial"
        else:
            sub_dir = f"epoch{trainer.current_epoch + 1}"
            
        epoch_dir = self.result_dir / sub_dir
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        batch_mses = self._save_batch_results(trainer, pl_module, batch, outputs, epoch_dir)
        self.epoch_mses.extend(batch_mses)

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
        if not self.epoch_mses:
            return
            
        if trainer.current_epoch == 0 and not trainer.training:
            epoch_name = "Initial"
            epoch_num = 0
        else:
            epoch_num = trainer.current_epoch + 1
            epoch_name = f"Epoch {epoch_num}"
            
        metrics = trainer.callback_metrics
        
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses)
        
        display_metrics = {'mse_viz': avg_mse}
        
        for k in ['auc@5', 'auc@10', 'auc@20']:
            if k in metrics:
                display_metrics[k] = metrics[k].item()
        
        for k in ['val/avg_loss_c', 'val/avg_loss_f', 'val/avg_loss']:
            if k in metrics:
                name = k.replace('val/avg_', '')
                display_metrics[name] = metrics[k].item()
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
        loguru_logger.info(f"{epoch_name} 验证总结 >> {metric_str}")
        
        pl_module.log("val_mse", avg_mse, on_epoch=True, prog_bar=False, logger=True)
        
        # 更新最新权重
        latest_path = self.result_dir / "latest_checkpoint"
        latest_path.mkdir(exist_ok=True)
        trainer.save_checkpoint(latest_path / "model.ckpt")
        with open(latest_path / "log.txt", "w") as f:
            f.write(f"Epoch: {epoch_num}\nLatest MSE: {avg_mse:.6f}")
            
        # 更新最优权重（忽略 initial 验证，从 epoch 5 开始计入）
        if epoch_num >= 5 and avg_mse < self.best_val_mse:
            self.best_val_mse = avg_mse
            best_path = self.result_dir / "best_checkpoint"
            best_path.mkdir(exist_ok=True)
            trainer.save_checkpoint(best_path / "model.ckpt")
            with open(best_path / "log.txt", "w") as f:
                f.write(f"Epoch: {epoch_num}\nBest MSE: {avg_mse:.6f}")
            loguru_logger.info(f"发现新的最优模型! {epoch_name}, MSE: {avg_mse:.6f}")
        elif epoch_num == 0:
            loguru_logger.info(f"Initial 验证不计入 best score")

    def _save_batch_results(self, trainer, pl_module, batch, outputs, epoch_dir):
        batch_size = batch['image0'].shape[0]
        mses = []
        
        # 处理 H_est，确保它是正确的格式
        if 'H_est' not in outputs:
            loguru_logger.warning(f"⚠️ H_est 未在 outputs 中找到，使用单位矩阵")
            H_ests = [np.eye(3) for _ in range(batch_size)]
        else:
            H_est_raw = outputs['H_est']
            
            # 先判断是否是列表或元组
            if isinstance(H_est_raw, (list, tuple)):
                H_ests = list(H_est_raw)
                if len(H_ests) < batch_size:
                    loguru_logger.warning(f"⚠️ H_est 长度 {len(H_ests)} < batch_size {batch_size}，用单位矩阵补齐")
                    H_ests.extend([np.eye(3) for _ in range(batch_size - len(H_ests))])
            else:
                # 如果是 tensor，转为 numpy
                if torch.is_tensor(H_est_raw):
                    H_est_raw = H_est_raw.detach().cpu().numpy()
                
                # 如果是单个矩阵 (3, 3)，复制 batch_size 次
                if H_est_raw.ndim == 2:
                    H_ests = [H_est_raw.copy() for _ in range(batch_size)]
                # 如果是 (batch_size, 3, 3)，分割为列表
                elif H_est_raw.ndim == 3 and H_est_raw.shape[0] == batch_size:
                    H_ests = [H_est_raw[i] for i in range(batch_size)]
                else:
                    loguru_logger.warning(f"⚠️ H_est 格式异常: shape={H_est_raw.shape if hasattr(H_est_raw, 'shape') else type(H_est_raw)}，使用单位矩阵")
                    H_ests = [np.eye(3) for _ in range(batch_size)]
            
            num_identity = sum([np.allclose(H, np.eye(3)) for H in H_ests])
            if num_identity > 0:
                loguru_logger.warning(f"⚠️ Batch 中有 {num_identity}/{batch_size} 个样本的 H_est 是单位矩阵")
        
        pair_names0 = batch['pair_names'][0]
        pair_names1 = batch['pair_names'][1]

        for i in range(batch_size):
            sample_name = f"{Path(pair_names0[i]).stem}_vs_{Path(pair_names1[i]).stem}"
            save_path = epoch_dir / sample_name
            save_path.mkdir(parents=True, exist_ok=True)
            
            img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1_origin = (batch['image1_origin'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            # 保存血管掩码
            if 'mask0' in batch:
                m0 = batch['mask0'][i].cpu().numpy()
                m1 = batch['mask1'][i].cpu().numpy()
                if m0.ndim == 3: m0 = m0[0]
                if m1.ndim == 3: m1 = m1[0]
                cv2.imwrite(str(save_path / "mask0_vessel.png"), (m0 * 255).astype(np.uint8))
                cv2.imwrite(str(save_path / "mask1_vessel_warped.png"), (m1 * 255).astype(np.uint8))
            
            H_est = H_ests[i]
            
            # 配准
            h, w = img0.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except Exception as e:
                loguru_logger.error(f"样本 {sample_name}: H_est 求逆失败: {e}")
                img1_result = np.zeros_like(img0)
                
            # 计算 MSE
            try:
                res_f, orig_f = filter_valid_area(img1_result, img1_origin)
                mask = (res_f > 10) & (orig_f > 10)
                if np.any(mask):
                    mse = np.mean(((res_f[mask] / 255.0) - (orig_f[mask] / 255.0)) ** 2)
                else:
                    mse = np.mean(((img1_result / 255.0) - (img1_origin / 255.0)) ** 2)
            except:
                mse = np.mean(((img1_result / 255.0) - (img1_origin / 255.0)) ** 2)
            
            mses.append(mse)
            
            # 保存图像
            cv2.imwrite(str(save_path / "fix.png"), img0)
            cv2.imwrite(str(save_path / "moving.png"), img1)
            cv2.imwrite(str(save_path / "moving_origin.png"), img1_origin)
            cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
            
            # 保存棋盘图
            try:
                chessboard = create_chessboard(img1_result, img1_origin)
                cv2.imwrite(str(save_path / "chessboard.png"), chessboard)
            except:
                pass
            
            # 保存匹配可视化
            try:
                mode = pl_module.config.TRAINER.PLOT_MODE
                if 'figures' in outputs and mode in outputs['figures'] and len(outputs['figures'][mode]) > i:
                    fig = outputs['figures'][mode][i]
                    fig.savefig(str(save_path / "matches.png"), bbox_inches='tight')
                    plt.close(fig)
                plt.close('all')
            except Exception as e:
                pass
                
        return mses

def main():
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # 初始化配置
    config = get_cfg_defaults()
    if args.main_cfg_path:
        config.merge_from_file(args.main_cfg_path)
    
    # 确保 PLOT_MATCHES_ALPHA 属性存在(兼容性修复)
    if not hasattr(config.TRAINER, 'PLOT_MATCHES_ALPHA'):
        config.defrost()
        config.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'
        config.freeze()
    
    # 日志文件
    result_dir = Path(f"results/{args.mode}/{args.name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / "log.txt"
    loguru_logger.add(log_file, enqueue=True, mode="a")
    loguru_logger.info(f"项目根目录: {root_dir}")
    loguru_logger.info(f"日志将同时保存到: {log_file}")

    # --- 关键参数配置区 (可在脚本内直接调整) ---
    config.defrost()
    
    # 模型参数
    config.ROMA.DINOV2_PATH = str(root_dir / "pretrained_models" / "dinov2_vits14_pretrain.pth")
    config.ROMA.LAMBDA_VESSEL = 1.0     # 血管增强强度 (在有血管掩码的生成数据集上建议设为 1.0)
    config.ROMA.CONF_THRESH = 0.01
    config.ROMA.TOP_K = 1000
    
    # 损失权重
    config.ROMA.LOSS.WEIGHT_COARSE = 1.0
    config.ROMA.LOSS.WEIGHT_FINE = 1.0
    
    # 训练参数
    config.DATASET.MGDPT_IMG_RESIZE = args.img_size
    config.TRAINER.SEED = 66
    config.TRAINER.PLOT_MODE = 'evaluation'
    # ---------------------------------------
    pl.seed_everything(config.TRAINER.SEED)

    # GPU 设置
    _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = max(_n_gpus, 1) * getattr(args, 'num_nodes', 1)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    
    config.freeze() # 在初始化模型前最后锁定
    
    # 初始化模型与数据
    model = PL_RoMa(config)
    data_module = MultimodalDataModule(args, config)

    # 训练器
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.name)
    val_callback = RoMaValidationCallback(args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # 早停机制：从epoch 50开始计数，patience=8（即8次验证 = 40个epoch）
    early_stop_callback = DelayedEarlyStopping(
        start_epoch=50,
        monitor='val_mse', 
        patience=8,
        verbose=True,
        mode='min',
        min_delta=0.0001
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=5,
        num_sanity_val_steps=3,
        callbacks=[val_callback, lr_monitor, early_stop_callback, LitProgressBar()],
        logger=tb_logger,
        strategy=DDPStrategy(find_unused_parameters=False) if _n_gpus > 1 else "auto",
        accelerator="gpu" if _n_gpus > 0 else "cpu",
        devices=_n_gpus if _n_gpus > 0 else "auto"
    )

    loguru_logger.info(f"开始训练前先运行一次完整的验证...")
    trainer.validate(model, datamodule=data_module)

    loguru_logger.info(f"开始训练 RoMa: {args.name} (模式: {args.mode})")
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()
