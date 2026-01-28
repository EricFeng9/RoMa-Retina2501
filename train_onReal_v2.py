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

# 日志配置
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
        loguru_logger.log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

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
            **auc_metrics
        }
        self.validation_step_outputs.append(output)
        return output

class MultimodalValidationCallback(Callback):
    def __init__(self, args, result_dir):
        super().__init__()
        self.args = args
        self.result_dir = Path(result_dir)
        self.best_mse = float('inf')
        self.epoch_mses = []
        
    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []
        pl_module.validation_step_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        batch_size = batch['image0'].shape[0]
        H_ests = outputs['H_est']
        
        for i in range(batch_size):
            H_est = H_ests[i]
            img1_warp = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1_gt = (batch['image1_gt'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            h, w = img1_gt.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_res = cv2.warpPerspective(img1_warp, H_inv, (w, h))
                mse = np.mean(((img1_res/255.0) - (img1_gt/255.0))**2)
                self.epoch_mses.append(mse)
            except:
                pass

    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = pl_module.validation_step_outputs
        if len(outputs) > 0:
            auc5 = sum(x['auc@5'] for x in outputs) / len(outputs)
            auc10 = sum(x['auc@10'] for x in outputs) / len(outputs)
            auc20 = sum(x['auc@20'] for x in outputs) / len(outputs)
            combined_auc = (auc5 + auc10 + auc20) / 3.0
            pl_module.log('auc@10', auc10, prog_bar=True, logger=True)
            pl_module.log('combined_auc', combined_auc, prog_bar=True, logger=True)

        if not self.epoch_mses: return
        
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses)
        epoch = trainer.current_epoch + 1
        
        loguru_logger.info(f"Epoch {epoch} Real Validation: MSE={avg_mse:.6f}")
        pl_module.log("val_mse_real", avg_mse, logger=True)
        
        # 基于 Combined AUC 的最优模型保存
        metrics = trainer.callback_metrics
        combined_auc_val = metrics.get('combined_auc', 0.0)
        
        is_best = False
        if combined_auc_val > (getattr(self, 'best_auc', -1.0)):
            self.best_auc = combined_auc_val
            is_best = True
            loguru_logger.info(f"New Best Real Model! Combined AUC={combined_auc_val:.4f}, MSE={avg_mse:.6f}")
            best_path = self.result_dir / "best_checkpoint"
            best_path.mkdir(exist_ok=True)
            trainer.save_checkpoint(best_path / "model.ckpt")

        # 仅在 best checkpoint 或 每 5 个 epoch 触发可视化
        if is_best or epoch % 5 == 0:
            self._trigger_visualization(trainer, pl_module, epoch, is_best=is_best)

    def _trigger_visualization(self, trainer, pl_module, epoch, is_best=False):
        suffix = "_best" if is_best else ""
        loguru_logger.info(f"Triggering Visualization for Epoch {epoch}{suffix}...")
        pl_module.force_viz = True
        save_path = self.result_dir / f"epoch_{epoch}{suffix}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        dl = trainer.val_dataloaders
        if dl is None: return
        
        pl_module.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dl):
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = pl_module.validation_step(batch, batch_idx)
                
                # 保存可视化图像
                batch_size = batch['image0'].shape[0]
                H_ests = outputs['H_est']
                pair_names0 = batch['pair_names'][0]
                pair_names1 = batch['pair_names'][1]
                
                for i in range(batch_size):
                    sample_name = f"{Path(pair_names0[i]).stem}_vs_{Path(pair_names1[i]).stem}"
                    img_dir = save_path / sample_name
                    img_dir.mkdir(parents=True, exist_ok=True)
                    
                    img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
                    img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
                    
                    h, w = img0.shape
                    try:
                        H_inv = np.linalg.inv(H_ests[i])
                        img1_res = cv2.warpPerspective(img1, H_inv, (w, h))
                    except:
                        img1_res = img1.copy()
                        
                    cv2.imwrite(str(img_dir / "fix.png"), img0)
                    cv2.imwrite(str(img_dir / "moving_warp.png"), img1)
                    cv2.imwrite(str(img_dir / "result.png"), img1_res)
                    
                    if 'figures' in outputs and 'evaluation' in outputs['figures'] and len(outputs['figures']['evaluation']) > i:
                        fig = outputs['figures']['evaluation'][i]
                        fig.savefig(str(img_dir / "matches.png"), bbox_inches='tight')
                        plt.close(fig)
        pl_module.force_viz = False

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
    
    model = PL_RoMa_Baseline_Real(config)
    
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
