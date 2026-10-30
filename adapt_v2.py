# adapt_v2.py
# MetaHFC3 Phase2：目标域适应 (多 Seed + 自定义保存路径版本)
#
# 相比 adapt.py 的改动：
#   1. 模型换为 WarmStartUNet
#   2. forward 调用改为 model(x, s=stat_vec, hfc_weight=1.0)
#   3. num_workers=0 避免 NFS .nfs 临时文件冲突
#   4. 引入多 Seed 实验，分别保存权重与日志，最终输出 Mean ± Std
#   5. 兼容了新的 HAM10000 数据集划分路径
#   6. [新增] 命令行支持 --save_dir 动态指定节点保存路径

import copy
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

from config    import TARGET_DOMAINS, ADAPT_CONFIG, MODEL_CONFIG, META_CONFIG
from dataset   import SegDataset
from loss      import BCEDiceLoss
from evaluate  import evaluate_loader
from hfc_utils import compute_hfc_from_images, normalize_hfc

from meta_train_v2 import WarmStartUNet


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DomainAdapterV2:
    def __init__(self, meta_ckpt_path, target_name,
                 train_img_dir, train_mask_dir,
                 test_img_dir,  test_mask_dir,
                 ratio_name, seed,
                 cfg=ADAPT_CONFIG, model_cfg=MODEL_CONFIG):

        self.cfg         = cfg
        self.target_name = target_name
        self.ratio_name  = ratio_name
        self.seed        = seed
        self.device      = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 针对当前 seed 进行固定
        seed_everything(self.seed)

        # ------------------------------------------------------------
        # 修改保存目录，加入 seed 维度避免冲突
        # ------------------------------------------------------------
        self.save_dir = Path(cfg['save_dir']) / target_name / ratio_name / f"seed_{self.seed}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.save_dir / 'adapt_log.csv'

        # ============================================================
        # 1. 加载元权重
        # ============================================================
        print(f"\n[适应V2 | Seed: {self.seed}] 加载元权重: {meta_ckpt_path}")
        ckpt = torch.load(meta_ckpt_path, map_location=self.device,
                          weights_only=False)
        self.global_mean = ckpt['global_mean'].to(self.device)
        self.global_std  = ckpt['global_std'].to(self.device)

        # ============================================================
        # 2. 初始化 WarmStartUNet
        # ============================================================
        self.model_B = WarmStartUNet(
            n_channels=model_cfg['n_channels'],
            n_classes= model_cfg['n_classes'],
            base_ch=   model_cfg['base_ch'],
            stat_dim=  model_cfg['stat_dim'],
            n_domains= model_cfg['n_domains'],
        ).to(self.device)
        self.model_B.load_state_dict(ckpt['model'])
        print("  θ_meta 加载完成")

        # ============================================================
        # 3. 数据集 + DataLoader
        # ============================================================
        self.train_ds = SegDataset(
            train_img_dir, train_mask_dir,
            img_size=cfg['img_size'], is_train=True, return_01=False
        )
        self.test_ds = SegDataset(
            test_img_dir, test_mask_dir,
            img_size=cfg['img_size'], is_train=False, return_01=False
        )
        n_train = len(self.train_ds)
        n_test  = len(self.test_ds)

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=min(cfg['batch_size'], n_train),
            shuffle=True, num_workers=0, pin_memory=False, drop_last=False
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=cfg['batch_size'],
            shuffle=False, num_workers=0, pin_memory=False
        )
        print(f"  训练集: {n_train}张 | 测试集: {n_test}张 (workers=0,NFS)")

        # ============================================================
        # 4. 计算 HFC 向量
        # ============================================================
        print("  计算目标域 HFC 向量...")
        raw_01_list = []
        for img_path, _ in self.train_ds.pairs:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((cfg['img_size'], cfg['img_size']), Image.BILINEAR)
            raw_01_list.append(TF.to_tensor(img))

        raw_01_t      = torch.stack(raw_01_list)
        hfc_raw       = compute_hfc_from_images(raw_01_t, self.device)
        self.stat_vec = normalize_hfc(
            hfc_raw, self.global_mean, self.global_std, self.device
        )
        print(f"  HFC 向量: shape={self.stat_vec.shape}, "
              f"min={self.stat_vec.min():.3f}, max={self.stat_vec.max():.3f}")

        # ============================================================
        # 5. AdaBN + 冻结 BN
        # ============================================================
        print("  执行 AdaBN...")
        self.model_B.apply_adabn(raw_01_list, self.device,
                                  stat_vec=self.stat_vec)
        self.model_B.freeze_bn_stats()
        print("  BN 统计量已冻结")

        # ============================================================
        # 6. frozen copy + EMA
        # ============================================================
        self.model_A_frozen = copy.deepcopy(self.model_B)
        for p in self.model_A_frozen.parameters():
            p.requires_grad_(False)
        self.model_A_frozen.eval()

        self.model_EMA = copy.deepcopy(self.model_B)
        for p in self.model_EMA.parameters():
            p.requires_grad_(False)
        self.model_EMA.eval()
        print("  model_A_frozen / model_EMA 已创建")

        # ============================================================
        # 7. 优化器 + 调度器
        # ============================================================
        self.optimizer = optim.AdamW([
            {'params': self.model_B.get_encoder_params(),
             'lr': cfg['lr_encoder']},
            {'params': self.model_B.get_decoder_params(),
             'lr': cfg['lr_decoder']},
        ], weight_decay=cfg['weight_decay'])

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg['epochs'], eta_min=1e-6
        )
        self.seg_loss     = BCEDiceLoss()
        self.consist_loss = nn.MSELoss()

        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'train_loss', 'l_sup', 'l_consist',
                'val_dice', 'val_iou', 'lr'
            ])

    def _update_ema(self):
        m = self.cfg['ema_momentum']
        with torch.no_grad():
            for p_ema, p_b in zip(self.model_EMA.parameters(),
                                   self.model_B.parameters()):
                p_ema.data = m * p_ema.data + (1 - m) * p_b.data

    def _get_mixed_anchor_pred(self, imgs, epoch):
        warmup   = self.cfg.get('progress_warmup', 100)
        progress = min(1.0, epoch / warmup)
        with torch.no_grad():
            pred_A   = torch.sigmoid(
                self.model_A_frozen(imgs, s=self.stat_vec, hfc_weight=1.0)
            )
            pred_EMA = torch.sigmoid(
                self.model_EMA(imgs, s=self.stat_vec, hfc_weight=1.0)
            )
        return (1 - progress) * pred_A + progress * pred_EMA

    def train_epoch(self, epoch):
        total_loss = sup_sum = consist_sum = 0.
        n_batches  = len(self.train_loader)

        for batch in self.train_loader:
            self.model_B.train()
            self.model_B.freeze_bn_stats()

            imgs  = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            pred_B = self.model_B(imgs, s=self.stat_vec, hfc_weight=1.0)
            l_sup  = self.seg_loss(pred_B, masks)

            pred_anchor = self._get_mixed_anchor_pred(imgs, epoch)
            l_consist   = self.consist_loss(
                torch.sigmoid(pred_B), pred_anchor.detach()
            )

            loss = l_sup + self.cfg['lambda_consist'] * l_consist
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model_B.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            self._update_ema()

            total_loss += loss.item()
            sup_sum    += l_sup.item()
            consist_sum += l_consist.item()

        return (total_loss / n_batches,
                sup_sum    / n_batches,
                consist_sum / n_batches)

    def run(self):
        print(f"\n{'='*60}")
        print(f"Phase2 适应V2：{self.target_name} - {self.ratio_name} (Seed: {self.seed})")
        print(f"  训练集: {len(self.train_ds)} | 测试集: {len(self.test_ds)}")
        print(f"{'='*60}")

        best_dice    = 0.
        patience_ctr = 0

        for epoch in range(1, self.cfg['epochs'] + 1):
            t_loss, s_loss, c_loss = self.train_epoch(epoch)

            self.model_B.eval()
            metrics = _evaluate(self.model_B, self.test_loader,
                                self.device, self.stat_vec)
            val_dice = metrics['dice']
            val_iou  = metrics['iou']

            self.scheduler.step()
            lr = self.optimizer.param_groups[1]['lr']

            with open(self.log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch,
                    f"{t_loss:.5f}", f"{s_loss:.5f}", f"{c_loss:.5f}",
                    f"{val_dice:.4f}", f"{val_iou:.4f}", f"{lr:.8f}"
                ])

            if epoch % 10 == 0:
                progress = min(1.0, epoch / self.cfg.get('progress_warmup', 100))
                print(f"  [Ep {epoch:3d}/{self.cfg['epochs']}] "
                      f"loss={t_loss:.4f} "
                      f"(sup={s_loss:.4f} con={c_loss:.4f}) | "
                      f"Dice={val_dice:.4f} IoU={val_iou:.4f} | "
                      f"EMA_prog={progress:.2f}")

            if val_dice > best_dice:
                best_dice    = val_dice
                patience_ctr = 0
                torch.save(self.model_B.state_dict(),
                           self.save_dir / 'best_model.pth')
                print(f"  *** Ep {epoch}: 新最佳 Dice={best_dice:.4f}")
            else:
                patience_ctr += 1

            if patience_ctr >= self.cfg['patience']:
                print(f"  [早停] Ep {epoch}, 最佳 Dice={best_dice:.4f}")
                break

        print(f"\n[完成] {self.target_name}-{self.ratio_name} (Seed: {self.seed}): "
              f"最佳 Dice={best_dice:.4f}")
        return best_dice


@torch.no_grad()
def _evaluate(model, loader, device, stat_vec):
    """WarmStartUNet 专用评估（传入 hfc_weight=1.0）"""
    model.eval()
    total_TP = total_FP = total_FN = total_TN = 0.

    for batch in loader:
        imgs  = batch['image'].to(device)
        masks = batch['mask'].to(device)
        preds = model(imgs, s=stat_vec, hfc_weight=1.0)

        pred_bin = (torch.sigmoid(preds) > 0.5).float()
        total_TP += (pred_bin * masks).sum().item()
        total_FP += (pred_bin * (1 - masks)).sum().item()
        total_FN += ((1 - pred_bin) * masks).sum().item()
        total_TN += ((1 - pred_bin) * (1 - masks)).sum().item()

    eps  = 1e-8
    dice = (2*total_TP + eps) / (2*total_TP + total_FP + total_FN + eps)
    iou  = (total_TP + eps)   / (total_TP + total_FP + total_FN + eps)
    prec = (total_TP + eps)   / (total_TP + total_FP + eps)
    rec  = (total_TP + eps)   / (total_TP + total_FN + eps)
    return {'dice': dice, 'iou': iou, 'precision': prec, 'recall': rec}


# ============================================================
# Phase2 主入口 (多 Seed 版本 + 自定义保存路径)
# ============================================================
if __name__ == '__main__':
    import argparse

    BASE = Path("/IMBR_Data/Student-home/2024M_WangHaiyu/onlytow")

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default=None,
                        help='只跑指定域，不填则跑全部')
    parser.add_argument('--ratio',  type=str, default=None,
                        help='只跑指定比例（需同时指定 --domain）')
    parser.add_argument('--ckpt',   type=str, default=None,
                        help='指定 Phase1 checkpoint 路径，不填用 meta_best.pth')
    parser.add_argument('--seeds',  type=str, default='42,3407,12345',
                        help='逗号分隔的种子列表，例如: 42,3407,12345')
    # 新增 save_dir 参数
    parser.add_argument('--save_dir', type=str, default=None,
                        help='自定义存放节点信息的目录地址（权重、日志的保存路径）')
    args = parser.parse_args()

    # 解析 seeds
    seed_list = [int(s.strip()) for s in args.seeds.split(',')]
    print(f"[Phase2] 将进行多 Seed 实验: {seed_list}")

    # checkpoint 路径
    if args.ckpt:
        META_CKPT = Path(args.ckpt)
    else:
        META_CKPT = BASE / 'MetaHFC3_checkpoints' / 'phase1_warmstart' / 'meta_best.pth'

    if not META_CKPT.exists():
        print(f"[错误] 元学习 checkpoint 不存在: {META_CKPT}")
        print("请先运行: python meta_train_v2.py")
        exit(1)

    print(f"[Phase2] 使用 checkpoint: {META_CKPT}")

    # 动态配置保存目录
    ADAPT_CFG_V2 = dict(ADAPT_CONFIG)
    if args.save_dir:
        ADAPT_CFG_V2['save_dir'] = str(Path(args.save_dir))
        print(f"[Phase2] 节点信息保存地址设为: {ADAPT_CFG_V2['save_dir']}")
    else:
        ADAPT_CFG_V2['save_dir'] = str(BASE / 'MetaHFC3_checkpoints' / 'phase2_warmstart')

    # 包含 HAM10000 的全部实验定义
    ALL_EXPERIMENTS = {
        'ISIC2018': ['5per', '10per', '25per', '50per', '75per', '100per'],
        'Drishti':  ['10per', '25per', '50per', '75per', '100per'],
        'ISIC2016': ['5per', '10per', '25per', '50per', '75per', '100per'],
        'ISIC2017': ['5per', '10per', '25per', '50per', '75per', '100per'],
        'REFUGE2':  ['5per', '10per', '25per', '50per', '75per', '100per'],
        'HAM10000': ['5per', '10per', '25per', '50per', '75per', '100per'], 
    }

    # 根据参数筛选要跑的域
    if args.domain:
        if args.domain not in ALL_EXPERIMENTS:
            print(f"[错误] 未知域: {args.domain}")
            print(f"  可选: {list(ALL_EXPERIMENTS.keys())}")
            exit(1)
        experiments = {args.domain: ALL_EXPERIMENTS[args.domain]}
        if args.ratio:
            experiments[args.domain] = [args.ratio]
    else:
        experiments = {k: v for k, v in ALL_EXPERIMENTS.items()
                       if k in TARGET_DOMAINS or k == 'HAM10000'}

    print(f"[Phase2] 待跑实验: {experiments}")

    all_results = {}
    for target_name, ratios in experiments.items():
        all_results[target_name] = {}
        
        # 兼容新的 HAM1000 数据集路径逻辑
        if target_name == 'HAM10000':
            custom_divide_root = Path("/IMBR_Data/Student-home/2024M_WangHaiyu/onlytow/ham1000divide")
            cfg_domain = TARGET_DOMAINS.get('HAM10000', {})
            test_img  = cfg_domain.get('test_img', custom_divide_root / 'test' / 'images')
            test_mask = cfg_domain.get('test_mask', custom_divide_root / 'test' / 'masks')
        else:
            cfg_domain = TARGET_DOMAINS[target_name]
            test_img   = cfg_domain['test_img']
            test_mask  = cfg_domain['test_mask']
            
        for ratio in ratios:
            # 根据域动态指向对应的数据划分路径
            if target_name == 'HAM10000':
                train_img  = custom_divide_root / ratio / 'images'
                train_mask = custom_divide_root / ratio / 'masks'
            else:
                train_img  = cfg_domain['divide_root'] / ratio / 'images'
                train_mask = cfg_domain['divide_root'] / ratio / 'masks'

            if not train_img.exists():
                print(f"  [跳过] {target_name}-{ratio}: 目录不存在 {train_img}")
                continue
                
            seed_dices = []
            
            # --- 核心：多 Seed 循环 ---
            for seed in seed_list:
                adapter = DomainAdapterV2(
                    meta_ckpt_path = META_CKPT,
                    target_name    = target_name,
                    train_img_dir  = train_img,
                    train_mask_dir = train_mask,
                    test_img_dir   = test_img,
                    test_mask_dir  = test_mask,
                    ratio_name     = ratio,
                    seed           = seed,  # 传入对应 seed
                    cfg            = ADAPT_CFG_V2,
                    model_cfg      = MODEL_CONFIG,
                )
                best_dice = adapter.run()
                seed_dices.append(best_dice)
                torch.cuda.empty_cache()
            
            # 计算该比例下的 Mean / Std
            mean_dice = np.mean(seed_dices)
            std_dice  = np.std(seed_dices)
            all_results[target_name][ratio] = {
                'mean': mean_dice, 
                'std': std_dice, 
                'raw': seed_dices
            }

    # ============================================================
    # 汇总打印 (Mean ± Std 格式)
    # ============================================================
    print("\n" + "=" * 80)
    print("实验结果汇总（Mean ± Std）")
    print("=" * 80)
    all_ratios = ['5per', '10per', '25per', '50per', '75per', '100per']
    
    header = f"{'比例':8s}"
    for t in all_results:
        header += f"  {t:16s}"
    print(header)
    print("-" * 80)
    
    seen_ratios = sorted(set(r for v in all_results.values() for r in v))
    for ratio in seen_ratios:
        row = f"{ratio:8s}"
        for t in all_results:
            if ratio in all_results[t]:
                res = all_results[t][ratio]
                # 输出 Mean ± Std
                row += f"  {res['mean']:.4f}±{res['std']:.4f}  "
            else:
                row += f"  {'—':16s}"
        print(row)