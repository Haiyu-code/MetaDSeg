import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os, random, numpy as np, csv
from glob import glob
from PIL import Image
import torchvision.transforms as T

from dataset import PathologyDataset, get_transforms
from loss import DiceLoss, WeightedBCELoss


# ==========================================
# 0. 随机种子
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    # ★ Fix：初始设为 False，AdaBN 完成后在 run() 中再开启
    torch.backends.cudnn.benchmark = False


# ==========================================
# 1. HFC 模块
# ==========================================
class HFC_Module(nn.Module):
    def __init__(self, channels, stat_dim=25):
        super().__init__()
        self.gamma_fc = nn.Sequential(
            nn.Linear(stat_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(stat_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Tanh()
        )
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // 4, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // 4, 1), channels),
            nn.Sigmoid()
        )
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)

    def forward(self, F_in, s):
        B, C, H, W = F_in.shape

        if s.dim() == 1:
            s = s.unsqueeze(0).expand(B, -1).contiguous()
        elif s.dim() == 2 and s.size(0) != B:
            s = s.expand(B, -1).contiguous()

        s = s.to(dtype=F_in.dtype)

        gamma   = self.gamma_fc(s).view(B, C, 1, 1)
        beta    = self.beta_fc(s).view(B, C, 1, 1)
        F_norm  = self.instance_norm(F_in)
        F_calib = gamma * F_norm + beta
        g       = self.gate_fc(F_in).view(B, C, 1, 1)
        return g * F_calib + (1.0 - g) * F_in


# ==========================================
# 2. UNet 基础组件
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ==========================================
# 3. DSCal_UNet
# ==========================================
class DSCal_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_ch=32, stat_dim=25):
        super().__init__()
        b = base_ch  # 32

        # ---- 编码器 ----
        self.inc   = DoubleConv(n_channels, b)      # 3   → 32
        self.down1 = Down(b,     b * 2)             # 32  → 64
        self.down2 = Down(b*2,   b * 4)             # 64  → 128
        self.down3 = Down(b*4,   b * 8)             # 128 → 256
        self.down4 = Down(b*8,   b * 16)            # 256 → 512

        # ---- 解码器 ----
        self.up1 = UpConv(in_ch=b*16, skip_ch=b*8,  out_ch=b*8)
        self.up2 = UpConv(in_ch=b*8,  skip_ch=b*4,  out_ch=b*4)
        self.up3 = UpConv(in_ch=b*4,  skip_ch=b*2,  out_ch=b*2)
        self.up4 = UpConv(in_ch=b*2,  skip_ch=b,    out_ch=b)

        # ---- HFC ----
        self.hfc1 = HFC_Module(b*8,  stat_dim)   # 256
        self.hfc2 = HFC_Module(b*4,  stat_dim)   # 128
        self.hfc3 = HFC_Module(b*2,  stat_dim)   # 64
        self.hfc4 = HFC_Module(b,    stat_dim)   # 32

        # ---- 输出头 ----
        self.outc = nn.Conv2d(b, n_classes, 1)

    def forward(self, x, s=None):
        # 编码
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        # 解码 + HFC
        d1 = self.up1(e5, e4)
        d1 = self.hfc1(d1, s) if s is not None else d1

        d2 = self.up2(d1, e3)
        d2 = self.hfc2(d2, s) if s is not None else d2

        d3 = self.up3(d2, e2)
        d3 = self.hfc3(d3, s) if s is not None else d3

        d4 = self.up4(d3, e1)
        d4 = self.hfc4(d4, s) if s is not None else d4

        return self.outc(d4)

    def load_unet_weights(self, std_state_dict):
        my_dict  = self.state_dict()
        new_dict = {}
        matched, skipped = 0, 0

        for k, v in std_state_dict.items():
            new_k = k
            new_k = new_k.replace('outc.conv.weight', 'outc.weight')
            new_k = new_k.replace('outc.conv.bias',   'outc.bias')

            if new_k in my_dict and my_dict[new_k].shape == v.shape:
                new_dict[new_k] = v
                matched += 1
            else:
                skipped += 1
                print(f"  [跳过] {k} → {new_k}  "
                      f"src={v.shape}, "
                      f"dst={my_dict.get(new_k, torch.tensor([])).shape}")

        my_dict.update(new_dict)
        self.load_state_dict(my_dict)
        print(f"===> 权重加载完成: 匹配 {matched} 层，跳过 {skipped} 层")


# ==========================================
# 4. 25维目标域统计向量
# ==========================================
def build_stats_vector(target_stats, device):
    lab_mean = target_stats['lab_mean'].to(device).float()
    lab_std  = target_stats['lab_std'].to(device).float()
    freq_amp = target_stats['freq_amp'].to(device).float()

    lab_mean_norm = lab_mean / torch.tensor([100., 128., 128.], device=device)
    lab_std_norm  = lab_std  / torch.tensor([50.,  64.,  64.],  device=device)

    H, W = freq_amp.shape
    cy, cx = H // 2, W // 2
    y_idx, x_idx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    dist = torch.sqrt((y_idx - cy)**2 + (x_idx - cx)**2)

    n_bands    = 16
    r_max      = min(H, W) // 2
    band_edges = torch.linspace(0, r_max, n_bands + 1, device=device)

    band_energies = []
    for i in range(n_bands):
        mask   = (dist >= band_edges[i]) & (dist < band_edges[i+1])
        energy = freq_amp[mask].mean() if mask.sum() > 0 \
                 else torch.tensor(0., device=device)
        band_energies.append(energy)

    freq_bands = torch.stack(band_energies)
    freq_bands = freq_bands / (freq_bands.max() + 1e-8)

    freq_flat = freq_amp.flatten()
    freq_mean = (freq_flat.mean() / (freq_flat.max() + 1e-8)).unsqueeze(0)
    freq_std  = (freq_flat.std()  / (freq_flat.max() + 1e-8)).unsqueeze(0)
    r_low     = int(min(H, W) * 0.1)
    low_e     = freq_amp[dist < r_low].mean()
    high_e    = freq_amp[dist >= r_low].mean()
    lh_ratio  = torch.clamp(low_e / (high_e + 1e-8) / 10., 0., 1.).unsqueeze(0)

    stat_vec = torch.cat([
        lab_mean_norm,  # [3]
        lab_std_norm,   # [3]
        freq_bands,     # [16]
        freq_mean,      # [1]
        freq_std,       # [1]
        lh_ratio        # [1]
    ])  # 共 25 维

    print(f"===> 统计向量: shape={stat_vec.shape}, dtype={stat_vec.dtype}, "
          f"min={stat_vec.min():.3f}, max={stat_vec.max():.3f}")

    return stat_vec.detach().float()


# ==========================================
# 5. 无标注数据集（弱增强 + 强增强）
# ==========================================
class UnlabeledDataset(Dataset):
    def __init__(self, image_dir, img_size=352):
        self.img_paths = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
            self.img_paths.extend(glob(os.path.join(image_dir, ext)))
        self.img_paths = sorted(self.img_paths)

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        self.weak = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        self.strong = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        print(f"===> 无标注数据集: {len(self.img_paths)} 张")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.weak(img), self.strong(img)


# ==========================================
# 6. 主训练器
# ==========================================
class DSCalTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(config['save_dir'], exist_ok=True)

        self.log_path = os.path.join(config['save_dir'], 'train_log.csv')
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'total_loss', 'sup_loss', 'cons_loss', 'val_dice', 'lr']
            )

        # Step1：初始化模型
        self.model = DSCal_UNet(
            n_channels=3, n_classes=1,
            base_ch=config['base_channels'], stat_dim=25
        ).to(self.device)

        # Step2：加载 Kvasir 权重
        ckpt = torch.load(config['pretrained_weights'], map_location=self.device)
        if 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        self.model.load_unet_weights(ckpt)

        # Step3：构建 25 维统计向量
        target_stats  = torch.load(config['target_stats_path'], map_location=self.device)
        self.stat_vec = build_stats_vector(target_stats, self.device)

        # Step4：全量 AdaBN（benchmark=False 状态下执行，安全稳定）
        self._apply_full_adabn(config)

        # Step5：损失函数
        self.dice_criterion = DiceLoss()
        self.bce_criterion  = WeightedBCELoss()
        self.cons_criterion = nn.MSELoss()

        # Step6：分层学习率
        encoder_tops = {'inc', 'down1', 'down2', 'down3', 'down4'}
        enc_params, dec_params = [], []
        for name, param in self.model.named_parameters():
            if name.split('.')[0] in encoder_tops:
                enc_params.append(param)
            else:
                dec_params.append(param)

        print(f"===> 分层LR: 编码器 {len(enc_params)} 参数组，"
              f"解码器+HFC {len(dec_params)} 参数组")

        self.optimizer = optim.AdamW([
            {'params': enc_params, 'lr': config['lr'] * 0.1},  # 1e-5
            {'params': dec_params, 'lr': config['lr']},         # 1e-4
        ], weight_decay=config['weight_decay'])

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs'], eta_min=1e-6
        )

        # Step7：有标注数据
        train_ds = PathologyDataset(
            config['train_image_dir'], config['train_mask_dir'],
            transform=get_transforms(config['img_size'], is_train=True),
            img_size=config['img_size']
        )
        val_ds = PathologyDataset(
            config['val_image_dir'], config['val_mask_dir'],
            transform=get_transforms(config['img_size'], is_train=False),
            img_size=config['img_size']
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=config['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True,
            persistent_workers=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=config['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True,
            persistent_workers=True
        )

        # Step8：无标注数据（一致性损失）
        unlabeled_ds = UnlabeledDataset(
            config['full_image_dir'], img_size=config['img_size']
        )
        self.unlabeled_loader = DataLoader(
            unlabeled_ds, batch_size=config['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True,
            drop_last=True, persistent_workers=True
        )
        self._unlabeled_iter = iter(self.unlabeled_loader)

        print(f"===> 有标注: {len(train_ds)} 张 | "
              f"无标注: {len(unlabeled_ds)} 张 | "
              f"验证: {len(val_ds)} 张")

    # --------------------------------------------------
    def _apply_full_adabn(self, config):
        print("===> [AdaBN-Full] 用全量 ISIC 初始化 BN 统计量...")
        ds = UnlabeledDataset(config['full_image_dir'], img_size=config['img_size'])
        loader = DataLoader(
            ds, batch_size=config['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True,
            persistent_workers=True
        )

        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = False

        # ★ Fix1：AdaBN 阶段临时关闭 benchmark，避免 cuDNN 算法搜索失败
        old_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False

        try:
            with torch.no_grad():
                for imgs_w, _ in loader:
                    # ★ Fix2：双重保险，完全绕过 cuDNN 优化路径
                    with torch.backends.cudnn.flags(enabled=True, benchmark=False,
                                                    deterministic=True):
                        self.model(
                            imgs_w.to(self.device).float().contiguous(),
                            s=None
                        )
        except RuntimeError as e:
            # ★ Fix3：若仍失败，回退到完全禁用 cuDNN 模式
            print(f"  [警告] cuDNN 模式失败: {e}")
            print("  [回退] 使用 cuDNN disabled 模式重试...")
            with torch.no_grad():
                with torch.backends.cudnn.flags(enabled=False):
                    for imgs_w, _ in loader:
                        self.model(
                            imgs_w.to(self.device).float().contiguous(),
                            s=None
                        )
        finally:
            # ★ 无论成功失败，都恢复原始设置
            torch.backends.cudnn.benchmark = old_benchmark

        for p in self.model.parameters():
            p.requires_grad = True

        print(f"===> [AdaBN-Full] 完成！基于 {len(ds)} 张图像")

    # --------------------------------------------------
    def _freeze_bn_stats(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if m.affine:
                    m.weight.requires_grad = True
                    m.bias.requires_grad   = True

    # --------------------------------------------------
    def _get_unlabeled_batch(self):
        try:
            return next(self._unlabeled_iter)
        except StopIteration:
            self._unlabeled_iter = iter(self.unlabeled_loader)
            return next(self._unlabeled_iter)

    # --------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()
        self._freeze_bn_stats()

        total_loss = sup_total = cons_total = 0.
        lambda_cons = self.config.get('lambda_cons', 0.1)

        for batch in self.train_loader:
            images = batch['image'].to(self.device).float().contiguous()
            masks  = batch['mask'].to(self.device).float().contiguous()
            rois   = batch['roi'].to(self.device).float().contiguous()

            # 有监督
            out      = self.model(images, s=self.stat_vec)
            loss_sup = (self.dice_criterion(out, masks, rois) +
                        self.bce_criterion(out, masks, rois))

            # 一致性（无标注）
            imgs_w, imgs_s = self._get_unlabeled_batch()
            imgs_w = imgs_w.to(self.device).float().contiguous()
            imgs_s = imgs_s.to(self.device).float().contiguous()

            with torch.no_grad():
                pseudo = torch.sigmoid(self.model(imgs_w, s=self.stat_vec))
            pred_s    = torch.sigmoid(self.model(imgs_s, s=self.stat_vec))
            loss_cons = self.cons_criterion(pred_s, pseudo)

            loss = loss_sup + lambda_cons * loss_cons

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            sup_total  += loss_sup.item()
            cons_total += loss_cons.item()

        n = len(self.train_loader)
        return total_loss / n, sup_total / n, cons_total / n

    # --------------------------------------------------
    def validate(self):
        self.model.eval()
        total_dice = 0.
        with torch.no_grad():
            for batch in self.val_loader:
                images  = batch['image'].to(self.device).float().contiguous()
                masks   = batch['mask'].to(self.device).float().contiguous()
                outputs = torch.sigmoid(self.model(images, s=self.stat_vec))
                pred    = (outputs > 0.5).float()
                inter   = (pred * masks).sum()
                union   = pred.sum() + masks.sum()
                total_dice += (2. * inter + 1e-6) / (union + 1e-6)

        if len(self.val_loader) == 0:
            return 0.
        return (total_dice / len(self.val_loader)).item()

    # --------------------------------------------------
    def run(self):
        # ★ Fix：AdaBN 初始化完成后，正式训练阶段开启 benchmark 加速
        torch.backends.cudnn.benchmark = True
        print("===> cuDNN benchmark 已启用，开始正式训练...")

        best_dice, patience_ctr = 0., 0
        for epoch in range(1, self.config['epochs'] + 1):
            t_loss, s_loss, c_loss = self.train_epoch(epoch)
            val_dice = self.validate()
            self.scheduler.step()

            lr = self.optimizer.param_groups[1]['lr']
            with open(self.log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch,
                    f"{t_loss:.6f}", f"{s_loss:.6f}",
                    f"{c_loss:.6f}", f"{val_dice:.4f}", f"{lr:.8f}"
                ])

            if epoch % 5 == 0:
                print(f"[{self.config['run_name']}] "
                      f"Ep {epoch}: loss={t_loss:.4f} "
                      f"(sup={s_loss:.4f} cons={c_loss:.4f}) "
                      f"Dice={val_dice:.4f}")

            if val_dice > best_dice:
                best_dice    = val_dice
                patience_ctr = 0
                torch.save(self.model.state_dict(),
                           f"{self.config['save_dir']}/best_model.pth")
                print(f"*** Ep {epoch}: 新最佳 Dice = {best_dice:.4f}")
            else:
                patience_ctr += 1

            if patience_ctr >= self.config['patience']:
                print(f"!!! 早停。最高 Dice: {best_dice:.4f}")
                break


# ==========================================
# 7. 主入口
# ==========================================
if __name__ == '__main__':
    seed_everything(42)

    TRAIN_BASE   = Path("/IMBR_Data/Student-home/2024M_WangHaiyu/onlytow/isic2017divide")
    VAL_ROOT     = Path("/IMBR_Data/Student-home/2024M_WangHaiyu/onlytow/isic2017/test")
    PRETRAINED   = "/IMBR_Data/Student-home/2024M_WangHaiyu/onlytow/name1/unet/ckkvasir/baseline_best.pth"
    SAVE_BASE    = Path("/IMBR_Data/Student-home/2024M_WangHaiyu/onlytow/unet/dscal_hfc_isic2017")
    FULL_IMG_DIR = "/IMBR_Data/Student-home/2024M_WangHaiyu/onlytow/isic2017/train/images"
    STATS_PATH   = "/IMBR_Data/Student-home/2024M_WangHaiyu/onlytow/name1/unet/totallog/ISIC2017_stats.pth"

    ratios = ['10per', '25per', '5per', '50per', '75per', '100per']

    for ratio in ratios:
        run_name  = f"DSCal_HFC_{ratio}"
        ratio_dir = TRAIN_BASE / ratio
        if not ratio_dir.exists():
            print(f"跳过 {ratio}，目录不存在")
            continue

        config = {
            'train_image_dir':    ratio_dir / 'images',
            'train_mask_dir':     ratio_dir / 'masks',
            'val_image_dir':      VAL_ROOT / 'images',
            'val_mask_dir':       VAL_ROOT / 'masks',
            'full_image_dir':     FULL_IMG_DIR,
            'save_dir':           str(SAVE_BASE / ratio),
            'pretrained_weights': PRETRAINED,
            'target_stats_path':  STATS_PATH,
            'base_channels': 32,
            'img_size':      352,
            'batch_size':    12,
            'epochs':        300,
            'patience':      30,
            'lr':            1e-4,
            'weight_decay':  1e-4,
            'lambda_cons':   0.1,
            'run_name':      run_name
        }

        print(f"\n{'='*60}")
        print(f">>>> 开始: {run_name}")
        print(f"{'='*60}")
        trainer = DSCalTrainer(config)
        trainer.run()
        torch.cuda.empty_cache()
