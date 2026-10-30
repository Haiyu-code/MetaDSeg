# model_v2.py
# WarmStartUNet：MetaHFC3 的主干网络
#
# 架构与 DSCal-Net（Adaptive_DSCal_UNet）完全对齐：
#   标准 UNet，base_ch=32，5层编码器，~31M 参数
#   解码器每层后接 HFC_Module 做域风格校准
#   额外增加 domain_head（Phase1 辅助损失，Phase2 不参与前向）
#
# 被 meta_train_v2.py 和 adapt_v2.py 共同 import

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import torchvision.transforms.functional as TF


class HFC_Module(nn.Module):
    def __init__(self, channels, stat_dim=25):
        super().__init__()
        self.gamma_fc = nn.Sequential(
            nn.Linear(stat_dim, channels), nn.ReLU(inplace=True),
            nn.Linear(channels, channels), nn.Sigmoid()
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(stat_dim, channels), nn.ReLU(inplace=True),
            nn.Linear(channels, channels), nn.Tanh()
        )
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, max(channels // 4, 1)), nn.ReLU(inplace=True),
            nn.Linear(max(channels // 4, 1), channels), nn.Sigmoid()
        )
        # eps=1e-5 防止 F_in 全零时 InstanceNorm 输出 nan
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False, eps=1e-5)

    def forward(self, F_in, s):
        B, C, H, W = F_in.shape
        if s.dim() == 1:
            s = s.unsqueeze(0).expand(B, -1)
        gamma  = self.gamma_fc(s).view(B, C, 1, 1)
        beta   = self.beta_fc(s).view(B, C, 1, 1)
        F_norm = self.instance_norm(F_in)
        # nan 保护：InstanceNorm 方差极小时输出 nan
        F_norm = torch.nan_to_num(F_norm, nan=0.0, posinf=1.0, neginf=-1.0)
        g      = self.gate_fc(F_in).view(B, C, 1, 1)
        out    = g * (gamma * F_norm + beta) + (1. - g) * F_in
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_ch, out_ch)
        )
    def forward(self, x): return self.maxpool_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x2, x1], dim=1))


class WarmStartUNet(nn.Module):
    """
    标准 UNet + HFC 域校准 + 域分类辅助头

    编码器/解码器/HFC 与 DSCal-Net（Adaptive_DSCal_UNet）完全对齐
    域分类头仅在 Phase1 内循环中使用（return_domain=True）
    Phase2 forward 时 return_domain=False，域分类头不参与任何计算
    """
    def __init__(self, n_channels=3, n_classes=1,
                 base_ch=32, stat_dim=25, n_domains=5):
        super().__init__()
        b = base_ch

        # 编码器（与 DSCal-Net key 命名完全一致）
        self.inc   = DoubleConv(n_channels, b)       # 3   → 32
        self.down1 = Down(b,    b*2)                 # 32  → 64
        self.down2 = Down(b*2,  b*4)                 # 64  → 128
        self.down3 = Down(b*4,  b*8)                 # 128 → 256
        self.down4 = Down(b*8,  b*16)                # 256 → 512

        # 解码器 + HFC
        self.up1  = UpConv(b*16, b*8,  b*8)
        self.up2  = UpConv(b*8,  b*4,  b*4)
        self.up3  = UpConv(b*4,  b*2,  b*2)
        self.up4  = UpConv(b*2,  b,    b)
        self.hfc1 = HFC_Module(b*8,  stat_dim)
        self.hfc2 = HFC_Module(b*4,  stat_dim)
        self.hfc3 = HFC_Module(b*2,  stat_dim)
        self.hfc4 = HFC_Module(b,    stat_dim)
        self.outc = nn.Conv2d(b, n_classes, 1)

        # 域分类辅助头（Phase1 专用）
        self.domain_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(b*16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_domains),
        )

    def forward(self, x, s=None, hfc_weight=1.0, return_domain=False):
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        d1 = self.up1(e5, e4)
        if s is not None and hfc_weight > 0.01:
            d1 = hfc_weight * self.hfc1(d1, s) + (1 - hfc_weight) * d1

        d2 = self.up2(d1, e3)
        if s is not None and hfc_weight > 0.01:
            d2 = hfc_weight * self.hfc2(d2, s) + (1 - hfc_weight) * d2

        d3 = self.up3(d2, e2)
        if s is not None and hfc_weight > 0.01:
            d3 = hfc_weight * self.hfc3(d3, s) + (1 - hfc_weight) * d3

        d4 = self.up4(d3, e1)
        if s is not None and hfc_weight > 0.01:
            d4 = hfc_weight * self.hfc4(d4, s) + (1 - hfc_weight) * d4

        seg_out = self.outc(d4)

        if return_domain:
            return seg_out, self.domain_head(e5)
        return seg_out

    def load_kvasir_weights(self, ckpt_path, device='cpu'):
        """
        从 baseline_best.pth（Standard UNet state_dict）加载权重

        baseline_best.pth 的 key 命名：
          inc/down1~4/up1~4 与本模型完全一致
          outc.conv.weight / outc.conv.bias  →  outc.weight / outc.bias（唯一差异）

        HFC 和 domain_head 不在 ckpt 中，保持随机初始化（正常）
        """
        ckpt_path = Path(ckpt_path)
        print(f"\n[Warm-Start] 加载 Kvasir 预训练权重: {ckpt_path}")
        std_sd = torch.load(ckpt_path, map_location=device, weights_only=False)

        my_dict  = self.state_dict()
        new_dict = {}
        matched = skipped = 0

        for k, v in std_sd.items():
            new_k = k.replace('outc.conv.weight', 'outc.weight') \
                      .replace('outc.conv.bias',   'outc.bias')
            if new_k in my_dict and my_dict[new_k].shape == v.shape:
                new_dict[new_k] = v
                matched += 1
            else:
                skipped += 1
                print(f"  [跳过] {k} → key不存在或shape不匹配")

        my_dict.update(new_dict)
        self.load_state_dict(my_dict)

        total = matched + skipped
        print(f"  权重加载: 匹配 {matched}/{total} 层，跳过 {skipped} 层")
        if skipped <= 2:
            print(f"  ✅ Warm-Start 完成（HFC/domain_head 随机初始化，正常）")
        else:
            print(f"  ⚠️  跳过层数={skipped}，请检查权重文件与模型是否匹配")

    def get_encoder_params(self):
        enc = {'inc', 'down1', 'down2', 'down3', 'down4'}
        return [p for n, p in self.named_parameters()
                if n.split('.')[0] in enc]

    def get_decoder_params(self):
        """解码器 + HFC + domain_head 参数（Phase2 主学习率）"""
        enc = {'inc', 'down1', 'down2', 'down3', 'down4'}
        return [p for n, p in self.named_parameters()
                if n.split('.')[0] not in enc]

    def freeze_bn_stats(self):
        """冻结 BN running stats，γ/β 仍可学习"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if m.affine:
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)

    def apply_adabn(self, img_01_list, device, stat_vec=None, batch_size=8):
        """
        AdaBN：用目标域 [0,1] 图更新 BN running stats
        stat_vec 不为 None 时 HFC 正常参与前向，路径与推理完全一致
        """
        self.train()
        with torch.no_grad():
            for i in range(0, len(img_01_list), batch_size):
                batch_raw = img_01_list[i:i + batch_size]
                tensors = []
                for img_01 in batch_raw:
                    t = TF.normalize(
                        img_01.clone(),
                        mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225]
                    )
                    tensors.append(t)
                batch  = torch.stack(tensors).to(device)
                hw     = 1.0 if stat_vec is not None else 0.0
                self(batch, s=stat_vec, hfc_weight=hw)
        status = '路径完全一致' if stat_vec is not None else 'stat_vec=None，HFC跳过'
        print(f"  [AdaBN] 完成，{len(img_01_list)} 张图，{status}")