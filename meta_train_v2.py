
import copy
import random
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

from config    import (META_TRAIN_DOMAINS, META_CONFIG, MODEL_CONFIG,
                        validate_paths)
from dataset   import augment_pair, find_mask
from loss      import BCEDiceLoss
from hfc_utils import (compute_hfc_single,
                        normalize_hfc,
                        precompute_global_hfc_stats)

from model_v2 import WarmStartUNet

# ============================================================
# 配置
# ============================================================
BASE = Path("/IMBR_Data/Student-home/2024M_WangHaiyu/onlytow")

KVASIR_CKPT = BASE / "baseline_best.pth"

# ── 续跑配置（从 ep500 checkpoint 出发）──────────────────────
RESUME_CKPT     = BASE / 'MetaHFC3_checkpoints' / 'phase1_warmstart' / 'meta_ep500.pth'
RESUME_START_EP = 501
# ─────────────────────────────────────────────────────────────

META_CONFIG_V2 = {
    'n_episodes':    5000,
    'K_support':     10,
    'Q_query':       16,
    'n_inner_steps': 5,
    'lr_inner':      1e-3,    # 基础内循环 lr（跨模态域使用）
    'lr_inner_enc':  2e-4,    # 内镜预训练域内循环 lr（Kvasir/ClinicDB）
    'lr_outer':      2e-4,    # ↑ 1e-4 → 2e-4，加速外循环收敛
    'skip_thresh':   -0.05,   # ↑ 0.02 → -0.05，只跳过真负增益
    'lambda_domain': 0.1,
    'img_size':      352,
    'save_dir':      str(BASE / 'MetaHFC3_checkpoints' / 'phase1_warmstart'),
    'save_interval': 500,
    'log_interval':  100,
    'seed':          42,
}

MODEL_CONFIG_V2 = {
    'n_channels': 3,
    'n_classes':  1,
    'base_ch':    32,
    'stat_dim':   25,
    'n_domains':  5,
}

# 内镜预训练相关域（使用降低的内循环 lr）
ENDOSCOPY_DOMAINS = {'Kvasir', 'ClinicDB'}


# ============================================================
# 工具函数
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ============================================================
# Episode 采样器（与 v2.0 完全相同）
# ============================================================
class MultiDomainEpisodeSampler:
    DOMAIN_NAMES = ['Kvasir', 'ClinicDB', 'ISIC2016', 'ISIC2017', 'REFUGE']
    IMG_EXTS     = ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'PNG']
    MASK_EXTS    = ['png', 'jpg', 'bmp', 'PNG']
    MODALITY_MAP = {
        'Kvasir':   'endoscopy',
        'ClinicDB': 'endoscopy',
        'ISIC2016': 'dermoscopy',
        'ISIC2017': 'dermoscopy',
        'REFUGE':   'fundus',
    }

    def __init__(self, domain_configs, global_mean, global_std,
                 K=10, Q=16, img_size=352, seed=42, device='cpu'):
        self.K           = K
        self.Q           = Q
        self.img_size    = img_size
        self.device      = device
        self.global_mean = global_mean
        self.global_std  = global_std

        self.domain_pairs = {}
        self.domain_to_id = {}
        valid_id = 0

        for name in self.DOMAIN_NAMES:
            cfg   = domain_configs[name]
            pairs = self._collect_pairs(cfg['train_img'], cfg['train_mask'])
            if len(pairs) == 0:
                print(f"  ⚠️  [{name}] 数据量=0，已排除")
            else:
                self.domain_pairs[name] = pairs
                self.domain_to_id[name] = valid_id
                valid_id += 1
                print(f"  [{name}] {len(pairs)} 张")

        self.active_domains = list(self.domain_pairs.keys())
        if not self.active_domains:
            raise RuntimeError("所有域均为空！")

        # 模态分组
        self.modality_groups = {}
        for d in self.active_domains:
            mod = self.MODALITY_MAP.get(d, d)
            self.modality_groups.setdefault(mod, []).append(d)
        self.active_modalities = list(self.modality_groups.keys())

        print(f"\n  [模态均衡采样]")
        for mod, domains in self.modality_groups.items():
            each = 100.0 / len(self.active_modalities) / len(domains)
            print(f"    {mod}: {domains} → 每域 {each:.1f}%")

        # 预缓存 HFC
        print("\n  [HFC预缓存] 开始...")
        self.hfc_cache = {}
        total = sum(len(p) for p in self.domain_pairs.values())
        done  = 0
        for name in self.active_domains:
            for img_path, _ in self.domain_pairs[name]:
                key = str(img_path)
                if key not in self.hfc_cache:
                    try:
                        img    = Image.open(img_path).convert('RGB')
                        img    = img.resize((img_size, img_size), Image.BILINEAR)
                        img_01 = TF.to_tensor(img)
                        self.hfc_cache[key] = compute_hfc_single(img_01, 'cpu')
                    except Exception as e:
                        print(f"    [跳过HFC] {Path(img_path).name}: {e}")
                done += 1
                if done % 500 == 0:
                    print(f"    进度: {done}/{total}")
        print(f"  [HFC预缓存] 完成，共 {len(self.hfc_cache)} 张\n")

        random.seed(seed)
        np.random.seed(seed)

    def _collect_pairs(self, img_dir, mask_dir):
        img_dir  = Path(img_dir)
        mask_dir = Path(mask_dir)
        if not img_dir.exists() or not mask_dir.exists():
            return []
        imgs = []
        for ext in self.IMG_EXTS:
            imgs.extend(img_dir.glob(f'*.{ext}'))
        imgs = sorted(set(imgs))
        pairs = []
        for img_path in imgs:
            found = find_mask(img_path.stem, mask_dir, self.MASK_EXTS)
            if found:
                pairs.append((img_path, found))
        return pairs

    def _get_hfc_for_pairs(self, pairs):
        vecs = [self.hfc_cache[str(p)] for p, _ in pairs
                if str(p) in self.hfc_cache]
        if not vecs:
            return torch.zeros(25)
        raw_mean = torch.stack(vecs).mean(0)
        return normalize_hfc(raw_mean,
                             self.global_mean.cpu(),
                             self.global_std.cpu(), 'cpu')

    def sample_episode(self):
        modality    = random.choice(self.active_modalities)
        domain_name = random.choice(self.modality_groups[modality])
        domain_id   = self.domain_to_id[domain_name]
        pairs       = self.domain_pairs[domain_name]

        total_needed = self.K + self.Q
        selected = (random.choices(pairs, k=total_needed)
                    if len(pairs) < total_needed
                    else random.sample(pairs, total_needed))

        support_pairs = selected[:self.K]
        query_pairs   = selected[self.K:]

        stat_vec       = self._get_hfc_for_pairs(support_pairs)
        s_norm, s_mask = self._load_batch(support_pairs, is_train=True)
        q_norm, q_mask = self._load_batch(query_pairs,   is_train=False)

        return s_norm, s_mask, q_norm, q_mask, stat_vec, domain_id, domain_name

    def _load_batch(self, pairs, is_train=True):
        imgs_norm, masks = [], []
        for img_path, mask_path in pairs:
            img  = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            img  = img.resize( (self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            if is_train:
                img, mask = augment_pair(img, mask)
            img_01   = TF.to_tensor(img)
            img_norm = TF.normalize(img_01,
                                    mean=[0.485, 0.456, 0.406],
                                    std= [0.229, 0.224, 0.225])
            mask_t = torch.from_numpy(
                np.array(mask, dtype=np.float32)
            ).unsqueeze(0) / 255.0
            mask_t = (mask_t > 0.5).float()
            imgs_norm.append(img_norm)
            masks.append(mask_t)
        return torch.stack(imgs_norm), torch.stack(masks)


# ============================================================
# MetaTrainer V2.1
# ============================================================
class MetaTrainerV2:
    def __init__(self, cfg=META_CONFIG_V2, model_cfg=MODEL_CONFIG_V2):
        self.cfg    = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[MetaHFC3 Warm-Start v2.1] 设备: {self.device}")
        print(f"  lr_inner(enc域)={cfg['lr_inner_enc']} | "
              f"lr_inner(其他)={cfg['lr_inner']} | "
              f"lr_outer={cfg['lr_outer']} | "
              f"skip_thresh={cfg['skip_thresh']}")

        seed_everything(cfg['seed'])

        self.save_dir = Path(cfg['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.save_dir / 'meta_train_log.csv'

        # 全局 HFC 统计（复用已有缓存）
        shared_stats = (BASE / 'MetaHFC_checkpoints' / 'phase1' /
                        'global_hfc_stats.pth')
        stats_path = shared_stats if shared_stats.exists() \
                     else self.save_dir / 'global_hfc_stats.pth'
        print(f"[HFC统计] 使用: {stats_path}")

        self.global_mean, self.global_std, self.domain_means = \
            precompute_global_hfc_stats(
                META_TRAIN_DOMAINS, self.device,
                save_path=stats_path,
                img_size=cfg['img_size'],
                max_per_domain=200,
                sample_seed=42,
            )

        # ---- 模型：WarmStartUNet ----
        self.model = WarmStartUNet(
            n_channels=model_cfg['n_channels'],
            n_classes= model_cfg['n_classes'],
            base_ch=   model_cfg['base_ch'],
            stat_dim=  model_cfg['stat_dim'],
            n_domains= model_cfg['n_domains'],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[模型] 参数量: {n_params:.1f}M")

        # ---- 续跑或 Warm-Start ----
        self._neg_gain_count = 0
        self._skip_count     = 0
        self._resume_ep      = 0

        if RESUME_CKPT is not None:
            self._saved_modality_counts = self._load_resume(RESUME_CKPT)
        else:
            self.model.load_kvasir_weights(KVASIR_CKPT, device=self.device)
            self._saved_modality_counts = None

        self.seg_loss    = BCEDiceLoss()
        self.domain_loss = nn.CrossEntropyLoss()

        self.sampler = MultiDomainEpisodeSampler(
            domain_configs=META_TRAIN_DOMAINS,
            global_mean=self.global_mean,
            global_std= self.global_std,
            K=cfg['K_support'],
            Q=cfg['Q_query'],
            img_size=cfg['img_size'],
            seed=cfg['seed'],
            device=str(self.device),
        )

        # 续跑时追加日志，否则新建
        log_mode = 'a' if (RESUME_CKPT is not None and self.log_path.exists()) \
                   else 'w'
        with open(self.log_path, log_mode, newline='') as f:
            if log_mode == 'w':
                csv.writer(f).writerow([
                    'episode', 'domain', 'modality',
                    'query_loss_before', 'query_loss_after',
                    'relative_gain', 'weight', 'skipped',
                    'avg100_before', 'avg100_after',
                    'neg_gain_total', 'skip_total', 'time_s',
                    'inner_lr',       # ← 新增：记录实际使用的内循环 lr
                ])

        print(f"[MetaHFC3 v2.1] 初始化完成，训练 {cfg['n_episodes']} episodes\n")

    def _load_resume(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        print(f"\n[续跑] 加载: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self._neg_gain_count = ckpt.get('neg_gain_count', 0)
        self._skip_count     = ckpt.get('skip_count', 0)
        self._resume_ep      = ckpt.get('episode', 0)
        saved_mc             = ckpt.get('modality_counts', None)
        saved_avg            = ckpt.get('avg_loss', float('inf'))
        print(f"  ✅ 恢复自 Ep {self._resume_ep}, avg_loss={saved_avg:.4f}")
        print(f"  ⚠️  注意：skip_count={self._skip_count} 含旧标准计数，"
              f"续跑后使用新阈值({self.cfg['skip_thresh']})，skip率会明显下降")
        return saved_mc

    def _inner_loop(self, support_norm, support_masks,
                    domain_id, stat_vec, domain_name):
        """
        内循环：在 support set 上做 n_inner_steps 步 SGD
        改动：根据域类型使用差异化内循环 lr
          - Kvasir/ClinicDB（已被预训练）：lr_inner_enc=2e-4，保护已学特征
          - ISIC/REFUGE（跨模态域）：lr_inner=1e-3，允许较大更新
        """
        fast_model = copy.deepcopy(self.model)
        fast_model.train()

        # ── 差异化内循环 lr ──────────────────────────────────────
        if domain_name in ENDOSCOPY_DOMAINS:
            inner_lr = self.cfg['lr_inner_enc']   # 2e-4
        else:
            inner_lr = self.cfg['lr_inner']        # 1e-3
        # ────────────────────────────────────────────────────────

        optimizer = optim.SGD(fast_model.parameters(),
                              lr=inner_lr, momentum=0.9)

        support_norm  = support_norm.to(self.device)
        support_masks = support_masks.to(self.device)
        stat_vec_dev  = stat_vec.to(self.device)
        domain_label  = torch.tensor(
            [domain_id] * len(support_norm),
            dtype=torch.long, device=self.device
        )

        for _ in range(self.cfg['n_inner_steps']):
            optimizer.zero_grad()
            seg_out, domain_logits = fast_model(
                support_norm, s=stat_vec_dev,
                hfc_weight=1.0, return_domain=True
            )
            l_seg    = self.seg_loss(seg_out, support_masks)
            l_domain = self.domain_loss(domain_logits, domain_label)
            loss     = l_seg + self.cfg.get('lambda_domain', 0.1) * l_domain
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                fast_model.parameters(), max_norm=5.0
            )
            optimizer.step()

        return fast_model, inner_lr

    def _difficulty_aware_update(self, fast_model,
                                  query_norm, query_masks,
                                  stat_vec, loss_before):
        """
        外循环：难度感知 Reptile 更新
        改动：skip 阈值从 0.02 → -0.05
          旧：gain < 0.02 就跳过（包含 gain=0 的内镜域 episode）
          新：gain < -0.05 才跳过（只跳过损失恶化超过 5% 的 episode）
          gain 在 [-0.05, 0) 时：正常更新，weight=1.0（退化为标准 Reptile）
          gain 在[0, 1] 时：正常难度加权更新
        """
        skip_thresh = self.cfg.get('skip_thresh', -0.05)

        fast_model.eval()
        query_norm   = query_norm.to(self.device)
        query_masks  = query_masks.to(self.device)
        stat_vec_dev = stat_vec.to(self.device)

        with torch.no_grad():
            seg_after, _ = fast_model(
                query_norm, s=stat_vec_dev,
                hfc_weight=1.0, return_domain=True
            )
            loss_after = self.seg_loss(seg_after, query_masks).item()

        relative_gain = (loss_before - loss_after) / (loss_before + 1e-8)

        if relative_gain < 0:
            self._neg_gain_count += 1

        # ── 新 skip 判断：只跳过损失恶化超过阈值的 episode ──────
        if relative_gain < skip_thresh:
            self._skip_count += 1
            return loss_after, float(np.clip(relative_gain, -1., 1.)), 1.0, True
        # ────────────────────────────────────────────────────────

        # gain 在 [skip_thresh, 0)：允许更新，weight=1.0（标准 Reptile 步）
        # gain 在 [0, 1]：难度加权，gain 越大 weight 越小（防止过适应）
        clipped_gain = float(np.clip(relative_gain, 0., 1.))
        weight       = min(1.0 - clipped_gain, 0.85)

        with torch.no_grad():
            for p_meta, p_fast in zip(self.model.parameters(),
                                       fast_model.parameters()):
                p_meta.data += (self.cfg['lr_outer'] * weight
                                * (p_fast.data - p_meta.data))

        return loss_after, clipped_gain, weight, False

    def train(self):
        print(f"\n{'='*60}")
        print("Phase1：MetaHFC3 Warm-Start 元学习 v2.1")
        print(f"  skip_thresh={self.cfg['skip_thresh']} | "
              f"lr_outer={self.cfg['lr_outer']}")
        print(f"{'='*60}\n")

        best_loss = float('inf')
        start_ep  = RESUME_START_EP if RESUME_CKPT is not None else 1

        loss_before_window = []
        loss_after_window  = []

        if self._saved_modality_counts is not None:
            modality_counts = self._saved_modality_counts
            print(f"[续跑] modality_counts 已恢复: {modality_counts}")
        else:
            modality_counts = {m: 0 for m in self.sampler.active_modalities}

        for episode in range(start_ep, self.cfg['n_episodes'] + 1):
            t0 = time.time()

            (s_norm, s_mask,
             q_norm, q_mask,
             stat_vec, domain_id, domain_name) = self.sampler.sample_episode()

            modality = self.sampler.MODALITY_MAP.get(domain_name, domain_name)
            modality_counts[modality] += 1

            # loss_before（元模型在 query set 上的当前损失）
            self.model.eval()
            stat_vec_dev = stat_vec.to(self.device)
            with torch.no_grad():
                seg_before, _ = self.model(
                    q_norm.to(self.device),
                    s=stat_vec_dev, hfc_weight=1.0, return_domain=True
                )
                loss_before = self.seg_loss(
                    seg_before, q_mask.to(self.device)
                ).item()

            # 内循环（差异化 lr）
            self.model.train()
            fast_model, inner_lr = self._inner_loop(
                s_norm, s_mask, domain_id, stat_vec, domain_name
            )

            # 外循环（新 skip 阈值）
            loss_after, rel_gain, weight, skipped = \
                self._difficulty_aware_update(
                    fast_model, q_norm, q_mask, stat_vec, loss_before
                )

            elapsed = time.time() - t0

            loss_before_window.append(loss_before)
            loss_after_window.append(loss_after)
            if len(loss_before_window) > 100:
                loss_before_window.pop(0)
            if len(loss_after_window) > 100:
                loss_after_window.pop(0)

            avg_before = np.mean(loss_before_window)
            avg_after  = np.mean(loss_after_window)

            with open(self.log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    episode, domain_name, modality,
                    f"{loss_before:.4f}", f"{loss_after:.4f}",
                    f"{rel_gain:.4f}", f"{weight:.4f}", int(skipped),
                    f"{avg_before:.4f}", f"{avg_after:.4f}",
                    self._neg_gain_count, self._skip_count,
                    f"{elapsed:.2f}",
                    f"{inner_lr:.0e}",
                ])

            if episode % self.cfg['log_interval'] == 0:
                # skip_rate 基于续跑后的总 episode 数计算
                episodes_since_start = episode - (RESUME_START_EP - 1) \
                    if RESUME_CKPT is not None else episode
                # 用总 episode 数计算累计比率（含旧计数，仅供参考）
                neg_rate  = self._neg_gain_count / episode * 100
                skip_rate = self._skip_count     / episode * 100

                total_sampled = sum(modality_counts.values())
                mod_dist = " | ".join(
                    f"{m}:{c/total_sampled*100:.1f}%"
                    for m, c in modality_counts.items()
                ) if total_sampled > 0 else "-"

                skip_tag   = " [SKIP]" if skipped else ""
                enc_tag    = " [ENC_LR]" if domain_name in ENDOSCOPY_DOMAINS else ""
                print(f"[Ep {episode:5d}/{self.cfg['n_episodes']}] "
                      f"域={domain_name:10s}{enc_tag} | "
                      f"q_loss:{loss_before:.4f}→{loss_after:.4f} | "
                      f"gain={rel_gain:.3f} w={weight:.3f}{skip_tag} | "
                      f"avg100={avg_before:.4f} | "
                      f"neg={self._neg_gain_count}({neg_rate:.1f}%) "
                      f"skip={self._skip_count}({skip_rate:.1f}%) | "
                      f"ilr={inner_lr:.0e} | t={elapsed:.1f}s")
                print(f"         模态分布: {mod_dist}")

            # Checkpoint
            if episode % self.cfg['save_interval'] == 0:
                ckpt = {
                    'episode':         episode,
                    'model':           self.model.state_dict(),
                    'global_mean':     self.global_mean.cpu(),
                    'global_std':      self.global_std.cpu(),
                    'domain_means':    {k: v.cpu() for k, v in
                                        self.domain_means.items()},
                    'avg_loss':        avg_before,
                    'avg_loss_after':  avg_after,
                    'neg_gain_count':  self._neg_gain_count,
                    'skip_count':      self._skip_count,
                    'active_domains':  self.sampler.active_domains,
                    'modality_counts': modality_counts,
                    'sampling':        'modality_balanced_warmstart_v2.1',
                    'cfg':             self.cfg,
                }
                ckpt_path = self.save_dir / f"meta_ep{episode}.pth"
                torch.save(ckpt, ckpt_path)
                print(f"  [保存] {ckpt_path.name} | "
                      f"avg100_before={avg_before:.4f} | "
                      f"avg100_after={avg_after:.4f}")

                if avg_before < best_loss:
                    best_loss = avg_before
                    torch.save(ckpt, self.save_dir / 'meta_best.pth')
                    print(f"  *** Ep {episode}: 新最佳 "
                          f"avg100_before={best_loss:.4f} ✅")

        # 最终保存
        torch.save({
            'episode':         self.cfg['n_episodes'],
            'model':           self.model.state_dict(),
            'global_mean':     self.global_mean.cpu(),
            'global_std':      self.global_std.cpu(),
            'domain_means':    {k: v.cpu() for k, v in
                                self.domain_means.items()},
            'neg_gain_count':  self._neg_gain_count,
            'skip_count':      self._skip_count,
            'active_domains':  self.sampler.active_domains,
            'modality_counts': modality_counts,
            'sampling':        'modality_balanced_warmstart_v2.1',
            'cfg':             self.cfg,
        }, self.save_dir / 'meta_final.pth')

        print(f"\n[完成] MetaHFC3 v2.1 保存至: {self.save_dir}")
        total = self.cfg['n_episodes']
        for mod, cnt in modality_counts.items():
            print(f"  {mod:12s}: {cnt} ({cnt/total*100:.1f}%)")
        print(f"  总负增益: {self._neg_gain_count}/{total} "
              f"({self._neg_gain_count/total*100:.1f}%)")
        print(f"  总跳过数: {self._skip_count}/{total} "
              f"({self._skip_count/total*100:.1f}%)")
        return self.model


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    from config import META_TRAIN_DOMAINS, validate_paths
    if not validate_paths():
        import sys
        sys.exit(1)

    if RESUME_CKPT is not None:
        if not Path(RESUME_CKPT).exists():
            print(f"[错误] 续跑 checkpoint 不存在: {RESUME_CKPT}")
            import sys
            sys.exit(1)
        print(f"[模式] 续跑：从 Ep {RESUME_START_EP} 开始")
    else:
        if not KVASIR_CKPT.exists():
            print(f"[错误] Kvasir 预训练权重不存在: {KVASIR_CKPT}")
            import sys
            sys.exit(1)
        print("[模式] 全新训练：从 Kvasir Warm-Start 开始")

    trainer = MetaTrainerV2(cfg=META_CONFIG_V2, model_cfg=MODEL_CONFIG_V2)
    trainer.train()
