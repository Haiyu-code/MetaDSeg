# dataset.py
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T


def augment_pair(img, mask):
    """
    配对数据增强（PIL Image → PIL Image）
    不使用 ColorJitter，保持 LAB 统计与 HFC 向量一致性
    """
    if random.random() > 0.5:
        img  = TF.hflip(img)
        mask = TF.hflip(mask)
    if random.random() > 0.5:
        img  = TF.vflip(img)
        mask = TF.vflip(mask)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        img   = TF.rotate(img,  angle,
                          interpolation=T.InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask, angle,
                          interpolation=T.InterpolationMode.NEAREST)
    if random.random() > 0.5:
        img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
    if random.random() > 0.5:
        img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
    return img, mask


def find_mask(stem, mask_dir, mask_exts):
    """
    按优先级查找 mask 文件：
      1. 精准匹配（支持标准命名及常规变体）
      2. 模糊匹配（只要 mask 文件名以原图 stem 开头即可，解决一切非常规命名和隐藏字符）
    """
    # 1. 优先进行精准匹配
    for ext in mask_exts:
        for suffix in ['', '_Segmentation', '_segmentation', '_mask', '_Mask']:
            c = mask_dir / f"{stem}{suffix}.{ext}"
            if c.exists():
                return c
                
    # 2. 兜底模糊匹配（终极杀手锏：自动寻找以 stem 开头的任意图片）
    for ext in mask_exts:
        # 使用通配符查找，例如查找 "ISIC_0024306*.png"
        matches = list(mask_dir.glob(f"{stem}*.{ext}"))
        if matches:
            return matches[0]  # 返回匹配到的第一个文件
            
    return None


class SegDataset(Dataset):
    """
    通用分割数据集
    支持多种 mask 命名规则（标准 / ISIC 大写S / ISIC 小写s）
    """
    IMG_EXTS  = ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'PNG', 'BMP']
    MASK_EXTS = ['png', 'jpg', 'bmp', 'PNG', 'BMP']

    def __init__(self, img_dir, mask_dir,
                 img_size=352, is_train=True, return_01=False):
        self.img_dir   = Path(img_dir)
        self.mask_dir  = Path(mask_dir)
        self.img_size  = img_size
        self.is_train  = is_train
        self.return_01 = return_01
        self.pairs     = self._match_pairs()

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"在 {img_dir} 中找不到有效的图像-mask对\n"
                f"masks目录: {mask_dir}\n"
                f"请检查文件命名（支持：stem.png / "
                f"stem_Segmentation.png / stem_segmentation.png）"
            )

    def _match_pairs(self):
        imgs = []
        for ext in self.IMG_EXTS:
            imgs.extend(self.img_dir.glob(f'*.{ext}'))
        imgs = sorted(set(imgs))

        pairs = []
        for img_path in imgs:
            found = find_mask(img_path.stem, self.mask_dir, self.MASK_EXTS)
            if found:
                pairs.append((img_path, found))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        img  = img.resize( (self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.is_train:
            img, mask = augment_pair(img, mask)

        img_01 = TF.to_tensor(img)
        img_norm = TF.normalize(
            img_01.clone(),
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        )
        mask_t = torch.from_numpy(
            np.array(mask, dtype=np.float32)
        ).unsqueeze(0) / 255.0
        mask_t = (mask_t > 0.5).float()

        result = {
            'image':    img_norm,
            'mask':     mask_t,
            'img_path': str(img_path),
        }
        if self.return_01:
            result['image_01'] = img_01
        return result
