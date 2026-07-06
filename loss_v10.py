import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureLoss(nn.Module):
    """
    混合损失函数: BCE With Logits + Dice Loss
    这种组合在医学图像分割中非常鲁棒 (参考 U-Net_v2)
    """
    def __init__(self):
        super(StructureLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, mask):
        # 1. BCE Loss
        wbce = self.bce(pred, mask)
        
        # 2. Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        inter = (pred_sigmoid * mask).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
        dice = 1 - (2. * inter + 1e-5) / (union + 1e-5)
        
        return wbce + dice.mean()

class CombinedLossV10(nn.Module):
    def __init__(self):
        super(CombinedLossV10, self).__init__()
        self.loss_func = StructureLoss()

    def forward(self, outputs, target):
        """
        计算主输出和所有辅助输出的 Loss
        """
        # 主输出 Loss
        loss_main = self.loss_func(outputs['main_seg'], target)
        
        # 辅助输出 Loss (深监督)
        # 需要将辅助输出上采样到 GT 尺寸
        loss_aux = 0
        aux_weights = [0.2, 0.3, 0.4] # 越浅层权重越大
        
        if 'aux1' in outputs:
            pred_aux1 = F.interpolate(outputs['aux1'], size=target.shape[2:], mode='bilinear', align_corners=True)
            loss_aux += aux_weights[0] * self.loss_func(pred_aux1, target)
            
        if 'aux2' in outputs:
            pred_aux2 = F.interpolate(outputs['aux2'], size=target.shape[2:], mode='bilinear', align_corners=True)
            loss_aux += aux_weights[1] * self.loss_func(pred_aux2, target)
            
        if 'aux3' in outputs:
            pred_aux3 = F.interpolate(outputs['aux3'], size=target.shape[2:], mode='bilinear', align_corners=True)
            loss_aux += aux_weights[2] * self.loss_func(pred_aux3, target)
            
        return loss_main + loss_aux