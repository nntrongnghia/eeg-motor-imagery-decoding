from typing import Dict
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_focal_loss(pred_logits, targets, gamma=2, alpha=0.25):
    p = torch.softmax(pred_logits, -1)
    targets = F.one_hot(targets.to(torch.int64), num_classes=pred_logits.shape[-1]).to(torch.float)
    ce_loss = F.binary_cross_entropy_with_logits(
        pred_logits, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()*10

def ce_loss(m_outputs, targets):
    logits = m_outputs["logits"]    
    return F.cross_entropy(logits, targets)


class SmoothCECenterLoss:
    def __init__(self, nb_classes=4, smooth_factor=0.5, center_factor=0.5) -> None:
        self.nb_classes = nb_classes
        self.smooth_factor = smooth_factor
        self.center_factor = center_factor
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=smooth_factor)

    def __call__(self, m_outputs: Dict[str, torch.Tensor], targets: torch.Tensor):
        ce_loss = self.ce_loss_fn(m_outputs["logits"], targets)
        # center loss
        if self.center_factor != 0:
            center_loss = 0
            ft = m_outputs["features"]
            for c in range(self.nb_classes):
                xc = ft[targets == c]
                if len(xc) > 0:
                    center_loss += ((xc - xc.mean(0, keepdim=True))**2).mean()
            return ce_loss + self.center_factor*center_loss
        else:
            return ce_loss